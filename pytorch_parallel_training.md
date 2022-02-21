### 参考资料
1. [detr](https://github.com/facebookresearch/detr)
2. [pytorch-distributed](https://github.com/tczhangzhi/pytorch-distributed/blob/master/distributed.py)

### 基本概念
1. word_size: 总的process的个数
2. rank: the index of process in the network
3. local_rank: the index of process on the local machine

假设我们有2个node(机器)， 每个node上面有2张GPU，一共有4个process, 那么

| |   |  |  | |
|----|----|----|----|----|
| local_rank | 0 | 1 | 0 | 1 |
|----|----|----|----|----|
|rank | 0 | 1 | 2 | 3|

### torch.distributed.launch 用法
#### torch.distributed.launch 参数解析
	nproc_per_node=N 表示一个节点上有N张显卡
	nnodes 节点的个数
	node_rank 指节点的编号
		比如上例中在机器A上启动时,node_rank=0, 指A机器的节点编号是0
		在机器B上启动时,node_rank=1，指B机器上的节点编号是1
	master_addr master节点的ip地址
	master_por master节点的port号, master节点的port号
在不同的节点上master_addr和master_port的设置是一样的，用来进行通信

#### torch.distributed.launch 环境变量解析
torch.distributed.launch的一个作用是，会把参数转成环境变量

	WORLD_SIZE: 通俗的解释下，就是一共有多少个进程参与训练， WORLD_SIZE = nproc_per_node*nnodes,不同的进程中，WORLD_SIZE是唯一的
	
	RANK：进程的唯一表示符，不同的进程中，这个值是不同的，上述在AB两台机器上共启动了8个进程，则不同进程的RANK号是不同的 
	
	LOCAL_RANK： 同一节点下，LOCAL_RANK是不同的，常根据LOCAL_RANK来指定GPU，但GPU跟LOCAL_RANK不一定一一对应，因为进程不一定被限制在同一块GPU上
	

#### torch.distributed.launch 使用方法
distributed.launch只要启动，就会启动world_size个process, 每一个process里面的rank, local_rank都是不同的。举个一个例子
```
source /home/luban/anaconda3/bin/activate working
python -m torch.distributed.launch \
        --nproc_per_node=4 \
        --use_env \
        --master_port 1324 train.py &> train_log.out
```
其中train.py
```
import os
import torch

if __name__=="__main__":
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        print("KKK:\t", "rank: ", rank, " world_size: ", world_size, " gpu: ", gpu)
```
输出结果是
```
KKK:     rank:  2  world_size:  4  gpu:  2
KKK:     rank:  0  world_size:  4  gpu:  0
KKK:     rank:  1  world_size:  4  gpu:  1
KKK:     rank:  3  world_size:  4  gpu:  3
```

### Dataloader and Sampler

```
train_dataset = datasets.ImageFolder( #----拿到数据
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
)

#----使用Samler采样器
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

#----dataloader
train_loader = torch.utils.data.DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       num_workers=num_worker,
                                       pin_memory=True,
                                       sampler=train_sampler)
```
假设我们是单机N个GPU，总的batch_size = single_batch_size * N
1. 使用DistributedSampler, 一个epoch里面只有一份数据集，每次N张卡都拿到single_batch_size数据，每张卡上的数据是不同的
2. 不使用DistributedSampler, 一个epoch里面有N份数据集，即每张卡都维护自己的一份完整的数据集, 每张卡上的数据可能会存在重复。同时这里需要注意，如果DataLoader里面不使用Sampler,那么上面代码中的batch_size=args.batch_size*N

但是一个很重要的地方是需要注意seed的设置方法, [detr](https://github.com/facebookresearch/detr/issues/297)里面提到，给每一个process设置不同的seed, 这样每一个process里面的数据增强变换都是不一样的
```
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
```

### problems

#### batch_size和lr的设定
我们假设这样的场景, 单卡的时候batch_size = single_batch_size, lr=single_lr, 那么我N卡的时候，总的batch_size=single_batch_size * N, lr=single_lr*N, 那么此时
1. dataloader里面的batch_size应该怎么设置? 
2. optimizer里面的lr应该怎么设置？

答：
1. 根据上面的讨论, 在Dataloader里面，如果使用sampler, 那么batch_size=single_batch_size; 如果没有使用sampler, 那么batch_size=single_batch_size * N
2. optimizer里面的lr和单卡的情况保持一致即可

#### load pretrained model
如何load pretrained model? 全部的process加载参数还是rank=0的process加载参数即可?
答:
我写过一个程序，使用rank==0的process加载之前训练的参数，发现只有process=0正确加载了参数，其他的process没有加载成功。正确的方法是每一个process都要加载模型参数，但是保存的时候可以只用rank=0的process保存。

### tricks

#### print和save
在并行训练的情况下，每一个process(运行在每一张显卡上的程序)都会print和save, 会造成输出的混乱和save的重复, 这个时候一般使用rank=0的process进行保存
```
import torch.distributed as dist
if dist.get_rank()==0:
    print("xxxx")
    save()
```

#### dist.all_reduce
把不同gpu上面的值求平均, 一般用于输出loss或者一些变量的平均值到日志里面
```
import torch.distributed as dist
def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
```


