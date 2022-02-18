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

### problems


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


