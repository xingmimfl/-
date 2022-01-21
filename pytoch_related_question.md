#### pytorch 训练Imagenet中默认读取的图片是RGB还是BGR?

在官方给的训练Imagenet的模型中，读取ImageNet数据用的是datasets.ImageFolder这接口.ImageFolder默认的image loader是PIL格式下的RGB, [0, 255]


#### 不要轻易使用Python list存储pytorch tensor / numpy. 因为存储的内容可能会发生变化。


#### 多卡训练保存的模型，在保存成pth之后读取出现问题
```
    #———保存pth的代码
    file_name = “modellandmarks_dense_windloss_righteye_ychannel_step1_to_step2_no_fine/
                    landmarks_dense_windloss_righteye_ychannel_step1_to_step2_no_fine_iter_764800_.model"
    model = torch.load(file_name)
    model = model.cuda(0)
    torch.save(model.state_dict(), "landmarks_dense_windloss_righteye_ychannel_step1_to_step2_no_fine_iter_764800_.pth")

    #——读取pth的代码
    eyebrow_model = model.LandmarksModel()
    eyebrow_model.load_state_dict(torch.load("landmarks_dense_windloss_righteye_ychannel_step1_to_step2_no_fine_iter_764800_.pth"))
    eyebrow_model.eval()
```
出现的问题
```
KeyError: 'unexpected key "module._preprocess_layer.0.0.weight" in state_dict'
```
这是因为模型.model使用多卡保存的，所以会自动添加一个module的模块。

https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3

#### pytorch child的问题




#### pytorch torchvision colorJitter/ pytorch中的数据增强
ColorJitter 在0.3之后开始出现
```
self.to_tensor = transforms.Compose([
transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
self.trans_train = Compose([
    ColorJitter(
        brightness = 0.5,
        contrast = 0.5,
        saturation = 0.5),
    HorizontalFlip(),
    RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
    RandomCrop(cropsize)
])
```
https://discuss.pytorch.org/t/data-augmentation-for-segmentation-task/18692
ee
这里面有一个问题，就是HorizontalFlip， RandomScale，RandomCrop如何应用到segmentation任务中，即这些函数也要对mask image做同样的操作。

https://raw.githubusercontent.com/CoinCheung/BiSeNet/master/transform.py 上面的代码是从这个里面来的，作者是自己实现了一套变换的方法
```
#!/usr/bin/python
# -*- encoding: utf-8 -*-

from PIL import Image
import PIL.ImageEnhance as ImageEnhance
import random

class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size
    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        if (W, H) == (w, h): return dict(im=im, lb=lb)
        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return dict(
                im = im.crop(crop),
                lb = lb.crop(crop)
                    )
class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p


    def __call__(self, im_lb):
        if random.random() > self.p:
            return im_lb
        else:
            im = im_lb['im']
            lb = im_lb['lb']
            return dict(im = im.transpose(Image.FLIP_LEFT_RIGHT),
                        lb = lb.transpose(Image.FLIP_LEFT_RIGHT),
                    )
class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales
    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return dict(im = im.resize((w, h), Image.BILINEAR),
                    lb = lb.resize((w, h), Image.NEAREST),
                )
class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]


    def __call__(self, im_lb):
        im = im_lb['im']
        lb = im_lb['lb']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        return dict(im = im,
                    lb = lb,
                )

class MultiScale(object):
    def __init__(self, scales):
        self.scales = scales


    def __call__(self, img):
        W, H = img.size
        sizes = [(int(W*ratio), int(H*ratio)) for ratio in self.scales]
        imgs = []
        [imgs.append(img.resize(size, Image.BILINEAR)) for size in sizes]
        return imgs

class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list


    def __call__(self, im_lb):
        for comp in self.do_list:
            im_lb = comp(im_lb)
        return im_lb
if __name__ == '__main__':
    flip = HorizontalFlip(p = 1)
    crop = RandomCrop((321, 321))
    rscales = RandomScale((0.75, 1.0, 1.5, 1.75, 2.0))
    img = Image.open('data/img.jpg')
    lb = Image.open('data/label.png')
```

#### pytorch 对不同的layer使用不同的learning rate
```
from torch.optim import Adam
model = SomeKindOfModel()
optim = Adam(
    [
        {"params": model.fc.parameters(), "lr": 1e-3},
        {"params": model.agroupoflayer.parameters()},
        {"params": model.lastlayer.parameters(), "lr": 4e-2},
    ],
    lr=5e-4,
)
```
 Other parameters which are didn't specify in optimizer will not optimize. So You should state all layers or groups(OR the layers you want to optimize). and if you didn't specify the learning rate it will take the global learning rate(5e-4). The trick is when you create the model you should give names to the layers or you can group it. 这里一定要注意，没有在group中出现的layer对一个的参数是不更新的。

#### pytorch固定layer参数

一种固定前几层layer参数的方法是
```
pretrained_state = torch.load(pretrained_model)

net.resnet.load_state_dict({k:v for k, v in pretrained_state.items() if k in net.resnet.state_dict()})
for p in net.resnet.conv1.parameters(): p.requires_grad=False
for p in net.resnet.bn1.parameters(): p.requires_grad=False
for p in net.resnet.layer1.parameters(): p.requires_grad=False
for p in net.resnet.layer2.parameters(): p.requires_grad=False

params = []
#params = list(net.parameters())
for p in list(net.parameters()):
if p.requires_grad == False: continue
   params.append(p)
```
如果根据layer的名字来固定layer的参数，可以这么做
```
    resnet_params = []
    model_other_params = []
    for name, param in model.named_parameters():
        if "res" in name:
            resnet_params.append(param)
        else:
            model_other_params.append(param)
    optimizer = torch.optim.RMSprop([
        {'params':resnet_params, 'lr':LEARNING_RATE * 0.1},
        {'params':model_other_params}],
        lr = LEARNING_RATE)
```
#### pytorch  weightedrandomsampler的作用


#### pytorch指定使用cuda的一种方法
```
self.device = torch.device('cuda:%d' % self.device_ids[0] if (torch.cuda.is_available() and opt.ngpu > 0) else "cpu”)
...
self.landmarks_model = FaceLandmarkModel6M(input_channel=3, landmark_scale=self.opt.landmark_scale).to(self.device)
```
我感觉这么做还是挺省事的，在之后不用显示指出来用GPU还是用CPU


#### pytorch 多卡训练
pytorch有一机多卡和多机多卡两种分布式训练方式，分别使用
DataParallel is for performing training on multiple GPUs, single machine 605.
DistributedDataParallel is useful when you want to use multiple machines 822.
DataParallel和DistributedDataParallel来实现

在实践中，我发现loss可能会集中在一个GPU上计算，导致这个GPU的显存不够，从而导致程序失败。


#### pytorch Mean 和 Std的问题
inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
这里参考的是https://github.com/xingyizhou/CenterNet 这个里面的，我猜测transform里面的变换和这个是一样的。


#### pytorch 测量运行时间
```
torch.cuda.synchronize()
start = time.time()
result = model(input)
torch.cuda.synchronize()
end = time.time()
```
其中需要用torch.cuda.synchronize 对运行时间做一个对齐。

#### pytorch循环读取dataset, 如果循环之前数据就读完的话，就重新开始循环
```
try:
    _images, _bbox, _labels, _angle_labels, _image_paths = train_iter.next()
except StopIteration:
    train_iter = iter(train_loader) #——重新开始循环
    _images, _bbox, _labels, _angle_labels, _image_paths = train_iter.next()
```

#### pytorch flops计算
 - https://github.com/Lyken17/pytorch-OpCounter
 - https://github.com/ShichenLiu/CondenseNet/blob/master/utils.py

#### OrderedDict的使用
OrderedDict其实是模块collections的类型，但是pytorch的参数都是保存在OrderedDict, 这个Dict和普通的dict不太一样。
```
import os
import sys
from collections import OrderedDict

if __name__=="__main__":
    the_dict = OrderedDict()
    the_dict['A'] = 1
    the_dict['B'] = 2
    the_dict['C'] = 3
    for a_elem in the_dict: #——注意在这个地方，直接遍历给出的是key
        print(a_elem)
    for key, value in the_dict.items():
        print(key, "\t", value)
```
好像普通的dict也支持这么做。看来是我记错了。

#### pytorch Image 和 cv2的方法

pytorch里面默认使用的PIL.image的方法读取图片
PIL.image 方法读取的是RGB，0~255； cv2是BGR, 0~255的方法

#### pytorch toTensor 和 normalize的方法
toTensor的输入要求必须是PIL读取的图片，即输入是0~255， RGB通道。同时要注意，如果是numpy的话，必须是.astype(np.uint8), 否则最后输出的值不是[0,1]范围内的，这点很容易出错。

toTensor把输入直接除以255， 把图片归一化到[0,1]; 同时把HxWxC变换到 CxHxW

Normalize 则是把toTensor之后剪去mean, 除以std

假如我们要把图像从
```
toTensor
Normalize
```
反变换回来，那么我们可以这么做

#### pytorch中squeeze的问题
我之前用resnet50,  回归关键点。我把resnet50的最后的fc层改成了nn.Conv2d,   最后的输出是
```
output.size:     torch.Size([2, 64, 1, 1])
```
我使用的wing loss, 模型一直不收敛
后来，我用了一个squeeze(), 把多余的通道去掉，最后loss收敛了，而且收敛的速度很快，非常奇怪

我刚才试了一下，torch1.2中，squeeze或者不squeeze，输出的结果是不一样的
```
a = torch.rand((2,3,1,1,))
b = torch.rand((2,3))
c = a-b  #torch.Size([2, 3, 2, 3])
d = a.squeeze() - b #torch.Size([2, 3])
```

#### pytorch固定BN层参数

固定参数的时候，一定要注意把batch norm一起固定死，因为batch norm的mean, var是动态计算的
```
def get_layers_dict(model):
    named_layers = OrderedDict()
    for nm, module in model.named_modules():
        if len(list(module.children())) == 0:
            named_layers["{}:{}".format(nm, module.__class__.__name__)] = module
    return named_layers

def freeze_params(model, module_names=["_preprocess_layer", "_second_stage", "_landmarks_top"],freeze_bn=True):
    idx = 0
    layers_dict = get_layers_dict(model)
    for layer_nm, layer in layers_dict.items():
        module_nm = layer_nm.strip().split(".")[idx]
        if module_nm in module_names:
            if freeze_bn and (isinstance(layer, (torch.nn.BatchNorm2d, apex.parallel.optimized_sync_batchnorm.SyncBatchNorm))):
                layer.eval()
            for p in layer.parameters():
                p.requires_grad = False

def freeze_bn(model, module_names=["_preprocess_layer", "_second_stage", "_landmarks_top"]):
    idx = 0
    layers_dict = get_layers_dict(model)
    for layer_nm, layer in layers_dict.items():
        module_nm = layer_nm.strip().split(".")[idx]
        if module_nm in module_names:
            if (isinstance(layer, (torch.nn.BatchNorm2d, apex.parallel.optimized_sync_batchnorm.SyncBatchNorm))):
                layer.eval()
```

下面的代码可以比较参数是否改变
```
import os
import sys
import torch
import model

if __name__=="__main__":
    model = model.LandmarksModel()
    state_dict = model.state_dict()
    pretrained_model_pth = "model_float_version1_iter_156800_.pth"
    pretrained_state_dict = torch.load(pretrained_model_pth)

    for key, value in pretrained_state_dict.items():
        if key in state_dict.keys():
            print(key)
            state_dict[key] = value
    model.load_state_dict(state_dict)
    torch.save(model.state_dict(), "model_156800_psedo_classification.pth")

    for nm, module in landmarks_model.named_modules():
        if len(list(module.children())) == 0:
            layer_name = "{}:{}".format(nm, module.__class__.__name__)
            if "classification_top" in nm: continue
            for p in module.parameters(): #---冻结参数
                print("freeze param:\t", layer_name)
                p.requires_grad = False
            if isinstance(module, torch.nn.BatchNorm2d): #----冻结bn layer
                print("freeze bn:\t", layer_name)
                module.eval()
```

#### pytorch 恢复优化器，接着训练

首先保存优化器，同时要注意，scheduler要使用绿框中的方法重载，而不是下面红框中的，否则lr还是config.py里面的初始lr


#### 删掉多余的layer
```
# create ResNet-50 backbone
self.backbone = resnet50()
del self.backbone.fc
```

#### pytorch 不计算gradients
```
@torch.no_grad()
def forward(self, outputs, targets):
```

#### pytorch卷积的计算量
dilation不占用计算量


#### timm
一个开源的pytorch的包，里面包含了很多的layer. 在Swin-transformer里面使用了这个库

#### pytorch optimizier初始化的方

来自Swin-transformer， 里面用到了timm这个库。这个里面包含了warm_up这个方法


#### Pytorch3d运行错误。
这个错误是这样的：我在我本地跑程序没有问题，但是在鲁班系统上跑程序一直报错。错误如下
```
File "/home/luban/anaconda3/envs/working/lib/python3.6/site-packages/pytorch3d-0.6.0-py3.6-linux-x86_64.egg/pytorch3d/renderer/mesh/rasterize_meshes.py", line 320, in forward
cull_backfaces,
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
我后来去了github上面的pytorch3d查询了一番，解决方案如下，在编译pytorch3d的时候，在setup.py里面加上下面的话
```
os.environ["NVCC_FLAGS"]="-gencode=arch=compute_35,code=sm_35 \
    -gencode=arch=compute_50,code=sm_50 \
    -gencode=arch=compute_60,code=sm_60 \
    -gencode=arch=compute_70,code=sm_70 \
    -gencode=arch=compute_75,code=sm_75 \
    -gencode=arch=compute_50,code=compute_50"
```
重新编译之后即可。


#### smooth label

https://github.com/pytorch/pytorch/issues/7455
```
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
```

#### pytorch做图像放射变换
- https://www.jianshu.com/p/723af68beb2e
- https://zhuanlan.zhihu.com/p/349741938
- https://blog.csdn.net/Jee_King/article/details/107876429

可以直接使用现成的包，比如kornia

https://kornia.readthedocs.io/en/latest/geometry.transform.html?highlight=warp_affine#kornia.geometry.transform.warp_affine

### 已知两组关键点，求转动矩阵
- http://nghiaho.com/?page_id=671
- https://math.stackexchange.com/questions/188442/rotation-matrix-for-a-set-of-points
