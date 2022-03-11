### 一些参考
- https://hub.fastgit.org/zldrobit/yolov5/blob/tf-android/models/tf.py 重点推荐这个
- https://github.com/qqwweee/keras-yolo3  参考这里如何处理dataset
- https://github.com/eriklindernoren/Keras-GAN
- https://github.com/qubvel/segmentation_models

keras是一个极其灵活的框架，我在这里提到的仅仅是我在实践中碰到的问题和解决方法，不是唯一标准的做法。

### keras tensor操作
#### keras.conv2D没有groups
keras=2.3.1之前没有这个参数。
	
```python
TypeError: ('Keyword argument not understood:', 'groups')
```
解决方法，升级keras版本，或者使用其他的卷积核操作。

#### Keras动态reshape
```python
class Reorg(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Reorg, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        #ss = x.shape
        #print(ss)
        #out = keras.layers.Reshape((self.width, self.height, self.channel))(x)
        out = tf.nn.space_to_depth(x,2,data_format='NHWC')
        return out
```
这里我想不起来当时为什么用这个，理论上其他的reshape layer都可以实现对应的功能，但是总有一些奇奇怪怪的问题.

####  keras squeeze方法
```python
tf.keras.backend.squeeze(out, axis=1)
```

#### keras concatenate方法
```
out = keras.backend.concatenate([out, classify_out], axis=1)
```

#### keras 打印tensor的值
```
Tensorflow.keras.backend.get_value(tensor)
```

### keras fit/fit_generator/train_on_batch
keras有三个可以用来训练的函数：
- fit
- fit_generator
- train_on_batch
如果想要自己写dataset, 那么需要使用后面两种方法
train_on_batch 
在tensorflow2.3中，fit_generator已经设置成了deprecated
[doc1](https://github.com/rstudio/keras/issues/1104)/[doc2](https://keras.io/api/models/model_training_apis/)/[doc3](https://stackoverflow.com/questions/49100556/what-is-the-use-of-train-on-batch-in-keras)/[doc4](https://blog.csdn.net/baoxin1100/article/details/107917633)/[dcgan](https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py)这些需要看下,介绍了fit_generator和train_on_batch

如果使用fit_generator的方法
```
landmarks_model.fit_generator(face_dataset.generate_data(), steps_per_epoch=steps,epochs=20)
```

### keras GradientTape方法
上面提到的fit/fit_generator/train_on_batch的方法，训练过程都是不可见的。但是在真正的工作中，我们往往需要自己进行精细化的控制，这个时候就推荐使用GradientTape方法，可以显示写出来训练过程，如下

```
for  i in range(MAX_EPOCH):  #---epoch
    epoch_time = time.time()
    for i_batch, sample_batched in enumerate(train_loader): #--iter.注意这里train_loader是我用pytorch方法写的
        with tf.GradientTape() as tape:
            image_data, pts_data = sample_batched[0], sample_batched[1]
            image_data = image_data.permute(0,2,3,1)
            image_data = image_data.numpy().astype(np.float32) #--这里转成numpy()的原因是，dataloader我用了pytorch的方法做的
            pts_data = pts_data.numpy().astype(np.float32)

            landmarks_out = landmarks_model(image_data)
            loss = wing_loss(landmarks_out, pts_data) #---这里的loss是我们自定义的
            mae = calc_accuracy(landmarks_out, pts_data)

        grads = tape.gradient(loss, landmarks_model.trainable_weights)
        #grads = tape.gradient(loss, landmarks_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, landmarks_model.trainable_weights))
        #optimizer.apply_gradients(zip(grads, landmarks_model.trainable_variables)) #--variables也可以
```

### keras如何使用自己的loss函数
在这里，我只用GradientTape方法为例，其他的fit/fit_generator/train_on_batch未经验证

```
#----define loss----
def mse_loss(pr, gt): #---定义loss函数
    loss = keras.backend.mean(tf.keras.losses.MSE(pr,gt))
    return loss

#----create optimize---
optimizer = optim.RMSprop(lr=LEARNING_RATE,rho=0.99, epsilon=1e-8)
landmarks_model.compile(loss=wing_loss, optimizer=optimizer) #----这个地方,loss=wing_loss似乎可以不用写，我记得没写的时候运行结果也是正确的

#---training----
for  i in range(MAX_EPOCH):
    epoch_time = time.time()
    for i_batch, sample_batched in enumerate(train_loader):
        with tf.GradientTape() as tape:
            ...
            ...
            loss =wing_loss(input, label) #---这个地方使用自定义的loss函数
            ...
            ...
```

### keras中的dataloader

I) 如果使用fit_generator的方法
```python
landmarks_model.fit_generator(face_dataset.generate_data(), steps_per_epoch=steps,epochs=20)
```
Dataset yield应该这么写
```
def generate_data(self,):
    n = len(self.pair_vec)
    i=0;
    while True:
        image_data = []
        pts_data = []
        for j in range(self.batch_size):
            a_image_path, a_pts_path = self.pair_vec[i]
            a_image = cv2.imread(a_image_path)
            a_pts = self.load_pts(a_pts_path)
            a_image, a_pts = self.crop(a_image, a_pts)
            a_image, a_pts = self.crop(a_image, a_pts)
            a_image = a_image / 255.
            image_data.append(image_data)
            pts_data.append(a_pts)
            i = (i+1) % n
        image_data = np.array(image_data)
        pts_data = np.array(pts_data)
        yield (image_data, pts_data)
```
验证yield的方法如下
```
for i_batch, sample_batched in enumerate(face_dataset.generate_data()):
    image_data, pts_data = sample_batched
    print("image_data.shape:\t", image_data.shape)
    print("pts_data.shape:\t", pts_data.shape
```

II) 我们可以使用pytorch里面的dataloader的方法，这样做的好处是一方面我们很多模型可能是pytorch写的，数据增强都是都是先写好的，另外一方面pytorch也提供了很多优秀的数据增强的方法。在我们的实践中，我们一直是在使用这种方法
```
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn,
                                      shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
...
def train():
    for i,(inputs, label) in enumerate(train_loader):
        #如果 inputs 和 label 是 torch Tensor
        #请用 inputs = inputs.numpy() 和 label = label.numpy() 转成 ndarray
        y_pred = model.train_on_batch(inputs, label) #---这个地方也可以使用GradientTape的方法
```
### keras调整学习率

- [ref1](https://towardsdatascience.com/learning-rate-schedule-in-practice-an-example-with-keras-and-tensorflow-2-0-2f48b2888a0c#:~:text=The%20constant%20learning%20rate%20is%20the%20default%20schedule,SGD%20optimizer%20and%20pass%20the%20argument%20learning_rate%3D0.01%20), 
- [ref2](https://blog.csdn.net/qq_36556893/article/details/103645204)
- [ref3](https://devdocs.io/tensorflow~2.3/keras/callbacks/learningratescheduler)

fit_generator里面使用回调函数的方法调整学习率
```
callbacks = []
lr_scheduler = get_lr_scheduler(args=args)
callbacks.append(lr_scheduler)
 
...
model.fit_generator(train_generator,
                            steps_per_epoch=train_generator.samples // args.batch_size,
                            validation_data=test_generator,
                            validation_steps=test_generator.samples // args.batch_size,
                            workers=args.num_workers,
                            callbacks=callbacks,  # 你的callbacks， 包含了lr_scheduler
                            epochs=args.epochs,
                    )
```
但是对于train_on_batch或者GradientTape方法，我们可以自己定义学习率的变化
multiStep
```
import keras.backend as K
for epoch in range(100):
    train()
    evaluate()
    # 每10个epoch，lr缩小0.1倍
    if epoch%10==0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr) # 获取当前学习率
        lr = lr * 0.1 # 学习率缩小0.1倍
        K.set_value(model.optimizer.lr, lr) # 设置学习率
```
cosine
```
mport keras.backend as K
for epoch in range(100):
    train()
    evaluate()
    #-------adjust learnig rate-----
    lr = K.get_value(landmarks_model.optimizer.lr) # 获取当前学习率
    lr = eta_min + (eta_max - eta_min) * (1 + math.cos(math.pi * i / MAX_EPOCH)) / 2.
    K.set_value(landmarks_model.optimizer.lr, lr) # 设置学习率
```
上面存在一个问题，就是如果我对不同的layer设定了不同的学习率的变化方式，那么set_value这种方法如何改变？？？

### keras 自己写layer

针对小模型的设计，一个通用的方法是，大家共同维护一套基本的layer, 然后用这些layer组建不同的模型。打一个比方，我们写一个自己的AveragePool layer
```
class AvgPool(keras.layers.Layer):
    def __init__(self,kernel_size,stride=None,padding="valid",ceil_model=False, count_include_pad=True, **kwargs):
        super(AvgPool, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_model = ceil_model
        self.count_include_pad = count_include_pad

        self.avgpool = keras.layers.AveragePooling2D(
                pool_size=self.kernel_size, strides=self.stride, padding=self.padding, data_format=None
            )

    def call(self, x, **kwargs):
        out = self.avgpool(x)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'ceil_model': self.ceil_model,
            'count_include_pad': self.count_include_pad,
            'avgpool':self.avgpool,
        })
        return config
```
这里我们首先要继承keras.layers.Layer这个方法。这个有一个函数叫做get_config(), 如果没有这个函数的话,那么保存模型参数h5之后再读取这个h5的时候就会报一些错误
对应的,模型读取pretrained_model的时候要这么做
```
    custom_objects_dicts={
            'conv_layer_no_group': conv_layer_no_group,
            'mobile_unit':mobile_unit,
            'AvgPool':AvgPool,
            ...
            ...
            "AvgQuant":AvgQuant,
            "Cat":Cat,
            "conv_layer":conv_layer,
            "mse_loss":mse_loss
        }
    pretrained_model_path = "20211223_version14_finetune_iter_3_train_i_batch_0_.h5"
    landmark_model.load_weights(pretrained_model_path, by_name=True)
```
上面的conv_layer_no_group等等就是我们自己定义的layer
这里需要注意，我们自定义layer的时候，里面可能会包含多种torch.nn里面的layer,比如conv/ReLU等等各种layer

### keras load_weights by_name/add name to layer
我们可以给构建keras model的layer添加上名字，如下
```
self.down_conv1_1 = mobile_unit(channel_in, channel_in, 2, name = "layer1")
self.down_conv1_2 = mobile_unit(channel_in, channel_in, name = "layer2")
self.down_conv1_3 = mobile_unit(channel_in, channel_in, name = "layer3")
```
这样做的好处有：
- 在之后的处理中，我们可以根据layer的名字对layer做相应的处理，比如说固定layer参数. if 'classify' in layer.name: layer.trainable=False
- 如果我们把当前模型当做一个raw的模型，训练好了这个模型。之后出于其他的目的，我们在原来的模型结构上面增加了很多的其他的layer, 那么我们读取pretrained model parameters的时候，就可以通过
```
pretrained_model_path = "20211223_version14_finetune_iter_3_train_i_batch_0_.h5"
model.load_weights(pretrained_model_path, by_name=True)
```
这种方法把原先的参数读进来

### 构建keras model的方法
这个地方有很多可以说的, keras里面有很多创建model的方法, 
I)方法1，[doc1](https://keras.io/examples/vision/image_classification_from_scratch/)使用x经过不同的layer不断往后迭代
```
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    ...
    ...
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)
```

II)方法2,[doc2](https://keras.io/examples/vision/mnist_convnet/)使用keras.squential的方法进行定义model
```
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
```

III)方法3, 使用keras.Model类的方法建立，如下
```
import tensorflow as tf
class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

  def call(self, inputs):
    x = self.dense1(inputs)
    return self.dense2(x)
model = MyModel()
```
在实践中，我们采用方法3的方法，原因是我们发现方法1、方法2不太好操控，比如我们如果想要冻结一些参数、拉多个分支进行训练等等, 用3比较适合。这是因为keras.Sequential的方法无法返回来多个值(一个方法是我们可以把连个result concate在一起然后一起回传，但是总感觉怪怪的)
下面我们详细讨论第三种方法

### keras构建自己的layer和model
我先讲下我们在实践中总结的方法
- 自己写的layer继承keras.layers.Layer这个类，模型继承keras.Model这个类
- 在model.py里面，各个layer尽量散开放, 不要把layer集成到一个稍大一些的模块里面

### keras 固定参数/batchNorm参数
I) 固定参数来自多个方面的需求，比如我们想要finetune的时候固定住前面几个layer的系数，或者说我们想要固定住主branch的参数，然后从这个branch上面拉小分支出来，只训练这个分支，但是同时又希望主branch的参数保持不变。
那么我们如何固定参数??
下面是一个读取模型pretrained model, 并且把名称里面不含有classify的layer参数固定住的示范
```
#----trained old model path model-----
classify_model_path = "xxx.h5"
model = model_file.LandmarksModel()
input_shape = (batch_size, input_size, input_size, channel)
input = np.random.random(input_shape)
output = model(input)
model.load_weights(classify_model_path, by_name=True)
for layer in model.layers:
    if 'classify' in layer.name: continue
    layer.trainable = False
```
II) BatchNorm参数的固定。
BatchNorm是一个极其特殊的layer, 在pytorch里面，requires_grad=False是无法固定住batchNorm的参数的。一般都需要采用下面的方式才能固定


在keras里面,最开始batchnorm也需要类似pytorch上面的方法进行固定。但是[ref](https://github.com/keras-team/keras/issues/7085)提到, 经过更新layer.trainable=False也对batchNorm起作用了。
还有另外一种固定keras里面BatchNorm参数的方法, 可以给bn层传一个参数来控制
```
keras.layers.BatchNormalization(trainable=False)
```
实际中我们这边固定bn的方法使用了第二种方法，第一种我试验过，但是最后放弃了。


### keras 固定主分支训练小分支参数加载

如上所述，有时候我们可以这么搞，我们先训练一个模型，并且称这个模型是主分支。训练好之后，我们从这个模型的某一个layer上面拉去一个小分支，然后再训练这个小分支，我们希望能够复用主分支已经训练好的feature, 即加载原来的主分支的参数，同时保证训练小分支的时候，主分支的参数不变。那么我们如果加载主分支的参数呢?
一个例子如下
```
class ExamModel(keras.Model):
    def __init__(self):
        super(LandmarksModel, self).__init__()
        self._preprocess_layer =  keras.layers.InputLayer(input_shape=(input_size, input_size, 1))
        self.layer1 = layer1(..., bn_trainable=True, name = "layer1")
        self.layer2 = layer2(..., bn_trainable=True, name = "layer2")
        self.layer3 = layer3(..., bn_trainable=True, name = "layer3")

    def call(self, x):
        x = self._preprocess_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.layer3(x)
        return output
```
我们用上面的模型训练了一个任务。上面的layer1/layer2/layer3都是我们自定义的layer, 本身就是多种conv/bn/relu的组合。
我们保存模型的时候，正常保存(为了之后可以在这个模型上面接着训练)
```
model.save_weights("model_example.h5")
```
然后我们单独写一个脚本读取这个h5,然后freeze参数值
```
model = ExamModel()
....
model.load_weights("model_example.h5", by_name=True)
for layer in model.layers:
    if 'classify' in layer.name: continue
    layer.trainable = False #固定参数

model.save_weights("model_example_frozen_params.h5")
```

然后我们增加了一个小分支，训练这个小分支, 改动如下
```
class ExamModelwithBranch(keras.Model):
    def __init__(self):
        super(LandmarksModel, self).__init__()
        self._preprocess_layer =  keras.layers.InputLayer(input_shape=(input_size, input_size, 1))
        self.layer1 = layer1(..., bn_trainable=False, name = "layer1") #---设为False, 让bn参数不参加训练
        self.layer2 = layer2(..., bn_trainable=False, name = "layer2")
        self.layer3 = layer3(..., bn_trainable=False, name = "layer3")
        self.classify_layer = layer4(..., bn_trainable=True, name = "classify_layer4")

    def call(self, x):
        x = self._preprocess_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.layer3(x)
        classify_out = self.classify_layer(x)
        return output, classify_out
```
读取pretraeined params代码如下
```
model = ExamModelwithBranch()
...
#---#把除了classify分支其他的layer系数都固定住, 这样我们才能加载上面model_example_frozen_params.h5的模型参数---
for layer in landmark_model.layers:
    if "classify" in layer.name: continue #classify分支需要训练
    layer.trainable=False
model.load_weights("model_example_frozen_params.h5", by_name=True)

train()
```


### keras如何复制参数

一些情况下，我可能需要把model1的某个layer的系数复制到model2的某一个layer里面，这个时候可以使用
```
set_weights()
get_weights()
```
比如我们将model1的第零个layer的参数复制给model2的第一个layer
```
model2.layers[1].set_weights(model1.layers[0].get_weights())
```
如果model1的第零个layer是一个sequential结构，可以这么搞
```
model2.layers[1].set_weights(model1.layers[0].layers[0].get_weights())
```
我自己这种复制方法把一个包含有sequential结构的model里面的layer都拆出来了，参数就直接复制到新的model对应的结构上

### keras其他一些小问题
#### train_on_batch收敛,GradientTape不收敛
我们之前碰到了这样一个问题，用train_on_batch训练loss可以收敛，GradientTape方法训练loss不收敛或者效果远远不如train_on_batch的好。参考[ref1](https://github.com/tensorflow/tensorflow/issues/28901)、[ref2](https://stackoverflow.com/questions/56868981/training-logistic-regression-with-tf-gradienttape-cant-converge), 原因是model output的维度和ground truth的维度不一致
