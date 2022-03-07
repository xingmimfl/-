### 一些参考
- https://hub.fastgit.org/zldrobit/yolov5/blob/tf-android/models/tf.py 重点推荐这个
- https://github.com/qqwweee/keras-yolo3  参考这里如何处理dataset
- https://github.com/eriklindernoren/Keras-GAN
- https://github.com/qubvel/segmentation_models

### keras.conv2D没有groups
keras=2.3.1之前没有这个参数。
	
```python
TypeError: ('Keyword argument not understood:', 'groups')
```
解决方法，升级keras版本，或者使用其他的卷积核操作。

### Keras动态reshape
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

### keras concatenate方法
```
out = keras.backend.concatenate([out, classify_out], axis=1)
```

### keras.model
```python
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
            loss = wing_loss(landmarks_out, pts_data) #---loss
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

II) 我们可以使用pytorch里面的dataloader的方法，这样做的好处是一方面我们很多模型可能是pytorch写的，数据增强都是都是先写好的，另外一方面pytorch也提供了很多优秀的数据增强的方法
```
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
mport keras.backend as K
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

### 构建keras model的方法
这个地方有很多可以说的, keras里面有很多创建model的方法
