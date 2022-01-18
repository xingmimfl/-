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

####  keras squeeze方法
```python
tf.keras.backend.squeeze(out, axis=1)
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
https://github.com/rstudio/keras/issues/1104
https://keras.io/api/models/model_training_apis/
https://stackoverflow.com/questions/49100556/what-is-the-use-of-train-on-batch-in-keras
https://blog.csdn.net/baoxin1100/article/details/107917633 这个需要看下,介绍了fit_generator和train_on_batch
https://github.com/jacobgil/keras-dcgan/blob/master/dcgan.py train_on_batch的使用方法

如果使用fit_generator的方法
```python
landmarks_model.fit_generator(face_dataset.generate_data(), steps_per_epoch=steps,epochs=20)
```
Dataset yield应该这么写
```python
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
