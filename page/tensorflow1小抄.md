[返回主页](../README.md)

------



#                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  Tensorflow 1.x

## 1 Queue

### 1.1 构造队列

- 只将文件送入队列，适用于文件名即为label、不需要存储label的情况

  ```python
  # 文件路径列表
  file_path_list=list()
  
  # 1.生成文件队列
  file_queue=tf.train.string_input_producer(file_path_list，num_epochs=10)
  
  # 2.文件读取以及预处理
  reader=tf.WholeFileReader() 
  key,value=reader.read(file_queue)
  images=tf.image.decode_jpeg(value,channels=3)
  images=tf.image.resize_images(images,[64,64]) # 统一尺寸
  # 此时images还没确定形状shape=[?,?,?]，使用set_shape()设置静态形状 
  # 如果images形状已经确定的时候,修改动态形状要用reshape()
  # 只有shape确定的Tensor才能送入批处理
  images.set_shape(shape=[64,64,3]) 
  
  # 3.批处理
  label_batch,images_batch=tf.train.batch([key,images],batch_size) # 标签和图片一起送入批处理
  images_batch=tf.train.batch([images],batch_size) # 只将图片送入批处理
  tf.train.shuffle_batch()
  ```

- 将文件和label同时送人队列

  ```python
  # 文件路径列表
  file_path_list=list()
  
  # 1.生成文件队列
  data_queue=tf.train.slice_input_producer([image_list,label_list],num_epochs=10)
  labels=data_queue[1]
  images=tf.read_file(data_queue[0]) # 不能使用tf.WholeFileReader()
  
  # 2.文件读取以及与处理
  images=tf.image.decode_jpeg(images)
  images = tf.image.resize_images(images, [32, 32]) / 255
  images.set_shape([32, 32, 3])
  
  # 3.批处理
  images_batch,label_batch=tf.train.batch([images,labels],batch_size)
  tf.train.shuffle_batch()
  
  ```
  
  ```python
  tf.train.batch(
      tensors,
      batch_size,
      num_threads=1,
      capacity=32,
      enqueue_many=False,
      shapes=None,
      dynamic_pad=False,
      allow_smaller_final_batch=False,
      shared_name=None,
      name=None
  )
  ```
  
  - **tensors：一个列表或字典的tensor用来进行入队**
  
  - **batch_size：设置每次从队列中获取出队数据的数量**
  
  - **num_threads：用来控制入队tensors线程的数量，如果num_threads大于1，则batch操作将是非确定性的，输出的batch可能会==乱序==**
  
  - **capacity：一个整数，用来设置队列中元素的最大数量**
  
  - enqueue_many：在tensors中的tensor是否是单个样本
  
  - shapes：可选，每个样本的shape，默认是tensors的shape
  
  - dynamic_pad：Boolean值.允许输入变量的shape，出队后会自动填补维度，来保持
  
    batch内的shapes相同
  
  - allow_samller_final_batch：可选，Boolean值，如果为True队列中的样本数量小
  
  - batch_size时，出队的数量会以最终遗留下来的样本进行出队，如果为Flalse，小于batch_size的样本不会做出队处理
  
  - shared_name：可选，通过设置该参数，可以对多个会话共享队列
  
  - name：可选，操作的名字
  
  ```python
  tf.train.shuffle_batch(
    				tensors, 
    				batch_size, 
    				capacity, 
    				min_after_dequeue, num_threads=1, 
    				seed=None, 
    				enqueue_many=False, 
    				shapes=None, 
    				name=None
  )
  ```
  
  - **min_after_dequeue:** 单次提取**batch_size**个样本后，剩余队列进行下一次提取保证的至少剩余量，如果太大，则刚开始需要向capacity中补充很多数据，**一定要保证这参数小于capacity参数的值，否则会出错**。

### 1.2 开启队列

```python
with tf.Session() as sess:
    #线程协调器
    coord=tf.train.Coordinator()
    #启动队列
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
    data=sess.run(images_batch)
    img, lab = sess.run([img_batch, lab_batch])
		
    coord.request_stop()
    coord.join(threads)
```

### 1.3 其他情况

```python
# 文本：
tf.TextLineReader() # 读取
tf.decode_csv() # 解码

# 图片
tf.WholeFileReader() # 读取：
tf.image.decode_jpeg(contents) # 解码
tf.image.decode_png(contents)

# 二进制：
tf.FixedLengthRecordReader(record_bytes) # 读取
tf.decode_raw() # 解码
# TFRecords
tf.TFRecordReader() # 读取

key, value = 读取器.read(file_queue)
key：文件名
value：一个样本
```

## 2 pickle

```python
# 序列化 存到文件中
with open(path,'wb') as f:
  data=list()
  pickle.dump(data,f)

# 反序列化 从文件中加载
with open(path,'rb') as f:
  data=pickle.load(f)
```



## 3 tfrecord

### 3.1 tfrecord格式简介

tfrecord文件中数据是通过tf.train.Example Protocol Buffer格式存储的，tf.trian.Example格式如下：

```python
message Example {
	Features features = 1;
};

message Features{
	map<string,Feature> featrue = 1;
};

message Feature{
  oneof kind{
    BytesList bytes_list = 1;
    FloatList float_list = 2;
    Int64List int64_list = 3;
    }
};
```

### 3.2 存储tfrecord

```python
with tf.python_io.TFRecordWriter("demo.tfrecords") as writer:
  # 循环构造example对象，并序列化写入文件
  for i in range(100):
    image = image_batch[i].tostring()
    label = label_batch[i][0]
    example = tf.train.Example(features=tf.train.Features(feature={
        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
    }))
    # example.SerializeToString()
    # 将序列化后的example写入文件
    writer.write(example.SerializeToString())
```



### 3.3 读取tfrecord

```python
 # 1、构造文件名队列
file_queue = tf.train.string_input_producer(["demo.tfrecords"])

# 2、读取与解码
# 读取
reader = tf.TFRecordReader()
key, value = reader.read(file_queue)

# 解析example
feature = tf.parse_single_example(value, features={
    "image": tf.FixedLenFeature([], tf.string),
    "label": tf.FixedLenFeature([], tf.int64)
})
image = feature["image"]
label = feature["label"]
print("read_tf_image:\n", image)
print("read_tf_label:\n", label)

# 解码
image_decoded = tf.decode_raw(image, tf.uint8)
print("image_decoded:\n", image_decoded)
# 图像形状调整
image_reshaped = tf.reshape(image_decoded, [self.height, self.width, self.channel])
print("image_reshaped:\n", image_reshaped)

# 3、构造批处理队列
image_batch, label_batch = tf.train.batch([image_reshaped, label], batch_size=100, num_threads=2, capacity=100)
print("image_batch:\n", image_batch)
print("label_batch:\n", label_batch)

# 开启会话
with tf.Session() as sess:

    # 开启线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    image_value, label_value = sess.run([image_batch, label_batch])
    print("image_value:\n", image_value)
    print("label_value:\n", label_value)

    # 回收资源
    coord.request_stop()
    coord.join(threads)
```

## 4 常用模块

### 4.1 tf.train.Saver

```python
saver1=tf.train.Saver()
saver2=tf.train.Saver({"w",w})
saver3=tf.train.Saver([w])
# 主要参数
# var_list: Saver存储的变量合集
# reshape:是否允许从ckpt文件中恢复时改变变量形状
# sharded:是否允许将ckpt中的变量轮循放置在所有设备上
# max_to_keep:保留最近的检查点个数
# restore_sequentially:是否按顺序恢复所有变量，当模型较大时按顺序恢复可以降低内存使用

with tf.Session() as sess:
	# 保存
	saver.save(sess,'test.ckpt')
  # 加载
  saver.restore(sess,'test.ckpt')

```



### 4.2 TensorBoard

```python
writer = tf.summary.FileWriter("output/summary/", graph=tf.get_default_graph())
# 记录标量
tf.summary.scalar("loss", loss)
# 记录统计量
tf.summary.historam()
# 记录图片
tf.summary.image()
# 记录音频
tf.summary.audio()
# 汇总所有summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
   for i in range(1, epoch):
      summary_values=sess.run(merged)
      # 写入事件文件
      writer.add_summary(summary_values,i)
   writer.close()
```

```shell
tensorboard --logdir "output/summary/"
```



## 5 Demo

### 5.1 梯度下降GradientDescent

```python
# GradientDescent
import tensorflow as tf
import numpy as np
# 梯度下降demo
DATA_NUM = 10
FEATURE_NUM = 4
W=[1,2,3,4]
B=1
lr = 0.1
epoch = 500
# 生成训练数据
np.random.seed((100))
train_x = np.random.normal(size=[DATA_NUM, FEATURE_NUM])
train_y = np.add(np.sum(np.multiply(train_x,W ), axis=1), B)
# 定义模型结构
with tf.name_scope('input'):
    x = tf.placeholder(dtype=tf.float32, shape=[DATA_NUM, FEATURE_NUM], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[DATA_NUM, ])
# 初始化权重w,b
with tf.name_scope('weights'):
    tf.random.set_random_seed(100)
    w = tf.Variable(initial_value=tf.random_normal(shape=[FEATURE_NUM]), dtype=tf.float32)
    tf.random.set_random_seed(100)
    b = tf.Variable(initial_value=tf.random_normal(shape=[1]), dtype=tf.float32)
# train
with tf.name_scope('train'):
    y_ = tf.add(tf.reduce_sum(tf.multiply(x, w), axis=1), b)
    loss = tf.reduce_mean(tf.square(y - y_))
    grad_w, grad_b = tf.gradients(loss, [w, b])
    updata_w = tf.assign(w, tf.subtract(w, tf.multiply(lr, grad_w)))
    updata_b = tf.assign(b, tf.subtract(b, tf.multiply(lr, grad_b)))

# tensorboard
writer = tf.summary.FileWriter(logdir='./log', graph=tf.get_default_graph())
tf.summary.scalar("loss", loss)
merged = tf.summary.merge_all()
saver=tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, epoch + 1):
        summary_res, uw, ub, l ,tmp_w,tmp_b= sess.run([merged, updata_w, updata_b, loss,w,b],
                                                  feed_dict={x: train_x, y: train_y})

        print(f'epoch:{i}  loss:{l},w：{tmp_w}  b:{tmp_b}')

        writer.add_summary(summary_res, i)
        saver.save(sess,"./ckpt/model.ckpt",global_step=i)
    print(f'w：{w.eval()}  b:{b.eval()}')
writer.close()
```

### 5.2 LeNet

```python
import tensorflow as tf
import os
import pickle
import PIL.Image as Image
import data2tensor
import matplotlib.pyplot as plt
from tensorflow.contrib import slim

# dataset_path = 'cifar10/'
# with open(os.path.join(dataset_path, 'data_batch_1'), "rb") as f:
#     cifar = pickle.load(f, encoding='latin1')
# labels = cifar['labels']
# data = cifar['data']
# filenames = cifar['filenames']

version="lenet_trian_cifar_demo"
def forward(x, weights):
    x = tf.reshape(x, [-1, 32, 32, 3], name='reshape1')
    x=tf.cast(x,tf.float32)
    c1 = tf.nn.conv2d(x, weights['c1w'], strides=[1, 1, 1, 1], padding='VALID')
    c1 = tf.nn.sigmoid(c1 + weights['c1b'])
    s2 = tf.nn.max_pool(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    c3 = tf.nn.conv2d(s2, weights['c3w'], strides=[1, 1, 1, 1], padding='VALID')
    c3 = tf.nn.sigmoid(c3 + weights['c3b'])
    s4 = tf.nn.max_pool(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flatten = tf.reshape(s4, [-1, 5 * 5 * 16])
    f5 = tf.nn.sigmoid(tf.matmul(flatten, weights['f5w']) + weights['f5b'])
    f6 = tf.nn.sigmoid(tf.matmul(f5, weights['f6w']) + weights['f6b'])
    f7 = tf.matmul(f6, weights['f7w']) + weights['f7b']
    # output=tf.nn.softmax(f7)
    output=f7
    return output


def conv_weights():
    weights = {}

    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d()

    fc_initializer = tf.contrib.layers.xavier_initializer()

    with tf.variable_scope('LeNet', reuse=tf.AUTO_REUSE):
        # c1 in 32*32*3 out 28*28*6
        weights['c1w'] = tf.get_variable('c1w', [5, 5, 3, 6], initializer=conv_initializer)
        weights['c1b'] = tf.get_variable('c1b', initializer=tf.zeros([6]))

        # s2 in 28*28*6 out 14*14*6

        # c3 in 14*14*6 out 10*10*16
        weights['c3w'] = tf.get_variable('c3w', [5, 5, 6, 16], initializer=conv_initializer)
        weights['c3b'] = tf.get_variable('c3b', initializer=tf.zeros([16]))

        # s4 in 10*10*16 out 5*5*16

        # f5 in 5*5*16 out 120
        weights['f5w'] = tf.get_variable('f5w', [5 * 5 * 16, 120], initializer=fc_initializer)
        weights['f5b'] = tf.get_variable('f5b', initializer=tf.zeros([120]))

        # f6 in 120 out  84
        weights['f6w'] = tf.get_variable('f6w', [120, 84], initializer=fc_initializer)
        weights['f6b'] = tf.get_variable('f6b', initializer=tf.zeros([84]))

        # f7 in 84 out nway
        weights['f7w'] = tf.get_variable('f7w', [84, 10], initializer=fc_initializer)
        weights['f7b'] = tf.get_variable('f7b',initializer=tf.zeros([10]))

        return weights

def train():


    x,y=data2tensor.get_tensor_batch(dataset="cifar10",stage="train",batch_size=256)

    w=conv_weights()
    y_=forward(x,w)
    # loss=tf.losses.mean_squared_error(y,y_)
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y,y_))
    acc=tf.metrics.accuracy(tf.argmax(y,axis=1),tf.argmax(y_,axis=1))
    optimizer=tf.train.AdamOptimizer().minimize(loss)

    # summary
    writer = tf.summary.FileWriter(f"output/summary/{version}/", graph=tf.get_default_graph())
    tf.summary.scalar("loss",loss)
    tf.summary.scalar("acc_total",acc[1])
    tf.summary.scalar("acc_batch",acc[0])
    merged=tf.summary.merge_all()

    # saver
    saver=tf.train.Saver(max_to_keep=10)
    save_path=f"output/weights/{version}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with tf.Session() as sess:
        loss_list=[]
        acc_list=[]

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # ckpt= tf.train.get_checkpoint_state(f"output/trianed_weights/")
        # saver.restore(sess, save_path=ckpt.model_checkpoint_path)
        # print(f"restore:{ckpt.model_checkpoint_path}")

        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)

        for i in range(1,1000):

            _, l, a = sess.run([optimizer, loss, acc])
            # l, a = sess.run([loss, acc])


            if i%20==0:
                # summary
                summary_values=sess.run(merged)
                writer.add_summary(summary_values,i)
                # loss_list.append(l)
                # acc_list.append(a[1])
                print(f'epoch{i}   loss:{l}   acc:{a}')
                saver.save(sess=sess,save_path=f"output/weights/{version}/lenet-epoch{i}-loss{l}-acc{round(a[1],2)}.ckpt",global_step=i)
        coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    train()

```



## 6 程序示例

### 6.1 data2tensor.py

```python
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_file(dataset_name='cifar10', stage='train',suffix='png'):
    if stage=="train":
        raw_dir = f"dataset/{dataset_name}/train"
    else:
        raw_dir = f"dataset/{dataset_name}/test"
    dir_names = os.listdir(raw_dir)
    dir_names = [name for name in dir_names if os.path.isdir(os.path.join(raw_dir, name))]
    image_list = []
    label_list=[]
    for dir in dir_names:
        img_names = os.listdir(os.path.join(raw_dir, dir))
        img_file = [os.path.join(raw_dir, dir, name) for name in img_names if name.endswith(suffix)]
        image_list.extend(img_file)
        label_list.extend(np.asarray(int(dir)).repeat(len(img_file)).tolist())


    return image_list, label_list


def get_tensor_batch(dataset="cifar10", stage='train', batch_size=128,suffix='png'):
    supported_dataset=["cifar10"]
    if dataset not in supported_dataset:
        print("only support ",supported_dataset)

    image_list, label_list = get_file(dataset, stage)

    data_queue = tf.train.slice_input_producer([image_list, label_list],shuffle=True)

    labels = tf.one_hot(data_queue[1], 10)
    # labels=data_queue[1]

    images = tf.read_file(data_queue[0])
    if suffix=="png":
        images = tf.image.decode_png(images)
    else:
        print("wrong suffix when get tensor batch")
    images = tf.image.resize_images(images, [32, 32])/255
    images.set_shape([32, 32, 3])

    images_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size,capacity=512,min_after_dequeue=256)


    return images_batch, label_batch

if __name__ == "__main__":
    print("debug")
    img, lab = get_tensor_batch(dataset="cifar10", stage='train')

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        im, l = sess.run([img, lab])

        coord.request_stop()
        coord.join(threads)
    plt.imshow(im[0].astype(np.uint8))
    plt.show()
    print(l)

```

### 6.2 model.py

```python
import tensorflow as tf
from tensorflow.contrib import slim
import data2tensor


class DemoNet():
    def __init__(self,class_num):
        self.class_nun=class_num


    def build_lenet(self, image_batch):
        with tf.variable_scope("lenet", reuse=tf.AUTO_REUSE):
            c1 = slim.conv2d(image_batch,
                             num_outputs=6,
                             kernel_size=[5, 5],
                             activation_fn=tf.nn.sigmoid,
                             stride=1,
                             weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                             biases_initializer=tf.constant_initializer(value=0.0),
                             reuse=tf.AUTO_REUSE,
                             scope="c1_layer")
            s2 = slim.max_pool2d(c1, kernel_size=[2, 2], stride=2, scope="s2_layer")

            c3 = slim.conv2d(s2,
                             num_outputs=16,
                             kernel_size=[5, 5],
                             activation_fn=tf.nn.sigmoid,
                             stride=1,
                             weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                             biases_initializer=tf.constant_initializer(value=0.0),
                             reuse=tf.AUTO_REUSE,
                             scope="c3_layer")

            s4 = slim.max_pool2d(c3, kernel_size=[2, 2], stride=2, scope="s4_layer")

            flatten = slim.flatten(s4, scope="flatten_layer")

            f5 = slim.fully_connected(flatten,
                                      num_outputs=120,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
                                                                                       seed=None),
                                      biases_initializer=tf.constant_initializer(value=0.0),
                                      reuse=tf.AUTO_REUSE,
                                      scope="f5_layer")
            f6 = slim.fully_connected(f5,
                                      num_outputs=84,
                                      activation_fn=tf.nn.sigmoid,
                                      weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
                                                                                       seed=None),
                                      biases_initializer=tf.constant_initializer(value=0.0),
                                      reuse=tf.AUTO_REUSE,
                                      scope="f6_layer",
                                      )
            f7 = slim.fully_connected(f6,
                                      num_outputs=self.class_nun,
                                      activation_fn=None,
                                      weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01,
                                                                                       seed=None),
                                      biases_initializer=tf.constant_initializer(value=0.0),
                                      reuse=tf.AUTO_REUSE,
                                      scope="f7_layer")
            output = f7

            return output

    def build_full_connected(self, image_batch):
        with tf.variable_scope("full_connected_net", reuse=tf.AUTO_REUSE):
            with slim.arg_scope([slim.fully_connected],
                               weights_initializer=tf.random_normal_initializer(mean=0.0,stddev=0.01,seed=None), 
                               biases_initializer=tf.constant_initializer(value=0.0),
                               reuse=tf.AUTO_REUSE):
                flatten = slim.flatten(image_batch, scope="flatten_layer")

            f1 = slim.fully_connected(flatten,
                                      num_outputs=120,
                                      activation_fn=tf.nn.sigmoid,

                                      scope="f1_layer")
            f2 = slim.fully_connected(f1,
                                      num_outputs=84,
                                      activation_fn=tf.nn.sigmoid,

                                      reuse=tf.AUTO_REUSE,
                                      scope="f2_layer",
                                      )
            f3 = slim.fully_connected(f2,
                                      num_outputs=self.class_nun,
                                      activation_fn=None,
                                      reuse=tf.AUTO_REUSE,
                                      scope="f3_layer")
            output = f3
            return output

    def build_demo_net(self, image_batch,is_training):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
                            biases_initializer=tf.constant_initializer(value=0.0),
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            biases_regularizer=slim.l2_regularizer(1e-4),
                            normalizer_fn=slim.batch_norm,
                            reuse=tf.AUTO_REUSE):
            with tf.variable_scope("lenet", reuse=tf.AUTO_REUSE):
                c1 = slim.conv2d(image_batch,
                                 num_outputs=6,
                                 kernel_size=[5, 5],
                                 activation_fn=tf.nn.sigmoid,
                                 stride=1,
                                 scope="c1_layer")
                s2 = slim.max_pool2d(c1, kernel_size=[2, 2], stride=2, scope="s2_layer")

                c3 = slim.conv2d(s2,
                                 num_outputs=16,
                                 kernel_size=[5, 5],
                                 activation_fn=tf.nn.sigmoid,
                                 stride=1,
                                 scope="c3_layer")

                s4 = slim.max_pool2d(c3, kernel_size=[2, 2], stride=2, scope="s4_layer")

                flatten = slim.flatten(s4, scope="flatten_layer")

                f5 = slim.fully_connected(flatten,
                                          num_outputs=120,
                                          activation_fn=tf.nn.sigmoid,
                                          scope="f5_layer")
                f6 = slim.fully_connected(f5,
                                          num_outputs=84,
                                          activation_fn=tf.nn.sigmoid,
                                          scope="f6_layer",
                                          )
                f6=slim.dropout(f6,is_training=is_training)
                f7 = slim.fully_connected(f6,
                                          num_outputs=10,
                                          activation_fn=None,
                                          reuse=tf.AUTO_REUSE,
                                          scope="f7_layer")
                output = f7

                return output

```

### 6.3 train.py

```python
import tensorflow as tf
from tensorflow.contrib import slim
from model import DemoNet
import data2tensor
import os
import numpy as np
import matplotlib.pyplot as plt


def train(cfgs):
    model = DemoNet(10)
    images_train, labels_train = data2tensor.get_tensor_batch(dataset="cifar10", stage='train',
                                                              batch_size=cfgs["train_batch"])
    images_val, labels_val = data2tensor.get_tensor_batch(dataset="cifar10", stage='test', batch_size=64)

    # train
    if cfgs['network'] == "FC":
        print("build train network FC")
        train_pred = model.build_full_connected(images_train)
    elif cfgs['network'] == "LENET":
        print("build train network LENET")
        train_pred = model.build_lenet(images_train)
    else:
        print("build train network DEMONET")
        train_pred = model.build_demo_net(images_train, is_training=True)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels_train, logits=train_pred))
    train_acc = tf.metrics.accuracy(labels=tf.argmax(labels_train, axis=1),
                                    predictions=tf.argmax(train_pred, axis=1))
    optimizer = tf.train.AdamOptimizer()
    tf.summary.scalar("lr", optimizer._lr)
    train_op = optimizer.minimize(loss)

    # val
    if cfgs['network'] == "FC":
        print("build val network FC")
        val_pred = model.build_full_connected(images_val)
    elif cfgs['network'] == "LENET":
        print("build val network LENET")
        val_pred = model.build_lenet(images_val)
    else:
        print("build val network DEMONET")
        val_pred = model.build_demo_net(images_val, is_training=True)
    val_acc = tf.metrics.accuracy(labels=tf.argmax(labels_val, axis=1),
                                  predictions=tf.argmax(val_pred, axis=1))
    val_op = val_acc
    saver = tf.train.Saver(max_to_keep=10)
    save_path = f"output/weights/{cfgs['version']}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = tf.summary.FileWriter(logdir=f"output/summary/cifar/{cfgs['version']}", graph=tf.get_default_graph())
    tf.summary.scalar("train_loss", loss)
    tf.summary.scalar("train_acc", train_acc[1])
    tf.summary.scalar("val_acc", val_acc[1])
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for epoch in range(1, cfgs["epoch"] + 1):
            best_val_acc = 0
            _, summary_value = sess.run([train_op, merged])
            writer.add_summary(summary_value, epoch)
            if epoch % 50 == 0:
                l, a = sess.run([loss, train_acc])
                print(f"epoch{epoch} | loss:{l} | acc:{a}")
            if epoch % 200 == 0:
                val_a = sess.run(val_op)
                print(f"test acc:{val_a}")
                if val_a[1] > best_val_acc:
                    best_val_acc = val_a[1]
                    saver.save(sess, save_path=os.path.join(save_path,
                                                f"demonet-epoch{epoch}-acc{round(best_val_acc, 2)}.ckpt"),
                                    global_step=epoch)

        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    cfgs = {}
    cfgs["version"] = "lenet_cifar10"
    cfgs["train_batch"] = 256
    cfgs["epoch"] = 80000
    cfgs["network"] = "LENET"  # FC LENET DEMONET
    # cfgs["lr"]=5e-3
    train(cfgs)

```

### 6.4 test.py

```python
import tensorflow as tf
from tensorflow.contrib import slim
from model import DemoNet
import data2tensor
import numpy as np
import matplotlib.pyplot as plt
def test():
    images, labels = data2tensor.get_tensor_batch(dataset="cifar10", stage="test", batch_size=1000)
    model = DemoNet(is_training=False)
    pred = model.build(images)
    val_acc = tf.metrics.accuracy(labels=tf.argmax(labels, axis=1),
                                  predictions=tf.argmax(pred, axis=1))
    restorer= tf.train.Saver()
    restorer_path=tf.train.latest_checkpoint(f'output/weights/{cfgs["version"]}/')
    print("restorer_path",restorer_path)

    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        restorer.restore(sess, restorer_path)
        acc = sess.run(val_acc)
        print(acc[1])
if __name__ == "__main__":
    cfgs={}
    cfgs["version"]="train_cifar10"
    test(cfgs)
```

