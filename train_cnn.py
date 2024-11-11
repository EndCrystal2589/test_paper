# -*- coding: utf-8 -*-
# @Time    : 2024/1/5
# @Author  : -
# @Email   : -
# @File    : train_cnn.py
# @Software: PyCharm ; tensorflow-cpu == 2.3.0
# @Brief   : cnn模型训练代码，训练的代码会保存在models目录下，折线图会保存在results目录下

import tensorflow as tf
import matplotlib.pyplot as plt
from time import *

import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')  # 加入中文字体

# 数据集加载函数，指明数据集的位置并统一处理为imgheight*imgwidth的大小，同时设置batch
def data_load(data_dir, test_data_dir, img_height, img_width, batch_size):
    # 加载训练集
    train_ds = tf.keras.preprocessing.image_dataset_from_directory\
    (
        data_dir,
        label_mode='categorical',  # 标签将被编码为分类向量（使用的损失函数应为：categorical_crossentropy loss）
        shuffle=True,  # 是否打乱数据。默认值：True。如果设置为False,则按字母数字顺序对数据进行排序
        seed=123,  # 用于shuffle和转换的可选随机种子
        image_size=(img_height, img_width),  # 后面会传入224*224
        batch_size=4,  # 数据批次的大小，默认值：32 批次大小影响可水
        color_mode="rgb",  # grayscale、rgb、rgba之一。默认值：rgb。图像将被转换为1、3或者4通道
        interpolation="bilinear",
        # 字符串, 当调整图像大小时使用的插值方法 默认为：bilinear。支持bilinear, nearest, bicubic, area, lanczos3, lanczos5, gaussian, mitchellcubic
        # 不同差值方法可以再水一节 https://blog.csdn.net/fenglepeng/article/details/121107271
    )
    # 加载测试集
    val_ds = tf.keras.preprocessing.image_dataset_from_directory\
    (
        test_data_dir,
        label_mode='categorical',
        shuffle=True,
        seed=123,
        image_size=(img_height, img_width),
        batch_size=4,
        color_mode="rgb",
        interpolation="bilinear",
    )
    class_names = train_ds.class_names

    # 数据检查
    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print('batch_size,image_shape,rgb')
        print(labels_batch.shape)
        print('batch_size,class_number')
        break

    # 运行加速 https://zhuanlan.zhihu.com/p/42417456 (此处可以展开水一节)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # shuffle()：打乱数据 prefetch()：预取数据，加速运行
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # cache()：将数据集缓存到内存当中，加速运行

    # 返回处理之后的训练集、验证集和类名
    return train_ds, val_ds, class_names


# 构建CNN模型
def model_load(IMG_SHAPE=(224, 224, 3), class_num=6):
    # 搭建神经网络 todo 这个classnun不知道是不是自适应的 改一下
    model = tf.keras.models.Sequential\
    ([
        # 对模型做归一化的处理，将0-255之间的像素值统一处理到0到1之间,并将图像大小统一调整为224*224
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),

        # 卷积层，该卷积层的输出为32个通道，卷积核的大小是3*3，激活函数为relu
        tf.keras.layers.Conv2D(32, (9, 9), activation='relu'),
        # 添加池化层，池化的kernel大小是2*2
        tf.keras.layers.MaxPooling2D(2, 2),

        # Add another convolution
        # 卷积层，输出为64个通道，卷积核大小为3*3，激活函数为relu
        tf.keras.layers.Conv2D(64, (9, 9), activation='relu'),
        # 池化层，最大池化，对2*2的区域进行池化操作
        tf.keras.layers.MaxPooling2D(2, 2),

        # 防止过拟合，提高模型的泛化能力
        # 训练准确率与验证准确率相差巨大的原因之一是由于模型过拟合导致的，加了这个提升非常明显，这里了可水一节
        tf.keras.layers.Dropout(0.3),

        # 展平层，将二维的输出转化为一维数组，不含计算参数
        tf.keras.layers.Flatten(),

        # 128个全连接层 神经元个数 激活函数
        tf.keras.layers.Dense(128, activation='relu'),

        # 通过softmax函数将模型输出为类名长度（有几个待识别种类）的神经元上，激活函数采用softmax对应概率值
        tf.keras.layers.Dense(class_num, activation='softmax')
    ])
    # 输出模型信息
    model.summary()
    # 指明模型的训练参数，优化器为sgd优化器，损失函数为交叉熵损失函数，监控指标为准确率
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    # 返回模型
    return model


# 展示训练过程的曲线
def show_loss_acc(history):
    # 从history中提取模型训练集和验证集准确率信息和误差信息
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']  # 训练集损失函数
    val_loss = history.history['val_loss']  # 测试集损失函数

    # 按照上下结构将图画输出
    plt.rcParams['font.size'] = 12  # 设置全局字体大小

    plt.figure(figsize=(10, 12))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='训练准确率')
    plt.plot(val_acc, label='测试准确率')
    plt.legend(loc='lower right')
    plt.ylabel('准确率')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('训练和测试准确率曲线')
    plt.xlabel('训练轮数')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='训练Loss')
    plt.plot(val_loss, label='测试Loss')
    plt.legend(loc='upper right')
    plt.ylabel('交叉熵')
    plt.title('训练和测试Loss曲线')
    plt.xlabel('训练轮数')
    plt.savefig('results/results_cnn_leaf_paper_add_Dropout_9x.png', dpi=100)  # todo 修改输入结果曲线图名称


def train(epochs):
    # 开始训练，记录开始时间
    begin_time = time()
    # todo 加载数据集， 修改为你的数据集的路径
    train_ds, val_ds, class_names = data_load("D:/CNNprogram/vegetables/newdata5/train",
                                              "D:/CNNprogram/vegetables/newdata5/val", 224, 224, 16)
    print(class_names)
    # 加载模型
    model = model_load(class_num=len(class_names))
    # 指明训练的轮数epoch，开始训练
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    # todo 保存模型， 修改为你要保存的模型的名称
    model.save("models/cnn_leaf_paper_add_Dropout_9x.h5")
    # 记录结束时间
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间
    # 绘制模型训练过程图
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)



