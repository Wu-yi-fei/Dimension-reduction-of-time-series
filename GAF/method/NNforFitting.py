#!/usr/bin/env python3
"""
用网络拟合时间序列后采样的方法，由于需要巨大的时间资源，没有作为首选方法，但其拟合精度仍然很高，拟合结果可在pic中找到。
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import csv
import pandas as pd
tf.compat.v1.disable_eager_execution()


def ReadData():
    with open('1.csv', 'r') as f:
        c = 0
        data = []
        csvf = csv.reader(f, delimiter=',')
        for row in csvf:
            if c == 0:
                c += 1
                continue
            data.append([int(row[0]), float(row[6])])
            c += 1
        return data

data = ReadData()
print(data)
data = np.array(data)
x_data,y_data = data[:576,0],data[:576,1]

print(x_data)
print(y_data)
plt.title("curve")
plt.plot(x_data,y_data)
plt.show()

df=pd.DataFrame({'x': x_data, 'y': y_data})
plt.plot('x', 'y', data=df, linestyle='none', marker='o')
plt.show()

# plt.annotate(
#     # 标签和协调
#     'This point is interesting!', xy=(25, 50), xytext=(0, 80),
#     # 自定义箭头
#     arrowprops=dict(facecolor='black', shrink=0.05))
x_data=x_data.reshape(-1,1)
y_data=y_data.reshape(-1,1)
print(x_data)


def NeuralNetwork():
    """
    构建一个神经网络用来拟合原时间序列曲线
    :return:
    xs: 原来序列的时间坐标
    ys: 原来序列的数据值
    """
    xs = tf.placeholder(tf.float32, [None, 1])
    ys = tf.placeholder(tf.float32, [None, 1])

    w1 = tf.get_variable("w1", initializer=tf.random_normal([1, 400],stddev = 0.01))
    w2 = tf.get_variable("w2", initializer=tf.random_normal([400, 300],stddev = 0.01))
    w3 = tf.get_variable("w3",initializer=tf.random_normal([300, 1],stddev = 0.01))
    b1 = tf.get_variable("b1", initializer=tf.zeros([1, 400]))
    b2 = tf.get_variable("b2", initializer=tf.zeros([1, 300]))
    b3 = tf.get_variable("b3", initializer=tf.zeros([1, 1]))

    l1 = tf.nn.tanh(tf.matmul(xs, w1) + b1)
    l2 = tf.nn.sigmoid(tf.matmul(l1,w2)+b2)
    prediction = tf.matmul(l2, w3) + b3
    return xs,ys,prediction



xs,ys,prediction = NeuralNetwork()

loss = tf.reduce_mean((tf.square(ys - prediction)))
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(120000):
    _, loss_value = sess.run([train_step, loss], feed_dict={xs: x_data, ys: y_data})
    if i % 10 == 0:
        print(loss_value)
Y_prediction = sess.run(prediction, feed_dict = {xs: x_data})
plt.title("curve")
plt.plot(x_data,Y_prediction)

plt.show()

