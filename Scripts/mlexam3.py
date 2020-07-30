'''
Logistic Classification은 True or False와 같은 이진분류에 쓰임 : Bernoulli Distribution

'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
Sigmoid함수를 가설로 선언함. 0 또는 1만을 리턴

'''
def logistic_regression(features):
    hypothesis  = tf.divide(1., 1. + tf.exp(-tf.matmul(features, W) + b))
    return hypothesis

def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy
def grad(features, labels):
    with tf.GradientTape() as tape:
        hypothesis = logistic_regression(features)
        loss_value = loss_fn(hypothesis,labels)
    return tape.gradient(loss_value, [W,b])

tf.random.set_seed(777) #실행할 때 마다 일정한 결과값 출력을 위한 Seed 설정

x_train = [[1., 2.],
          [2., 3.],
          [3., 1.],
          [4., 3.],
          [5., 3.],
          [6., 2.]]
y_train = [[0.],
          [0.],
          [0.],
          [1.],
          [1.],
          [1.]]
x_test = [[3.85,1.83]]
y_test = [[1.]]
x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]
colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1,x2, c=colors , marker='^')
plt.scatter(x_test[0][0],x_test[0][1], c="red")

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) #데이터셋 지정

W = tf.Variable(tf.zeros([2,1]), name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')
# 초기화 과정, 랜덤으로 수행해도 상관없음.

EPOCHS = 1001
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

for step in range(EPOCHS):
    for features, labels  in iter(dataset.batch(len(x_train))):
        hypothesis = logistic_regression(features)
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads,[W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(hypothesis,labels)))
test_acc = accuracy_fn(logistic_regression(x_test),y_test)
print("Test Result = {}".format(tf.cast(logistic_regression(x_test) > 0.5, dtype=tf.int32)))
print("Testset Accuracy: {:.4f}".format(test_acc))
