import tensorflow as tf
import numpy as np
#tf.enable_eager_excution() #즉시실행
def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))
W_values = np.linspace(-3, 5, num=15)
cost_values = []


x_data = [1,2,3,4,5]
y_data = [1,2,3,4,5]
#초기값 임의 지정. 대부분 랜덤값으로 지정함.

W = tf.Variable(2.9)
b = tf.Variable(0.5)
learning_rate = 0.001  # 굉장히 작은 값을 주로 사용

for i in range(1000):
    with tf.GradientTape() as tape:
        # 가설함수
        hypothesis = W*x_data+b
        #비용
        cost = tf.reduce_min(tf.square(hypothesis - y_data))

        w_grad, b_grad = tape.gradient(cost, [W,b])

        W.assign_sub(learning_rate*w_grad)
        b.assign_sub(learning_rate*b_grad)
    if(i%10==0):
        print("{} | {} | {}| {}".format(i,W.numpy(), b.numpy(), cost))
        #v = [1., 2.,3.,4.]
        #tf.reduce_min(v) => 차원이 줄어듦 2.5


X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))