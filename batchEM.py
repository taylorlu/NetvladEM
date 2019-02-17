import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow.contrib.slim as slim

num_clusters = 7
input_dim = 2
xCount = 500
time = 20
batch_size = 5
logits = tf.placeholder(tf.float32, shape=[batch_size, time, input_dim]) #(n, t, d)

inp1 = np.random.normal(loc=[-2+200, -5+100], scale=0.5, size=(xCount, input_dim)).astype(np.float32)
inp2 = np.random.normal(loc=[1+100, 7+100], scale=6, size=(xCount, input_dim)).astype(np.float32)
inp3 = np.random.normal(loc=[-2+200, -1+100], scale=1.1, size=(xCount, input_dim)).astype(np.float32)
inp4 = np.random.normal(loc=[3+100, -5+200], scale=2.5, size=(xCount, input_dim)).astype(np.float32)
inp5 = np.random.normal(loc=[-3+200, 9+200], scale=4, size=(xCount, input_dim)).astype(np.float32)
inp6 = np.random.normal(loc=[-10+200, -9+100], scale=3, size=(xCount, input_dim)).astype(np.float32)
inp7 = np.random.normal(loc=[10+100, 20+200], scale=6, size=(xCount, input_dim)).astype(np.float32)

# inp1 = np.random.normal(loc=[-2, -5], scale=0.5, size=(xCount, input_dim)).astype(np.float32)
# inp2 = np.random.normal(loc=[1, 7], scale=6, size=(xCount, input_dim)).astype(np.float32)
# inp3 = np.random.normal(loc=[-2, -1], scale=1.1, size=(xCount, input_dim)).astype(np.float32)
# inp4 = np.random.normal(loc=[3, -5], scale=2.5, size=(xCount, input_dim)).astype(np.float32)
# inp5 = np.random.normal(loc=[-3, 9], scale=4, size=(xCount, input_dim)).astype(np.float32)
# inp6 = np.random.normal(loc=[-10, -9], scale=3, size=(xCount, input_dim)).astype(np.float32)
# inp7 = np.random.normal(loc=[10, 20], scale=6, size=(xCount, input_dim)).astype(np.float32)

inpu_arr = np.vstack((inp1,inp2,inp3,inp4,inp5,inp6,inp7))
inpu = inpu_arr.tolist()
random.shuffle(inpu)

global_means = np.mean(inpu_arr, axis=(0)).reshape((-1, input_dim))
global_variances = np.std(inpu_arr, axis=(0)).reshape((-1, input_dim))

with tf.variable_scope('scope0', reuse=tf.AUTO_REUSE):
    inputs_shape = logits.get_shape()
    params_shape = inputs_shape[-1:]
    reduction_axis = 0
    epsilon = 1e-8

    mean, variance = tf.nn.moments(logits, [0,1], keep_dims=True)
    stddev = ( (variance + epsilon) ** (.5) )
    logits2 = (logits - mean) / stddev


with tf.variable_scope('scope1', reuse=tf.AUTO_REUSE):
    #(k, d)
    weights_list = []
    biases_list = []
    for i in range(num_clusters):
        weight = tf.get_variable('weight_{}'.format(i), [logits2.get_shape().as_list()[-1], 1],
            dtype=tf.float32, initializer=tf.glorot_normal_initializer(), trainable=True)
        bias = tf.get_variable('bias_{}'.format(i), [1],
            dtype=tf.float32, initializer=tf.constant_initializer(-1), trainable=True)
        weights_list.append(weight)
        biases_list.append(bias)

    weights = tf.transpose(tf.squeeze(tf.convert_to_tensor(weights_list)))
    biases = tf.squeeze(tf.convert_to_tensor(biases_list))

    alfa = -0.25*tf.div(tf.reduce_sum(tf.square(weights), 0), biases)
    alfa = tf.reduce_mean(alfa)

    inputs = tf.reshape(logits2, [-1, logits2.get_shape()[-1]])   # (n*t, h)
    output = tf.matmul(inputs, weights) + biases    # (n*t, k)
    output = tf.reshape(output, [batch_size, -1, num_clusters]) # (n, t, k)
    PI = tf.nn.softmax(output)  # (n, t, k)

    weights_tran = tf.transpose(weights) # (k, h)
    weights_pow = tf.reduce_sum(tf.square(weights_tran), -1)  + 1e-8 # (k)
    biases_tile = tf.tile(tf.div(biases, weights_pow), [logits2.get_shape()[-1]])
    biases_tile = tf.reshape(biases_tile, [logits2.get_shape()[-1], num_clusters])
    MUs = -2*tf.multiply(tf.transpose(biases_tile), weights_tran)  # (k, h)

    MU_Origin = tf.add(tf.multiply(MUs, tf.tile(global_variances, [num_clusters,1])), global_means)

    inputs = tf.reshape(logits, [-1, logits.get_shape()[-1]])
    tileInputs = tf.tile(inputs, [1, num_clusters])
    tileInputs = tf.reshape(tileInputs, [-1, num_clusters, logits.get_shape()[-1]])    # (n*t, k, h)

    offcenter = tileInputs - MU_Origin # (n*t, k, h)

    distances = tf.reduce_sum(tf.square(offcenter), -1)   # (n*t, k)
    distances = tf.reshape(distances, [batch_size, -1, num_clusters])  # (n, t, k)

    vladLoss = tf.reduce_sum(tf.multiply(PI, distances))
    vladLoss = vladLoss/(num_clusters*logits.get_shape().as_list()[0])

    gStep = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(float(0.01), gStep, 1000, 0.97, staircase=True)

    train_op_list = []
    for i in range(num_clusters):
        opt1 = tf.train.AdamOptimizer(learning_rate) # tf.gather_nd(adjusts, [i])*tf.gather_nd(adjustments, [i])*tf.gather_nd(distances_norm, [i])*
        grads1 = opt1.compute_gradients(vladLoss, [weights_list[i],biases_list[i]])
        train_op1 = opt1.apply_gradients(grads1)
        train_op_list.append(train_op1)
    train_op = tf.group(train_op_list)



sess = tf.Session()
sess.run(tf.global_variables_initializer())


fig = plt.figure()

ax1 = fig.add_subplot(111)
centerArr = []
centerMarker = ['$1$', '$2$', '$3$', '$4$', '$5$', '$6$', '$7$']
ax1.set_xlim([50,250])
ax1.set_ylim([50,250])
# ax1.set_xlim([-30,30])
# ax1.set_ylim([-30,30])
ax1.scatter(inp1[:,0],inp1[:,1],c = 'r',marker = 'o', s=1)
ax1.scatter(inp2[:,0],inp2[:,1],c = 'b',marker = 'o', s=1)
ax1.scatter(inp3[:,0],inp3[:,1],c = 'g',marker = 'o', s=1)
ax1.scatter(inp4[:,0],inp4[:,1],c = 'y',marker = 'o', s=1)
ax1.scatter(inp5[:,0],inp5[:,1],c = 'c',marker = 'o', s=1)
ax1.scatter(inp6[:,0],inp6[:,1],c = 'm',marker = 'o', s=1)
ax1.scatter(inp7[:,0],inp7[:,1],c = 'c',marker = 'p', s=1, alpha=0.5)
plt.ion()
plt.show()

curIndex = 0
for epoch in range(800000):
    if(curIndex + batch_size*time > xCount*num_clusters):
        random.shuffle(inpu)
        curIndex = 0
        
    feed_input = np.array(inpu[curIndex:curIndex+batch_size*time]).reshape((batch_size, time, input_dim))
    curIndex += batch_size*time
    _, vladLoss2, m, lr, _alfa = sess.run([train_op, vladLoss, MU_Origin, learning_rate, alfa], feed_dict={logits:feed_input, gStep: epoch})
    if(epoch%400==0):
        print('lr = {}, loss = {}, \nalfa = {}'.format(lr, vladLoss2, _alfa))

        for i in range(len(centerArr)):
            centerArr[i].remove()
        centerArr = []
        for i in range(num_clusters):
            centerArr.append(ax1.scatter(m[i][0],m[i][1],c = 'black', marker = centerMarker[i], linewidths=1))
        plt.pause(0.001)
        