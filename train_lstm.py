#Inspired by https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3%20-%20Neural%20Networks/recurrent_network.py
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

import os
from random import shuffle
# configuration
#						O * W + b -> 10 labels for each image, O[? 28], W[28 10], B[10]
#					   ^ (O: output 28 vec from 28 vec input)
#					   |
#	  +-+  +-+	   +--+
#	  |1|->|2|-> ... |28| time_step_size = 28
#	  +-+  +-+	   +--+
#	   ^	^	...  ^
#	   |	|		 |
# img1:[28] [28]  ... [28]
# img2:[28] [28]  ... [28]
# img3:[28] [28]  ... [28]
# ...
# img128 or img256 (batch_size or test_size 256)
#	  each input size = input_vec_size=lstm_size=28

# configuration variables
input_vec_size = lstm_size = 4500
time_step_size = 20 

batch_size = 64
test_size = 128

def load_data(taxonomy):
	data_list= list(filter(lambda x:x[-3:]=='npy', os.listdir('fna_data/%s' %taxonomy)))[0:2000]
	result=[]
	for data_file in data_list:
		data= np.load('fna_data/%s/%s' %(taxonomy, data_file)).reshape(time_step_size, input_vec_size).tolist()
		result.append(data)
		if len(result)%1000== 0:
			print("%s: %d/%d" %(taxonomy, len(result), len(data_list)))

	return np.array(result, dtype='f')


def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, W, B, lstm_size):
	# X, input shape: (batch_size, time_step_size, input_vec_size)
	XT = tf.transpose(X, [1, 0, 2])  # permute time_step_size and batch_size
	# XT shape: (time_step_size, batch_size, input_vec_size)
	XR = tf.reshape(XT, [-1, lstm_size]) # each row has input for each lstm cell (lstm_size=input_vec_size)
	# XR shape: (time_step_size * batch_size, input_vec_size)
	X_split = tf.split(XR, time_step_size, 0) # split them to time_step_size (28 arrays)
	# Each array shape: (batch_size, input_vec_size)

	# Make lstm with lstm_size (each input vector size)
	lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=True)

	# Get lstm cell output, time_step_size (28) arrays with lstm_size output: (batch_size, lstm_size)
	outputs, _states = rnn.static_rnn(lstm, X_split, dtype=tf.float32)

	# Linear activation
	# Get the last output
	return tf.matmul(outputs[-1], W) + B, lstm.state_size # State size to initialize the stat

def train():

	#tax_list= ['archaea','bacteria','protozoa','fungi','plant','vertebrate','invertebrate']
	tax_list= ['archaea','bacteria']
	num_classes= len(tax_list)
	tax_data=[]
	tax_idx=[]
	for tax in tax_list:
		data= load_data(tax)
		tax_data.append(data)
		idx= [0]*len(tax_list)
		idx[tax_list.index(tax)]=1
		tax_idx+= [idx]*len(data)
	
	tax_data= np.array(tax_data)
	tax_data= tax_data.reshape(tax_data.shape[0]*tax_data.shape[1], tax_data.shape[2], tax_data.shape[3])
	data_idx= list(zip(tax_data, tax_idx))
	shuffle(data_idx)
	tax_data, tax_idx= zip(*data_idx)
	tax_data= np.array(tax_data)
	tax_idx= np.array(tax_idx)

	train_len= int(len(tax_data)/3)
	trX= tax_data[0:train_len]
	trY= tax_idx[0:train_len]
	teX= tax_data[-train_len:]
	teY= tax_idx[-train_len:]

	#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	#trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	#trX = trX.reshape(-1, 28, 28)
	#teX = teX.reshape(-1, 28, 28)

	X = tf.placeholder("float", [None, time_step_size, input_vec_size])
	Y = tf.placeholder("float", [None, num_classes])

	# get lstm_size and output 10 labels
	W = init_weights([lstm_size, num_classes])
	B = init_weights([num_classes])

	py_x, state_size = model(X, W, B, lstm_size)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
	train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
	predict_op = tf.argmax(py_x, 1)

	session_conf = tf.ConfigProto()
	session_conf.gpu_options.allow_growth = True

	# Launch the graph in a session
	with tf.Session(config=session_conf) as sess:
		# you need to initialize all variables
		tf.global_variables_initializer().run()

		for i in range(100):
			for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size)):
				sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

			test_indices = np.arange(len(teX))  # Get A Test Batch
			np.random.shuffle(test_indices)
			test_indices = test_indices[0:test_size]

			print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==sess.run(predict_op, feed_dict={X: teX[test_indices]})))
