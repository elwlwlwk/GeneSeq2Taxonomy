import os
import json
import tensorflow as tf
import numpy as np

def load_data(taxonomy):
	data_list= list(filter(lambda x:x[-3:]=='npy', os.listdir(taxonomy)))
	result=[]
	for data_file in data_list:
		result.append(np.load('%s/%s' %(taxonomy, data_file))[0:1000])
		if len(result)%1000== 0:
			print("%s: %d/%d" %(taxonomy, len(result), len(data_list)))

	return np.array(result, dtype='f')

def get_batch(data, batch_size, label):
	np.random.shuffle(data)
	x_train= data[0:min(batch_size, len(data))]
	y_train=[]
	for i in range(batch_size):
		y_train.append([label])
	y_train= np.array(y_train)
	return (x_train, y_train)

def train(batch_size=1000, steps_per_batch=1000, whole_step= 1000):
	archaea_data= load_data('archaea')
	bacteria_data= load_data('bacteria')
	protozoa_data= load_data('protozoa')
	fungi_data= load_data('fungi')

	val_len= int(min(len(archaea_data), len(bacteria_data), len(protozoa_data), len(fungi_data))/3)

	print('Using %d data for validating each taxonomy' %(val_len))

	archaea_train= archaea_data[0:val_len*-1]
	bacteria_train= bacteria_data[0:val_len*-1]
	protozoa_train= protozoa_data[0:val_len*-1]
	fungi_train= fungi_data[0:val_len*-1]

	archaea_val= archaea_data[int(len(archaea_data)-val_len):]
	bacteria_val= bacteria_data[int(len(bacteria_data)-val_len):]
	protozoa_val= protozoa_data[int(len(protozoa_data)-val_len):]
	fungi_val= fungi_data[int(len(fungi_data)-val_len):]

	x_val= np.concatenate((archaea_val, bacteria_val, protozoa_val, fungi_val), axis=0)
	y_val= []
	for i in range(len(archaea_val)):
		y_val.append([0])
	for i in range(len(bacteria_val)):
		y_val.append([1])
	for i in range(len(protozoa_val)):
		y_val.append([2])
	for i in range(len(fungi_val)):
		y_val.append([3])
	y_val= np.array(y_val)
	'''
	x= tf.placeholder(tf.float64, [None, x_data.shape[1]])
	W= tf.Variable(tf.zeros([x_data.shape[1], 2], dtype=tf.float64))
	b= tf.Variable(tf.zeros([2], dtype=tf.float64))
	y= tf.matmul(x, W)+b

	y_= tf.placeholder(tf.float64, [None, 2])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess= tf.InteractiveSession()

	tf.global_variables_initializer().run()

	for _ in range(10000):
		sess.run(train_step, feed_dict={x:x_data, y_:y_data})
	'''
	from tensorflow.contrib import learn
	import logging
	logging.getLogger().setLevel(logging.INFO)

	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x_val.shape[1])]

	clf= learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[int(x_val.shape[1]/5),int(x_val.shape[1]/50)], n_classes=4, model_dir='./batch_dnn_model')

	for i in range(whole_step):
		(x_t0, y_t0)= get_batch(archaea_train, batch_size, 0)
		(x_t1, y_t1)= get_batch(bacteria_train, batch_size, 1)
		(x_t2, y_t2)= get_batch(protozoa_train, batch_size, 2)
		(x_t3, y_t3)= get_batch(fungi_train, batch_size, 3)
		x_train= np.concatenate((x_t0, x_t1, x_t2, x_t3))
		y_train= np.concatenate((y_t0, y_t1, y_t2, y_t3))
		clf.fit(x=x_train, y=y_train, steps=steps_per_batch)

	predictions= np.array(list(clf.predict(np.asarray(x_val, dtype=np.float32))))
	predictions.shape= y_val.shape
	precision= list(predictions==y_val).count(True)/len(y_val)
	print('Predict as %f precision' %(precision))
	return (clf, predictions)
