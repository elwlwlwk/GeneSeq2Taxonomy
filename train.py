import os
import json
import tensorflow as tf
import numpy as np

def get_complete_list(taxonomy):
	faa_list= list(filter(lambda x:x[-3:]=='faa', os.listdir(taxonomy)))
	complete_list=[]
	for faa_file in faa_list:
		with open(taxonomy+'/'+faa_file) as f:
			index= f.readline()
			if 'complete' in index:
				complete_list.append(faa_file)
	return complete_list

def get_dct_data(taxonomy):
	#faa_list= get_complete_list(taxonomy)
	faa_list= list(filter(lambda x:x[-3:]=='fft', os.listdir(taxonomy)))
	result=[]
	for faa_file in faa_list:
		with open(taxonomy+'/'+faa_file) as f:
			dct_data= f.read().strip().split('\n')[-3:]
			dct_data[0]= json.loads('['+dct_data[0].strip().replace(' ',',')+']')
			dct_data[1]= json.loads('['+dct_data[1].strip().replace(' ',',')+']')
			dct_data[2]= json.loads('['+dct_data[2].strip().replace(' ',',')+']')
			z_curve_data=[]
			z_curve_data+= dct_data[0]
			z_curve_data+= dct_data[1]
			z_curve_data+= dct_data[2]
			result.append(z_curve_data)
		if len(result)%1000== 0:
			print("%s: %d/%d" %(taxonomy, len(result), len(faa_list)))

	return np.array(result, dtype='f')

def train(steps=10000):
	archaea_data= get_dct_data('archaea')
	bacteria_data= get_dct_data('bacteria')
	protozoa_data= get_dct_data('protozoa')

	val_len= min(len(archaea_data), len(bacteria_data), len(protozoa_data))/3

	archaea_train= archaea_data[0:int(len(archaea_data)-val_len)]
	bacteria_train= bacteria_data[0:int(len(bacteria_data)-val_len)]
	protozoa_train= protozoa_data[0:int(len(protozoa_data)-val_len)]

	archaea_val= archaea_data[int(len(archaea_data)-val_len):]
	bacteria_val= bacteria_data[int(len(bacteria_data)-val_len):]
	protozoa_val= protozoa_data[int(len(protozoa_data)-val_len):]

	x_train= np.append(archaea_train, bacteria_train, axis=0)
	x_train= np.append(x_train, protozoa_train, axis=0)
	y_train= []
	x_val= np.append(archaea_val, bacteria_val, axis=0)
	x_val= np.append(x_val, protozoa_val, axis=0)
	y_val= []
	for i in range(len(archaea_train)):
		y_train.append([0])
	for i in range(len(bacteria_train)):
		y_train.append([1])
	for i in range(len(protozoa_train)):
		y_train.append([2])
	y_train= np.array(y_train)

	for i in range(len(archaea_val)):
		y_val.append([0])
	for i in range(len(bacteria_val)):
		y_val.append([1])
	for i in range(len(protozoa_val)):
		y_val.append([2])
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

	feature_columns = [tf.contrib.layers.real_valued_column("", dimension=x_train.shape[1])]

	clf= learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[int(x_train.shape[1]/10),int(x_train.shape[1]/100)], n_classes=3, model_dir='./dnn_model')
	clf.fit(x=x_train, y=y_train, steps=steps)

	predictions= list(clf.predict(np.asarray(x_val, dtype=np.float32)))
	print(predictions)
	return clf
