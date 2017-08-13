import os
import json
import tensorflow as tf
import numpy as np
import sys

def load_data(taxonomy, count= 20000):
	data_list= list(filter(lambda x:x[-3:]=='npy' and int(x.split('.')[-2].split('_')[1])>15000, os.listdir(taxonomy)))[0:count]
	result=[]
	for data_file in data_list:
		data= np.load('%s/%s' %(taxonomy, data_file)).tolist()
		new_data=[]
		for i in range(3000):
			new_data.append(max(data[i*2:(i+1)*2]))

		result.append(np.array(new_data, dtype='f'))
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

def train(batch_size=1000, steps_per_batch=500, whole_step= 750):
	#taxonomy_list=['archaea','bacteria','protozoa','fungi', 'invertebrate','plant']
	taxonomy_list=['archaea','bacteria','fungi','protozoa', 'plant','invertebrate']
	taxonomy_data= []
	for tax in taxonomy_list:
		taxonomy_data.append(load_data(tax))

	val_len= int(min(list(map(lambda x: len(x), taxonomy_data)))/3)

	print('Using %d data for validating each taxonomy' %(val_len))

	taxonomy_train= list(map(lambda x: x[0:val_len*-1], taxonomy_data))

	taxonomy_val= list(map(lambda x: x[int(len(x)-val_len):], taxonomy_data))

	x_val = taxonomy_val[0]
	for val in taxonomy_val[1:]:
		x_val= np.concatenate((x_val, val), axis= 0)
	y_val= []
	for idx in range(len(taxonomy_val)):
		for i in taxonomy_val[idx]:
			y_val.append([idx])
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

	clf= learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[int(x_val.shape[1]/20),int(x_val.shape[1]/100)], n_classes=len(taxonomy_list), model_dir='./batch_dnn_model')

	for i in range(whole_step):
		training_data=list(map( lambda x: get_batch(x[1], batch_size, x[0]), enumerate(taxonomy_train)))
		(x_t,y_t) = training_data[0]
		for t_d in training_data[1:]:
			(xt_d, yt_d)= t_d
			x_t= np.concatenate((x_t, xt_d))
			y_t= np.concatenate((y_t, yt_d))
		x_train= x_t
		y_train= y_t
		clf.fit(x=x_train, y=y_train, steps=steps_per_batch)

		predictions= np.array(list(clf.predict(np.asarray(x_val, dtype=np.float32))))
		predictions.shape= y_val.shape
		precision= list(predictions==y_val).count(True)/len(y_val)
		print('Predict as %f precision' %(precision))
	return (clf, predictions)
