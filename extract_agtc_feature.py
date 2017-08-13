import sys
import os
import numpy as np
import fft
import math
from oct2py import octave

from random import random

def get_sequence(name, feature_number):
	with open(name) as f:
		seqs= f.read().upper().replace('N','').split('>')
		seqs= list(filter(lambda seq: len(seq)>feature_number*2, map(lambda seq: ''.join(seq.split('\n')[1:]),seqs)))
	return seqs

def calc_z_curve(sequence):
	an=[]
	gn=[]
	tn=[]
	cn=[]
	for base in sequence:
		an.append(0)
		gn.append(0)
		tn.append(0)
		cn.append(0)
		if base=="A":
			an[-1]=1
		elif base=="T":
			tn[-1]=1
		elif base=="G":
			gn[-1]=1
		elif base=="C":
			cn[-1]=1
	return [an , gn, tn, cn]

def positive_fft(seq):
	fs= np.absolute(octave.calc_fft(seq)[0])
	return fs[0:math.ceil(len(fs)/2)]

def z_curve_fft(z_curve):
	return list(map(lambda curve: positive_fft(curve), z_curve))

def extract_features(fz, features= 1000):
	length= len(fz[0])
	af=[]
	gf=[]
	tf=[]
	cf=[]
	for i in range(features):
		af.append(max(np.concatenate((fz[0][int(length/features*i):max(int(length/features*(i+1)),int(length/features*i)+1)],[0]))))
		gf.append(max(np.concatenate((fz[1][int(length/features*i):max(int(length/features*(i+1)),int(length/features*i)+1)],[0]))))
		tf.append(max(np.concatenate((fz[2][int(length/features*i):max(int(length/features*(i+1)),int(length/features*i)+1)],[0]))))
		cf.append(max(np.concatenate((fz[3][int(length/features*i):max(int(length/features*(i+1)),int(length/features*i)+1)],[0]))))
	
	af= np.divide(af, np.max(af))
	gf= np.divide(gf, np.max(gf))
	tf= np.divide(tf, np.max(tf))
	cf= np.divide(cf, np.max(cf))
	return np.array(np.concatenate((af, gf, tf, cf)), dtype='f')

if __name__=='__main__':
	taxonomy= sys.argv[1]
	file_list= list(filter(lambda x: 'fna' == x[-3:], os.listdir(taxonomy)))
	feature_number= 2000
	for seq_file in file_list:
		print(seq_file)
		seqs= get_sequence(taxonomy+'/'+seq_file, feature_number)
		if len(seqs)==0:
			continue
		feature_idx=1
		max_len= 2000000
		for t_seq in seqs:
			for seq_idx in range(int((len(t_seq)-1)/max_len)+1):
				seq= t_seq[seq_idx*max_len: max(seq_idx*max_len+1,(seq_idx+1)*max_len- int(random()*max_len/2))]
				if len(seq) is 0:
					continue
				print(seq_file+"_"+str(feature_idx)+"_"+str(len(seq)))
				z_curve= calc_z_curve(seq)
				fft_result= z_curve_fft(z_curve)
				feature= extract_features(fft_result, feature_number)
				np.save(taxonomy+'/'+ seq_file+'_'+str(feature_idx)+'_agtc.len_'+str(len(seq)), feature)
				feature_idx+=1
