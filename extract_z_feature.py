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
		seqs= list(filter(lambda seq: len(seq)>feature_number, map(lambda seq: ''.join(seq.split('\n')[1:]),seqs)))
	return seqs

def calc_z_curve(sequence):
	An=Gn=Cn=Tn=0
	xn=[]
	yn=[]
	zn=[]
	for base in sequence:
		An=Gn=Cn=Tn=0
		if base=="A":
			An=1
		elif base=="T":
			Tn=1
		elif base=="G":
			Gn=1
		elif base=="C":
			Cn=1
		xn.append(An+Gn-Cn-Tn)
		yn.append(An+Cn-Gn-Tn)
		zn.append(An+Tn-Cn-Gn)
	return [xn , yn, zn]

def positive_fft(seq):
	fs= np.absolute(octave.calc_fft(seq)[0])
	return fs[0:math.ceil(len(fs)/2)]

def z_curve_fft(z_curve):
	return list(map(lambda curve: positive_fft(curve), z_curve))

def extract_features(fz, features= 1000):
	length= len(fz[0])
	xf=[]
	yf=[]
	zf=[]
	for i in range(features):
		xf.append(max(np.concatenate((fz[0][int(length/features*i):max(int(length/features*(i+1)),int(length/features*i)+1)],[0]))))
		yf.append(max(np.concatenate((fz[1][int(length/features*i):max(int(length/features*(i+1)),int(length/features*i)+1)],[0]))))
		zf.append(max(np.concatenate((fz[2][int(length/features*i):max(int(length/features*(i+1)),int(length/features*i)+1)],[0]))))
	
	xf= np.divide(xf, np.max(xf))
	yf= np.divide(yf, np.max(yf))
	zf= np.divide(zf, np.max(zf))
	return np.array(np.concatenate((xf, yf, zf)), dtype='f')

if __name__=='__main__':
	taxonomy= sys.argv[1]
	file_list= list(filter(lambda x: 'fna' == x[-3:], os.listdir(taxonomy)))
	feature_number= 1000
	for seq_file in file_list:
		print(seq_file)
		seqs= get_sequence(taxonomy+'/'+seq_file, 1)
		if len(seqs)==0:
			continue
		feature_idx=1
		max_len=feature_number
		for t_seq in seqs:
			for seq_idx in range(int((len(t_seq)-1)/max_len)+1):
				seq= t_seq[seq_idx*max_len: (seq_idx+1)*max_len]
				if len(seq) < feature_number:
					continue
				print(seq_file+"_"+str(feature_idx)+"_"+str(len(seq)))
				z_curve= calc_z_curve(seq)
				fft_result= z_curve_fft(z_curve)
				feature= np.concatenate((fft_result[0], fft_result[1],fft_result[2]))
				print(feature.shape)
				np.save(taxonomy+'/'+ seq_file+'_'+str(feature_number)+'_'+str(feature_idx)+'_z.len_'+str(len(seq)), feature)
				feature_idx+=1
