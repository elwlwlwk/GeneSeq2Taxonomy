import sys
import os
import numpy as np
import math

def get_sequence(name):
	with open(name) as f:
		seqs= f.read().upper().split('>')
		seqs= list(filter(lambda seq: len(seq)>15000, map(lambda seq: ''.join(seq.split('\n')[1:]),seqs)))
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
	fs= np.absolute(np.fft.fft(seq))
	return fs[0:math.ceil(len(fs)/2)]

def z_curve_fft(z_curve):
	return list(map(lambda curve: positive_fft(curve), z_curve))

def extract_features(fz, features= 1000):
	length= len(fz[0])
	xf=[]
	yf=[]
	zf=[]
	for i in range(features):
		xf.append(max(fz[0][int(length/features*i):int(length/features*(i+1))]))
		yf.append(max(fz[1][int(length/features*i):int(length/features*(i+1))]))
		zf.append(max(fz[2][int(length/features*i):int(length/features*(i+1))]))
	
	xf= np.divide(xf, np.max(xf))
	yf= np.divide(yf, np.max(yf))
	zf= np.divide(zf, np.max(zf))
	return np.array(np.concatenate((xf, yf, zf)), dtype='f')

if __name__=='__main__':
	taxonomy= sys.argv[1]
	file_list= list(filter(lambda x: 'fna' == x[-3:], os.listdir(taxonomy)))
	for seq_file in file_list:
		print(seq_file)
		seqs= get_sequence(taxonomy+'/'+seq_file)
		z_curves= list(map(lambda seq: calc_z_curve(seq), seqs))
		fft_results= list(map(lambda z_curve: z_curve_fft(z_curve), z_curves))
		features= list(map(lambda fz: extract_features(fz), fft_results))
		feature_idx=1
		for feature in features:
			np.save(taxonomy+'/'+ seq_file+'_'+str(feature_idx), feature)
			feature_idx+=1