import sys
import os
import numpy as np
import math
from oct2py import octave
from extract_feature import get_sequence, calc_z_curve, z_curve_fft

if __name__=='__main__':
	taxonomy= sys.argv[1]
	fft_length= int(sys.argv[2])
	time_length= int(sys.argv[3])
	file_list= list(filter(lambda x: 'fna' == x[-3:], os.listdir(taxonomy)))
	for seq_file in file_list:
		print(seq_file)
		seqs= get_sequence(taxonomy+'/'+seq_file, 1000)#1000 is not very meaningfull.
		if len(seqs)==0:
			continue
		feature_idx= 1
		seqs= list(filter(lambda x: len(x) > fft_length* time_length,
			seqs))
		for seq in seqs:
			for sub_seq_idx in range(int(len(seq)/(fft_length*time_length))):
				cur_seqs= seq[sub_seq_idx*fft_length*time_length: (sub_seq_idx+1)*fft_length*time_length]
				cur_seqs= np.reshape(list(cur_seqs), (time_length, fft_length)).tolist()
				cur_ffts=[]
				for cur_seq in cur_seqs:
					z_curve= calc_z_curve(cur_seq)
					fft_result= z_curve_fft(z_curve)
					cur_ffts.append(fft_result)
				
				print(seq_file+"_"+str(feature_idx))
				np.save(taxonomy+'/'+seq_file+'_'+str(feature_idx)+'_'+str(fft_length)+'_'+str(time_length), np.array(cur_ffts, dtype='f'))

				feature_idx+= 1
