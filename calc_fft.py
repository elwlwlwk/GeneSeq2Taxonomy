import os

file_list= list(map(lambda x: x[0:-2], filter(lambda x: '.z' ==x[-2:], os.listdir())))
for faa_file in file_list:
	print(faa_file)
	os.system('octave extract_fft.m %s' % faa_file)
	os.system('mv result.fft %s.fft' % faa_file)
