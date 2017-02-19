file_name=argv(){1};
s= load([file_name '.z']);
s= reshape(s, 3, length(s)/3);
s1= s(1,:);
s2= s(2,:);
s3= s(3,:);

[fs1, fr1]= positiveFFT(s1, 1);
[fs2, fr2]= positiveFFT(s2, 1);
[fs3, fr3]= positiveFFT(s3, 1);

fs1= abs(fs1);
fs2= abs(fs2);
fs3= abs(fs3);

sampling_freq= 1000;
cfr= ceil(fr1*2*(sampling_freq));
result=zeros(3,sampling_freq);

for i= 2:length(cfr)
	result(1, cfr(i))= max(result(1, cfr(i)), fs1(i));
	result(2, cfr(i))= max(result(2, cfr(i)), fs2(i));
	result(3, cfr(i))= max(result(3, cfr(i)), fs3(i));
end

result(1, :)= result(1, :)/max(result(1, :));
result(2, :)= result(2, :)/max(result(2, :));
result(3, :)= result(3, :)/max(result(3, :));

#result= [fs1; fs2; fs3];
save 'result.fft' result;
