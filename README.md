# GeneSeq2Taxonomy
Taxonomy Classifer Using Gene Sequence.

Using FFT and Neural Net to identify gene sequence's taxonomy.

## How it works
0. Collect lots of sequence data.

1. Translate gene sequence to z-curve sequence. It's for reduce training cost by reducing features.
I think there's no meaningful difference comparing other methods.
http://www.mecs-press.org/ijitcs/ijitcs-v4-n8/IJITCS-V4-N8-3.pdf

2. Apply Fourier transform to extract it's spectral features. Manipulate Fourier transformed data to train neural net.
I split frequence axis by 1000 area and picked up max value for each area. I've got about 85% accuracy using this method.(Best method I've tried. See below to other methods I've tried)
For example if Fourier transformed data is [1, 2, 3, 6, 8, 7, 5, 3, 1, 4, 6, 4]. I split frequence axis to 4 and pick max value, then result will be [3, 8, 5, 6].

3. Use neural net, train data.

## Another methods manipulating FFT result.
- Pick frequency order by magnitude.
Suppose magnitudes [1, 2, 3, 4, 8, 7, 5, 3, 6, 4] and frequence of each magnitudes [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]. Pick 3 frequences order by magnitude. [0.5, 0.6, 0.9].

Result: Failed cause there's lots of peak around 2/3 pi area. It means there's many 3-periodic sequence; Codon. Data looks like this.
[0.333333, 0.333321, 0.3333342, 0.333432 ....]

- Pick frequency order by magnitude at band pass filtered data.
I wiped up data around 2/3 pi and 0 pi. And pick frequences order by it's magnitude.

Result: Accuracy about 80% classifing Archaea and Bacteria. But between Archaea, Bacteria and Protozoa, accuracy falls around 65%.

- Pick magnitude at random frequency and pick best classifier.
Result: Failed. Frequency is near real number. Too many learning case. Accuracies are about 70% classifing Archaea and Bacteria.
