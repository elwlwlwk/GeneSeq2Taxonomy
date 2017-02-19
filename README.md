# GeneSeq2Taxonomy
Taxonomy Classifer Using Gene Sequence.

Using FFT and Neural Net to identify gene sequence's taxonomy.

## Requires
- Python3
- numpy
- oct2py
- tensorflow
- Octave

## How to use

### Directory tree

0. Collect lots of sequence data.

```
|-GeneSeq2Taxonomy
  |-archaea
    |- *.fna
  |-bacteria
    |- *.fna
  |-protozoa
    |- *.fna
  |-calc_fft.m
  |-extract_feature.py
  |-train.py
```

1. Extract features.

Extract features executing extract_features.py for each directory.

```
>python3 extract_feature.py archaea
Candidatus_Altiarchaeales_archaeon_WOR_SM1_SCG.fna
Candidatus_Altiarchaeales_archaeon_WOR_SM1_SCG.fna_1
Candidatus_Altiarchaeales_archaeon_WOR_SM1_SCG.fna_2
Candidatus_Altiarchaeales_archaeon_WOR_SM1_SCG.fna_3
...
>python3 extract_feature.py bacteria
Arthrobacter_globiformis.fna
Arthrobacter_globiformis.fna_1
Arthrobacter_globiformis.fna_2
Arthrobacter_globiformis.fna_3
...
>python3 extract_feature.py bacteria
protozoa.1.1.genomic.fna
protozoa.1.1.genomic.fna_1
protozoa.1.1.genomic.fna_2
...
```
Then .npy files are generated at each directory.

2. Train features.

Open Python shell, execute train method at train.py

```
python3
Python 3.6.0 |Anaconda 4.3.0 (64-bit)| (default, Dec 23 2016, 12:22:00) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import train from train
>>> clf= train(steps=50000)
```

2/3 of datas are used for train and 1/2 datas are used for calculate accuracy.
Predictions are printed after training.
Tensorflow checkpoint is saved at GeneSeq2Taxonomy/dnn_model.

## How it works

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
