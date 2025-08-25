# notes about count+seq

- in `exprbyseq.py` there is a bit of code which will get you sequences and paired count profiles for single cells. it is demo'd in `get_seq_test.ipynb`

- it requires you to `pip install snapatac2-scooby` which is probably best done in a fresh environment

- you get the counts for a window of the genome by passing by the actual nucleotide index of the reference sequence. so integers 0 to 3e9 roughly.

- you get the counts for particular cells by the index of the dataset, so 0 to 3e5 roughly