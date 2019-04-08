import csv

val_data = 'train_data.csv'
pair_data = 'pair_train.csv'

out = open(val_data, 'r')
read = csv.reader(out)
o = []
for i in read:
    o.append(i)

out1 = open(pair_data, 'r')
read1 = csv.reader(out1)
oo = []
for i in read1:
    oo.append(i)
pp = 1