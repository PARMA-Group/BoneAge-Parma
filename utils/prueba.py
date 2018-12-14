import csv

dicc = {}

with open('male_train.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        dicc[int(row[1])] = row[1]


with open('female_train.csv', newline='') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        dicc[int(row[1])] = row[1]

l = list(dicc.keys())
l.sort()
print(l)



def intersection():
    import os
    d1 = os.listdir("dnlm_test")
    d2 = os.listdir("dnlm_train")

    d3 = os.listdir("fitted_test")
    d4 = os.listdir("registered_train")

    if (d1==d2) and (d3==d4):
        print("diferentes")
    else:
        print(len(set(d1).intersection(d2)))