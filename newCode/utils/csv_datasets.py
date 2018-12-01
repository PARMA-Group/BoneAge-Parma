import csv
import random
import os


def split_list(files, train_rate=0.8):
    random.shuffle(files)
    m = int(len(files) * train_rate)
    train_data = files[:m]
    test_data = files[m:]

    return train_data, test_data

def writte_csv(csv_file, rows):
    for i in rows:
        csv_file.writerow(i)   

CSV_FILE = "train.csv"
CSV_MALE_TRAIN = "male_train.csv"
CSV_FEMALE_TRAIN = "female_train.csv"
CSV_MALE_TEST = "male_test.csv"
CSV_FEMALE_TEST = "female_test.csv"
CSV_UNISEX_TRAIN = "unisex_train.csv"
CSV_UNISEX_TEST = "unisex_test.csv"
PATH_OF_FILES = "../datasets/dnlm_train"

files = os.listdir(PATH_OF_FILES)
files.sort()

id = 'id'
boneage = 'boneage'
male = 'male'
fieldnames = [id, boneage]

males = []
females = []
unisex = []

csvfile = open(CSV_FILE)
reader = csv.DictReader(csvfile)

for row in reader:
    temp =  row[id] + ".png"
    if temp in files:
        if row[male] == 'True':
            males.append({id:row[id], boneage:row[boneage]})
            #writtermale.writerow({id:row[id], boneage:row[boneage]})    
        else:
            #writtefemale.writerow({id:row[id], boneage:row[boneage]})
            females.append({id:row[id], boneage:row[boneage]})
        unisex.append({id:row[id], boneage:row[boneage]})


# open male files
csv_male_train = open(CSV_MALE_TRAIN, 'w')
csv_male_test = open(CSV_MALE_TEST, 'w')

# open female files
csv_female_train = open(CSV_FEMALE_TRAIN, 'w')
csv_female_test = open(CSV_FEMALE_TEST, 'w')

# open unisex files
csv_unisex_train = open(CSV_UNISEX_TRAIN, 'w')
csv_unisex_test = open(CSV_UNISEX_TEST, 'w')

# writters de cada csv respectivo
writter_male_train = csv.DictWriter(csv_male_train, fieldnames=fieldnames)
writter_male_test = csv.DictWriter(csv_male_test, fieldnames=fieldnames)

writter_female_train = csv.DictWriter(csv_female_train, fieldnames=fieldnames)
writter_female_test = csv.DictWriter(csv_female_test, fieldnames=fieldnames)

writter_unisex_train = csv.DictWriter(csv_unisex_train, fieldnames=fieldnames)
writter_unisex_test = csv.DictWriter(csv_unisex_test, fieldnames=fieldnames)

# writte headers
writter_male_train.writeheader()
writter_male_test.writeheader()
writter_female_train.writeheader()
writter_female_test.writeheader()
writter_unisex_train.writeheader()
writter_unisex_test.writeheader()

# splitea 80~20 a male
male_train, male_test = split_list(males)

# crea los csv de cada dataset
writte_csv(writter_male_train, male_train)
writte_csv(writter_male_test, male_test)

# splitea 80~20 a female
female_train, female_test = split_list(females)

# crea los csv de cada dataset
writte_csv(writter_female_train, female_train)
writte_csv(writter_female_test, female_test)

# splitea 80~20 a unisex
unisex_train, unisex_test = split_list(unisex)

# crea los csv de cada dataset
writte_csv(writter_unisex_train, unisex_train)
writte_csv(writter_unisex_test, unisex_test)

csvfile.close()
csv_male_train.close()
csv_male_test.close()
csv_female_train.close()
csv_female_test.close()
csv_unisex_train.close()
csv_unisex_test.close()