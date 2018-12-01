import os
import sys
import shutil


real_path = "../datasets/fitted_train"#sys.argv[1]
#dnlm_path = sys.argv[2]
move_to = "../datasets/fitted_train_temp" #sys.argv[3]

real_files = os.listdir(real_path)
#dnlm_files = os.listdir(dnlm_path)
#real_files.sort()
#dnlm_files.sort()

real_path += "/"
move_to += "/"

#dnlm_size = len(dnlm_files)

for i in real_files:
    os.rename(real_path+i, move_to+i)

print("done")