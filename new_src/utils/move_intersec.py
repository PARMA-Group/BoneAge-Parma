import os
import sys
import shutil

def intersec(a,b):
	s = set(a)
	return [val for val in b if val in s]

real_path = sys.argv[1]
dnlm_path = sys.argv[2]
move_to = sys.argv[3]

real_files = os.listdir(real_path)
dnlm_files = os.listdir(dnlm_path)
real_files.sort()
dnlm_files.sort()

files_interec = intersec(real_files, dnlm_files)

real_path += "/"
move_to += "/"

for i in files_interec:
	os.rename(real_path+i, move_to+i)
print("done")
