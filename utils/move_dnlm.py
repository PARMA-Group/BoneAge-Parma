import os
import sys
import shutil

"""
    ARGS:
    0 -> file name
    1 -> PATH WITH IMAGES
    2 -> NEW PATH TO DNLM IMAGES
"""

# path with images
cwd = sys.argv[1]
# path of the new image directory
path_to_files = sys.argv[2]

dir = path_to_files
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

# lista que contiene las imágenes
dirr = os.listdir(cwd)

# for para obtener las imágenes dnlm
dnlm_images = list()
EXT = "DeNLM.png"
EXT_LEN = len(EXT)

# para mover las imagenes
cwd += "/"
path_to_files += "/"

for i in dirr:
    ext = i[-EXT_LEN:]
    # Si es una imágen DNLM
    if ext == EXT:
        print(i)
        os.rename(cwd+i, path_to_files+i)