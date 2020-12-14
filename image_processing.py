import cv2 
import numpy as np
import glob
import os


data_list = sorted(glob.glob(r'D:\YJ\database\dogs-vs-cats\tes_target\*.jpg'))
save_dir = r'D:\YJ\database\dogs-vs-cats\tes_blur'
os.makedirs(save_dir, exist_ok=True)


for f in data_list:
    im = cv2.imread(f)
    
    im = cv2.GaussianBlur(im,(11,11),1.2)
    # im = cv2.Canny(im, 60, 120)
    
    name = f[f.rfind('\\')+1:]
    cv2.imwrite(os.path.join(save_dir,name), im)