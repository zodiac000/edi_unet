import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt 
from PIL import Image
from pdb import set_trace

matplotlib.use('tkagg')


titleTxt = """
threshold_cutout: {}
threshold_crop: {}
============================================
Mean Euclidean Distance {}
Number of Positive Predictions: {}
Number of Total Predictions: {}
"""


def cal_dist(l1, l2):
    return np.mean(np.sum((l1 - l2) ** 2, axis=1) ** 0.5)


all_images = 'all_927.csv'
label927 = 'fusion_927.csv'
out_file = 'all-927.csv'


with open(all_images, 'r') as f:
    lines = f.readlines()
    all_images = np.array([l.strip().split(',') for l in lines])

with open(label927, 'r') as f:
    lines = f.readlines()
    test = np.array([l.strip().split(',') for l in lines])

total = len(all_images)
print(total)

test = test[:, 0]

with open(out_file, 'w') as f:
    for i in all_images:
        if i[0] not in test:
            f.write(str(i[0] + ',0,0\n'))





