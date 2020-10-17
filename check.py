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


all_images = 'csv/all/all-927.csv'
test927 = 'csv/test_927.csv'
duplicated = 'duplicated.csv'


with open(all_images, 'r') as f:
    lines = f.readlines()
    all_images = np.array([l.strip().split(',') for l in lines])

with open(test927, 'r') as f:
    lines = f.readlines()
    test = np.array([l.strip().split(',') for l in lines])

total = len(all_images)
print(total)

test = test[:, 0]

with open(duplicated, 'w') as f:
    for i in all_images:
        if i[0] in test:
            f.write(str(i[0] + ',0,0\n'))





