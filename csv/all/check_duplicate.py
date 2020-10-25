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


file_1 = 'label_7415.csv'
file_2 = 'test_927.csv'
out_file = 'duplicated.csv'


with open(file_1, 'r') as f:
    lines = f.readlines()
    file_1 = np.array([l.strip().split(',') for l in lines])

with open(file_2, 'r') as f:
    lines = f.readlines()
    file_2 = np.array([l.strip().split(',') for l in lines])


file_2 = file_2[:, 0]

with open(out_file, 'w') as f:
    for i in file_1:
        if i[0] in file_2:
            f.write(str(i[0] + ',0,0\n'))


with open(out_file, 'r') as f:
    lines = f.readlines()
    file_3 = np.array([l.strip().split(',') for l in lines])


numbers_1 = len(file_1)
numbers_2 = len(file_2)
numbers_3 = len(file_3)
print('lenth of file_1 is {}'.format(numbers_1))
print('lenth of file_2 is {}'.format(numbers_2))
print('lenth of duplicated is {}'.format(numbers_3))

