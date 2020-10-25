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


# file_1 = 'label_7415.csv'
file_1 = 'all/88231.csv'
file_2 = '7000.csv'
file_duplicated = 'duplicated.csv'
file_independent = 'independent.csv'


with open(file_1, 'r') as f:
    lines = f.readlines()
    file_1 = np.array([l.strip().split(',') for l in lines])

with open(file_2, 'r') as f:
    lines = f.readlines()
    file_2 = np.array([l.strip().split(',') for l in lines])


file_2 = file_2[:, 0]

with open(file_duplicated, 'w') as f:
    for i in file_1:
        if i[0] in file_2:
            f.write(str(i[0] + ',0,0\n'))

with open(file_duplicated, 'r') as f:
    lines = f.readlines()
    duplicated = np.array([l.strip().split(',') for l in lines])



with open(file_independent, 'w') as f:
    for i in file_1:
        if i[0] not in file_2:
            f.write(str(i[0] + ',0,0\n'))

with open(file_independent, 'r') as f:
    lines = f.readlines()
    independent = np.array([l.strip().split(',') for l in lines])

numbers_1 = len(file_1)
numbers_2 = len(file_2)
numbers_3 = len(duplicated)
numbers_4 = len(independent)
print('lenth of file_1 is {}'.format(numbers_1))
print('lenth of file_2 is {}'.format(numbers_2))
print('lenth of duplicated is {}'.format(numbers_3))
print('lenth of independent is {}'.format(numbers_4))

