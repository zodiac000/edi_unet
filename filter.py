import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt 
from PIL import Image
from pdb import set_trace

# matplotlib.use('tkagg')


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


file_cutout = 'csv/eval/eval_cutout.csv'
file_crop = 'csv/eval/eval_crop.csv'
file_output = 'csv/eval/evaluation_all.csv'


with open(file_cutout, 'r') as f:
    lines = f.readlines()
    lst_cutout = np.array([l.strip().split(',') for l in lines])

with open(file_crop, 'r') as f:
    lines = f.readlines()
    lst_crop = np.array([l.strip().split(',') for l in lines])

total = len(lst_cutout)

threshold_cutout = 0.03
threshold_crop = 0.9

mask_1 = lst_cutout[:,5].astype(float) <= threshold_cutout
mask_2 = lst_crop[:, 5].astype(float) >= threshold_crop
valid = lst_cutout[mask_1 & mask_2]


# with open('csv/pred_4615_2.csv', 'w') as f:
with open(file_output, 'w') as f:
    for i, pred in enumerate(valid):
        f.write(str(pred[0]) + ',' 
                + str(pred[3]) + ','
                + str(pred[4]) + '\n'
                )

print(len(valid))


# names = valid[:, 0]

# x = valid[:, 1].astype(int)
# y = valid[:, 2].astype(int)
# x_pred = valid[:, 3].astype(int)
# y_pred = valid[:, 4].astype(int)
# colors = valid[:, 5].astype(float)

# dx = np.absolute(x - x_pred)
# dy = np.absolute(y - y_pred)

# label = valid[:, 1:3].astype(int)
# pred = valid[:, 3:5].astype(int)
# dist = cal_dist(label, pred)



# print(dist)

# plt.scatter(dx, dy)
# plt.title(titleTxt.format(threshold_cutout, threshold_crop, dist, len(valid), total))
# plt.xlabel('E distance on X')
# plt.ylabel('E distance on Y')
# plt.show()

