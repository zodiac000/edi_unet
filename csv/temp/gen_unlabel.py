import numpy as np
from pdb import set_trace


f1 = 'all.csv'
f2 = 'pass_valid_tail_1000.csv'
f3 = 'unlabel_88158.csv'

with open(f1, 'r') as f:
    f1_lines = f.readlines()
    f1_list = []
    for line in f1_lines:
        f1_list.append(line.strip().split(',')[0])




with open(f2, 'r') as f:
    f2_lines = f.readlines()
    f2_list = []
    for line in f2_lines:
        f2_list.append(line.strip().split(',')[0])

unlabels = np.array(list(set(f1_list) - set(f2_list)))

with open(f3, 'w') as f:
    for name in unlabels:
        f.write(name + '\n')


set_trace()
