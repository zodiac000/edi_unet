import numpy as np
from pdb import set_trace

with open('all.csv', 'r') as f:
    lst = []
    lines = f.readlines()
    for l in lines:
        temp = l.strip().split(',')
        if temp[1] == '-1':
            lst.append(temp)


    lst = np.array(lst)

with open('invalid_85.csv', 'w') as f:
    for l in lst:
        f.write(','.join(l) + '\n')
set_trace()

