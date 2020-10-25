import numpy as np
from numpy.random import shuffle

with open('csv/15d/15d.csv', 'r') as f_read:
    lines = f_read.readlines()
    shuffle(lines)
    with open('10d.csv', 'w') as f_write:
        for line in lines[:10]:
            f_write.write(line)
    with open('5d', 'w') as f_write:
        for line in lines[10:]:
            f_write.write(line)
