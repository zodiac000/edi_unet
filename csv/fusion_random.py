import numpy as np
import random
from pdb import set_trace

file_pass = 'pass_unlabel_all/pass_all_tail_69890.csv'
file_fail = 'fail_unlabel_all/fail_unlabel_all.csv'

file_fusion_pred = 'fusion_label_1000_pred.csv'
file_valid = 'fusion_label_1000_pred_valid.csv'
file_invalid = 'fusion_label_1000_pred_invalid.csv'

def import_data(file_name):
    result = []
    with open(file_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            result.append(line)

    return result

def export_data(file_name, data):
    with open(file_name, 'w') as f:
        for row in data:
            f.write(row)

def extract_data(file_name, file_to_write, valid=True):
    data = import_data(file_name)
    with open(file_to_write, 'w') as f:
        for row in data:
            if (valid and row.strip().split(',')[1] != '-1') or  \
                (not valid and row.strip().split(',')[1] == '-1'):
                f.write(row)


def main():
    # data_pass = import_data(file_pass)
    # data_fail = import_data(file_fail)
    # sample_pass = np.array(random.choices(data_pass, k=875))
    # sample_fail = np.array(random.choices(data_fail, k=125))
    # export_data(file_to_write, np.concatenate((sample_pass, sample_fail)))

    extract_data(file_fusion_pred, file_valid)


if __name__ == "__main__":
    main()
