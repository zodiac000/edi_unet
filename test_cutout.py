from models.Discriminator import Discriminator
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import CutoutDataset, CutoutDataset_pred 
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
from utils import heatmap_to_coor, accuracy_sum, spike, gaussion_hmap
from tqdm import tqdm

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

from pdb import set_trace

import matplotlib
#matplotlib.use('tkagg')

#Test on predictions
def eval_prediction():
    batch_size = 180
    semi = '_8'
    train_gen = '10d' + semi
    train_dis = '15d' 
    
    file_to_read = './csv/all/pred/pred_88231_' + train_gen + '.csv'
    file_to_write = './csv/eval/eval_cutout_' + train_dis + '' + semi + '.csv'
    dir_weight = 'check_points/weights_cutout_' + train_dis + '.pth'

    dataset = CutoutDataset_pred(csv_file=file_to_read)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    discriminator = Discriminator().cuda()
    discriminator.load_state_dict(torch.load(dir_weight))
    discriminator.eval()
    
    with torch.no_grad():
        predictions = []
        all_logits = []
        all_labels = []
        all_preds = []
        all_names = []
        pbar = tqdm(total=len(dataloader))
        for i, batch_data in enumerate(dataloader):
            image_cutout = batch_data['image_cutout'].cuda()
            image_name = batch_data['image_name']
            coor_label = batch_data['coor_label']
            coor_pred = batch_data['coor_pred']
            b_size = len(image_cutout)
            logits = discriminator(image_cutout)
            for l in logits:
                all_logits.append(l.flatten().cpu().detach().item())
            all_labels.extend(coor_label.numpy())
            all_preds.extend(coor_pred.numpy())
            all_names.extend(image_name)

            pbar.update()

    with open(file_to_write, 'w') as f:
        for i, name in enumerate(all_names):
            f.write(name + ',' \
                    + str(all_labels[i][0]) + ',' 
                    + str(all_labels[i][1]) + ',' 
                    + str(all_preds[i][0]) + ',' 
                    + str(all_preds[i][1]) + ',' 
                    + str(all_logits[i]) 
                    + '\n')

if __name__ == '__main__':
    eval_prediction()

