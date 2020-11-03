from models.Discriminator import Discriminator
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import CropDataset, CropDataset_pred
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
from utils import heatmap_to_coor, accuracy_sum, spike, gaussion_hmap
from tqdm import tqdm

from pdb import set_trace


#Test on predictions
def eval_prediction():
    batch_size = 240
    num_workers = 12

    name = '10b_10'
    name_weights = '15b'

    file_to_read = './csv/all/pred/pred_88231_' + name + '.csv'
    file_to_write = './csv/eval/eval_crop_' + name + '.csv'
    dir_weight = 'check_points/weights_crop_' + name_weights + '.pth'

    dataset = CropDataset_pred(csv_file=file_to_read)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
        # positive = 1
        # total = 1
        for i, batch_data in enumerate(dataloader):
            # print('{}/{}'.format(i, len(dataloader)))
            image_crop = batch_data['image_crop'].cuda()
            image_name = batch_data['image_name']
            coor_label = batch_data['coor_label']
            coor_pred = batch_data['coor_pred']
            b_size = len(image_crop)
            logits = discriminator(image_crop)
            for l in logits:
                all_logits.append(l.flatten().cpu().detach().item())
            # threshold = 1.1115
            # positive += (logits<threshold).sum().item()
            # total += b_size
            # results = [1 if l.item()<threshold else 1 for l in logits]
            # predictions.extend(results)
            all_labels.extend(coor_label.numpy())
            all_preds.extend(coor_pred.numpy())
            all_names.extend(image_name)

            pbar.update()
        # print('positive / total: {}/{}'.format(
                # positive,
                # total,
                # ))
    with open(file_to_write, 'w') as f:
        for i, name in enumerate(all_names):
            f.write(name + ',' \
                    + str(all_labels[i][0]) + ',' 
                    + str(all_labels[i][1]) + ',' 
                    + str(all_preds[i][0]) + ',' 
                    + str(all_preds[i][1]) + ',' 
                    + str(all_logits[i]) 
                    # + str(predictions[i]) 
                    + '\n')

if __name__ == '__main__':
    eval_prediction()

