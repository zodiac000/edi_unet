import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize
from torch.utils.data.dataset import Subset

import numpy as np
import math
from PIL import Image

from unet.unet_model import UNet
from test_unet import my_eval
from utils import accuracy_sum, heatmap_to_coor, crop
from Dataset import WeldingDatasetToTensor

from pdb import set_trace

def split_dataset(data_set, split_at, order=None):
    n_examples = len(data_set)

    if split_at < 0:
        raise ValueError('split_at must be non-negative')
    if split_at > n_examples:
        raise ValueError('split_at exceeds the dataset size')

    if order is not None:
        subset1_indices = order[0:split_at]
        subset2_indices = order[split_at:n_examples]
    else:
        subset1_indices = list(range(0,split_at))
        subset2_indices = list(range(split_at,n_examples))

    subset1 = Subset(data_set, subset1_indices)
    subset2 = Subset(data_set, subset2_indices)

    return subset1, subset2


num_epochs = 30000
batch_size = 20
lr = 1e-4

saved_weight_dir = './check_points/weights_unet_200.pth'

train_csv = './csv/pass_label_valid_200.csv'
validation_csv = './csv/fusion_label_valid_927.csv'

model = UNet().cuda()

labeled_dataset = WeldingDatasetToTensor(data_root='all_images', csv_file=train_csv, root_dir='./')
val_dataset = WeldingDatasetToTensor(data_root='all_images', csv_file=validation_csv, root_dir='./')

train_loader = DataLoader(labeled_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=40, num_workers=4, shuffle=False)

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def train():
    max_total_acc_x = 0
    max_euclidean_distance = 99999
    for epoch in range(num_epochs):
        for batch_index, sample_batched in enumerate(train_loader):

            inputs = sample_batched['image'].cuda()
            labels = sample_batched['hmap'].cuda()
            coors_bc = sample_batched['coor_1'].cpu().detach().numpy()

            # img_names = sample_batched['img_name']
            origin_imgs = sample_batched['origin_img']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.float(), labels.float())

            loss.backward()
            optimizer.step()
            

            if (batch_index+1) % 5 == 0:    # every 20 mini-batches...
                print('Train batch/epoch: {}/{}\tLoss: {:.30f}'.format(
                        batch_index+1,
                        epoch,
                        torch.mean(loss).item())) #/ len(inputs)))


            if (batch_index+1) % 50 == 0:    # every 20 mini-batches...
                with torch.no_grad():
                    valid_loss = 0
                    total_acc_x = 0
                    total_acc_y = 0
                    e_distance = 0
                    distances = []
                    for i, batch in enumerate(valid_loader):
                        inputs = batch['image'].float().cuda()
                        labels = batch['hmap'].float().cuda()
                        coors_bc = batch['coor_1'].cpu().detach().numpy()
                        img_names = batch['img_name']

                        outputs = model(inputs)


                        loss = criterion(outputs, labels)

                        outputs = outputs.cpu().detach().numpy()
                        labels = labels.cpu().detach().numpy()
                        
                        sum_acc_x, sum_acc_y, _, _ = accuracy_sum(outputs, coors_bc)
                        total_acc_x += sum_acc_x
                        total_acc_y += sum_acc_y

                        for index, out in enumerate(outputs):
                            coor_pred = np.array(heatmap_to_coor(out.squeeze()))
                            coor_pred = (coor_pred * [1280/224, 1024/224]).astype(int)
                            dist = np.sum((coor_pred - coors_bc[index]) ** 2) ** 0.5
                            distances.append(dist)

                    distances = np.array(distances)
                    valid_loss = valid_loss / len(valid_loader)
                    print('Valid loss {}'.format(valid_loss))


                    print("=" * 30)
                    print("total acc_x = {:.10f}".format(total_acc_x/len(valid_loader.dataset)))
                    print("total acc_y = {:.10f}".format(total_acc_y/len(valid_loader.dataset)))
                    print("Euclidean Distance: {}".format(np.mean(distances)))
                    print("=" * 30)
                
                    if np.mean(distances) < max_euclidean_distance:
                        max_euclidean_distance = np.mean(distances)
                        torch.save(model.state_dict(), saved_weight_dir)
                        print('model saved to ' + saved_weight_dir)
                    
train()
