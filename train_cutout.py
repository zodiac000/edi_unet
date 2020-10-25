from models.Discriminator import Discriminator
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import CutoutDataset, InvalidDataset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import math
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
from random import uniform
from utils import heatmap_to_coor, accuracy_sum, spike, gaussion_hmap
from torch.utils.data.dataset import Subset
from tqdm import tqdm


from pdb import set_trace


training_number = 200
num_epochs = 1000000
dist_lower_bound = 4.0
batch_size = 2
invalid_batch_size = batch_size
lr = 1e-5
invalid_csv = None


# train_csv = './csv/100.csv'
# invalid_csv = './csv/pass_invalid_85.csv'
# dir_weight = 'check_points/weights_cutout_7415.pth'
# invalid_dataset = InvalidDataset(csv_file=invalid_csv)
# invalid_loader = DataLoader(invalid_dataset, batch_size=invalid_batch_size, shuffle=True)
name = '15d'
directory = '15d'
train_csv = './csv/' + directory + '/' + name + '.csv'
dir_weight = 'check_points/weights_cutout_' + name + '.pth'


discriminator = Discriminator().cuda()

train_dataset = CutoutDataset(csv_file=train_csv, dist_lower_bound=dist_lower_bound)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


criterion = nn.BCELoss()


# G_solver = torch.optim.Adam(student.parameters(), lr=lr_G)
solver = torch.optim.Adam(discriminator.parameters(), lr=lr)
# G_solver = torch.optim.RMSprop(student.parameters(), lr=lr_G)
# solver = torch.optim.RMSprop(discriminator.parameters(), lr=lr_D)
# G_solver = torch.optim.Adadelta(student.parameters(), lr=lr_G)
# solver = torch.optim.Adadelta(discriminator.parameters(), lr=lr_D)

def train():
    batch_index = 0
    correct_rate = 0
    counter = 0
    loss_min = 999
    for epoch in range(num_epochs):
        valid_threshold = 0.9
        for i, batch_data in enumerate(train_loader):
            counter += 1
            correct = 0
            total = 0
            discriminator.zero_grad()

            if invalid_csv is not None:
                try:
                    image_invalid = invalid_batch['image'].cuda()
                    image_invalid_cutouts = invalid_batch['random_cutouts'].cuda().permute(1,0,2,3,4)

                except:
                    invalid_iter = iter(invalid_loader)
                    invalid_batch = next(invalid_iter)
                    image_invalid = invalid_batch['image'].cuda()
                    image_invalid_cutouts = invalid_batch['random_cutouts'].cuda().permute(1,0,2,3,4)

                b_invalid_size = len(image_invalid)
                b_invalid_cut_size = image_invalid_cutouts.shape[0] * image_invalid_cutouts.shape[1]

                for idx, image in enumerate(image_invalid_cutouts):
                    logits_invalid_cut = discriminator(image)
                    label_invalid_cut_zeros = torch.zeros(len(image), 1).cuda()
                    correct += (logits_invalid_cut<valid_threshold).sum().item() 
                    if idx == 0:
                        loss_invalid_cut = criterion(logits_invalid_cut, label_invalid_cut_zeros)
                    else:
                        loss_invalid_cut = loss_invalid_cut \
                                        + criterion(logits_invalid_cut, label_invalid_cut_zeros)

                logits_invalid = discriminator(image_invalid)
                label_invalid_zeros = torch.zeros(b_invalid_size, 1).cuda()
                loss_invalid = criterion(logits_invalid, label_invalid_zeros)


            image_valid = batch_data['image'].cuda()
            image_invalid_coor_cut = batch_data['image_cutout'].cuda()
            image_valid_rand_cuts = batch_data['random_cutouts'].cuda().permute(1,0,2,3,4)

            b_valid_size = len(image_valid)
            b_valid_coor_cut_size = len(image_invalid_coor_cut)
            b_valid_rand_cut_size = image_valid_rand_cuts.shape[0] * image_valid_rand_cuts.shape[1]
            
            for idx, image in enumerate(image_valid_rand_cuts):
                logits_valid_rand_cut = discriminator(image)
                label_valid_rand_cut_ones = torch.ones(len(image), 1).cuda()
                correct += (logits_valid_rand_cut>=valid_threshold).sum().item() 
                if idx == 0:
                    loss_valid_rand_cut = criterion(logits_valid_rand_cut, label_valid_rand_cut_ones)
                else:
                    loss_valid_rand_cut = loss_valid_rand_cut \
                                        + criterion(logits_valid_rand_cut, label_valid_rand_cut_ones)


            logits_valid = discriminator(image_valid)
            logits_invalid_coor_cut = discriminator(image_invalid_coor_cut)

            label_valid_ones = torch.ones(b_valid_size, 1).cuda()
            label_valid_coor_cut_zeros = torch.zeros(b_valid_coor_cut_size, 1).cuda()

            loss_valid = criterion(logits_valid, label_valid_ones)
            loss_valid_coor_cut = criterion(logits_invalid_coor_cut, label_valid_coor_cut_zeros)

            
            loss = loss_valid \
                    + loss_valid_coor_cut \
                    + loss_valid_rand_cut  

            if invalid_csv is not None:
                loss += loss_invalid \
                    + loss_invalid_cut \

            loss.backward()
            solver.step()
        
            correct += (logits_valid>=valid_threshold).sum().item() \
                    + (logits_invalid_coor_cut<valid_threshold).sum().item() 
            if invalid_csv is not None:
                correct += (logits_invalid<valid_threshold).sum().item() 


            total += b_valid_size \
                    + b_valid_coor_cut_size  \
                    + b_valid_rand_cut_size  

            if invalid_csv is not None:
                total += b_invalid_size  \
                         + b_invalid_cut_size \

            batch_index += 1

            if (counter) % 10 == 0:    # every 20 mini-batches...
                print('Train batch {}:\tD_loss: {:.10f} acc_training_D: {:.3f}%  {}/{}'.format(
                        counter,
                        loss.item(),
                        100 * correct / total,
                        correct,
                        total))

                if loss.item() < loss_min:
                    loss_min = loss.item()
                    torch.save(discriminator.state_dict(), dir_weight)
                    print('model saved to ' + dir_weight)
                

if __name__ == "__main__":
    train()

