from models.Discriminator import Discriminator
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import CropDataset
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

# from test_cutout_cls import eval_cutout_cls

from pdb import set_trace


lr = 1e-5
training_number = 200
num_epochs = 1000000
batch_size = 10
invalid_batch_size = batch_size

# train_csv = './csv/200_1.csv'
# dir_weight = 'check_points/weights_crop_200_1.pth'
name = 'train_discriminator_10'
train_csv = 'csv/' + name + '.csv'
dir_weight = 'check_points/weights_crop_' + name + '.pth'

discriminator = Discriminator().cuda()
# load_weights = './check_points/weights_crop_7415_1.pth'
# discriminator.load_state_dict(torch.load(load_weights))

dist_lower_bound = 10.0
train_dataset = CropDataset(csv_file=train_csv)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

criterion = nn.BCELoss()


# G_solver = torch.optim.Adam(student.parameters(), lr=lr_G)
solver = torch.optim.Adam(discriminator.parameters(), lr=lr)
# G_solver = torch.optim.RMSprop(student.parameters(), lr=lr_G)
# solver = torch.optim.RMSprop(discriminator.parameters(), lr=lr_D)
# G_solver = torch.optim.Adadelta(student.parameters(), lr=lr_G)
# solver = torch.optim.Adadelta(discriminator.parameters(), lr=lr_D)

def train():
    correct_rate = 0
    batch_index = 0
    loss_min = 999
    counter = 0
    for epoch in range(num_epochs):
        for i, batch_data in enumerate(train_loader):
            counter += 1
            correct = 0
            total = 0
            discriminator.zero_grad()

            image_valid_coor_crop = batch_data['image_crop'].cuda()
            image_valid_rand_crop = batch_data['random_crop'].cuda().permute(1,0,2,3,4)

            b_valid_coor_crop_size = len(image_valid_coor_crop)
            b_valid_rand_crop_size = image_valid_rand_crop.shape[0] * image_valid_rand_crop.shape[1]
            
            for idx, image in enumerate(image_valid_rand_crop):
                logits_valid_rand_crop = discriminator(image)
                label_valid_rand_crop_zeros = torch.zeros(len(image), 1).cuda()
                correct += (logits_valid_rand_crop<0.5).sum().item() 
                if idx == 0:
                    loss_valid_rand_crop = criterion(logits_valid_rand_crop, label_valid_rand_crop_zeros)
                else:
                    loss_valid_rand_crop = loss_valid_rand_crop \
                                        + criterion(logits_valid_rand_crop, label_valid_rand_crop_zeros)


            logits_valid_coor_crop = discriminator(image_valid_coor_crop)
            label_valid_coor_crop_ones= torch.ones(b_valid_coor_crop_size, 1).cuda()
            loss_valid_coor_crop = criterion(logits_valid_coor_crop, label_valid_coor_crop_ones)

            loss =  loss_valid_coor_crop \
                    + loss_valid_rand_crop  \
                    # + loss_valid \
                    # + loss_invalid \
                    # + loss_invalid_cut \

            loss.backward()
            solver.step()
        
            correct = correct  \
                    + (logits_valid_coor_crop>=0.5).sum().item() \
                    # + (logits_valid_rand_crop<0.5).sum().item() \
                    # + (logits_valid>=0.5).sum().item() \
                    # + (logits_invalid<0.5).sum().item() 
                    # + (logits_invalid_cut<0.5).sum().item() 

            total = total \
                    + b_valid_coor_crop_size  \
                    + b_valid_rand_crop_size  \
                    # + b_valid_size \
                    # + b_invalid_size  \
                    # + b_invalid_cut_size \

            # pbar.update()
            batch_index += 1

            if (counter) % 10 == 0:    # every 20 mini-batches...
                print('Train batch {}:\tD_loss: {:.10f} acc_training_D: {:.3f}%  {}/{}'.format(
                        counter,
                        # G_loss.item(),
                        loss.item(),
                        100 * correct / total,
                        correct,
                        total))

                if loss.item() < loss_min:
                    loss_min = loss.item()
                    torch.save(discriminator.state_dict(), dir_weight)
                    print('model saved to ' + dir_weight)

        # if (epoch+1) % 5 == 0:    # every 20 mini-batches...
            # eval_cutout_cls()

if __name__ == "__main__":
    train()

