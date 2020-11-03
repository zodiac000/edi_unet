import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import ToTensor, Resize

from PIL import Image
from random import randint
import numpy as np
from numpy import unravel_index
import math

from unet.unet_model import UNet
from Dataset import WeldingDatasetToTensor
from utils import accuracy_sum, heatmap_to_coor, crop, get_crop_hmap
from test_unet import my_eval

from pdb import set_trace

batch_size = 20

num_epochs = 20
lr = 1e-2
num_workers = 12

train = '10b_10'
valid = '5b'
directory = '15b'

train_csv = './csv/' + directory + '/' + train + '.csv'
saved_weight_dir = './check_points/weights_' + train + '_1.pth'
saved_weight_dir2 = './check_points/weights_' + train + '_2.pth'

validation_csv = './csv/' + directory + '/' + valid + '.csv'

model = UNet().cuda()
model2 = UNet().cuda()

labeled_dataset = WeldingDatasetToTensor(data_root='all_images', csv_file=train_csv, root_dir='./')
val_dataset = WeldingDatasetToTensor(data_root='all_images', csv_file=validation_csv, root_dir='./')

train_loader = DataLoader(labeled_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=60, num_workers=8, shuffle=False)

criterion = nn.MSELoss().cuda()
criterion2 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
# optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr, momentum=0.9)
def train():
    max_total_acc_x = 0
    max_euclidean_distance = 99999
    min_loss = 99999
    batch_counter = 0
    for epoch in range(num_epochs):
        for batch_index, sample_batched in enumerate(train_loader):

            inputs = sample_batched['image'].cuda()
            labels = sample_batched['hmap'].cuda()
            coors_bc = sample_batched['coor_1'].cpu().detach().numpy()
            origin_imgs = sample_batched['origin_img']

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.float(), labels.float())

            loss.backward()
            optimizer.step()

            x_delta = randint(-50, 50)
            y_delta = randint(-50, 50)
            coors_bc_delta = coors_bc + [x_delta, y_delta]
            for index, out in enumerate(outputs):
                crop_image, crop_hmap, x_shift, y_shift = get_crop_hmap(origin_imgs[index], coors_bc_delta[index], x_delta, y_delta)

                crop_image = crop_image.unsqueeze(0).unsqueeze(0).float()
                crop_hmap = torch.from_numpy(crop_hmap).unsqueeze(0).unsqueeze(0)
                if index == 0:
                    stacked_crop_image = crop_image
                    stacked_crop_hmap = crop_hmap
                else:
                    stacked_crop_image = torch.cat([stacked_crop_image, crop_image], dim=0)
                    stacked_crop_hmap = torch.cat([stacked_crop_hmap, crop_hmap], dim=0)

            
            optimizer2.zero_grad()
            outputs_crop = model2(stacked_crop_image.cuda())
            loss = criterion2(outputs_crop.float(), stacked_crop_hmap.cuda().float())
            loss.backward()
            optimizer2.step()

            batch_counter += 1
            if (batch_counter) % 5 == 0:    # every 20 mini-batches...
                print('Train batch_counter/epoch: {}/{}\tLoss: {:.30f}'.format(
                        batch_counter,
                        epoch,
                        torch.mean(loss).item())) #/ len(inputs)))

                # if loss.item() < min_loss:
                    # min_loss = loss.item()
                    # torch.save(model.state_dict(), saved_weight_dir)
                    # print('model saved to ' + saved_weight_dir)
                    # torch.save(model2.state_dict(), saved_weight_dir2)
                    # print('model2 saved to ' + saved_weight_dir2)





                if batch_counter % 25 == 0:
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
                            origin_imgs = batch['origin_img']
                            img_names = batch['img_name']

                            outputs = model(inputs)

                            for index, out in enumerate(outputs):
                                out = out.cpu().detach().numpy()
                                prediction = np.array(heatmap_to_coor(out.squeeze()))
                                prediction = (prediction * [1280/224, 1024/224]).astype(int)
                                crop_image, _, x_shift, y_shift = get_crop_hmap(origin_imgs[index], prediction)

                                crop_image = crop_image.unsqueeze(0).unsqueeze(0).float()
                                shift = np.expand_dims(np.array([x_shift, y_shift]), axis=0)
                                if index == 0:
                                    stacked_crop_image = crop_image
                                    stacked_shift = shift
                                else:
                                    stacked_crop_image = torch.cat([stacked_crop_image, crop_image], dim=0)
                                    stacked_shift = np.concatenate((stacked_shift, shift))


                            outputs_crop = model2(stacked_crop_image.cuda())
                            outputs_crop = outputs_crop.cpu().detach().numpy()


                            loss = criterion(outputs, labels)
                            valid_loss += loss.item()

                            outputs = outputs.cpu().detach().numpy()
                            labels = labels.cpu().detach().numpy()
                            
                            sum_acc_x, sum_acc_y, _, _ = accuracy_sum(outputs, coors_bc)
                            total_acc_x += sum_acc_x
                            total_acc_y += sum_acc_y

                            for index, out in enumerate(outputs_crop):
                                coor_pred = np.array(heatmap_to_coor(out.squeeze()))
                                shift = stacked_shift[index]
                                coor_pred = coor_pred + shift

                                dist = np.sum((coor_pred - coors_bc[index]) ** 2) ** 0.5
                                distances.append(dist)

                        distances = np.array(distances)
                        valid_loss = valid_loss / len(valid_loader)
                        print('Valid loss {}'.format(valid_loss))


                        print("=" * 30)
                        print("total acc_x = {:.20f}".format(total_acc_x/len(valid_loader.dataset)))
                        print("total acc_y = {:.20f}".format(total_acc_y/len(valid_loader.dataset)))
                        print("Euclidean Distance: {}".format(np.mean(distances)))
                        print("=" * 30)
                    
                        # if np.mean(distances) < max_euclidean_distance:
                            # max_euclidean_distance = np.mean(distances)
                        if valid_loss < min_loss:
                            min_loss = valid_loss
                            torch.save(model.state_dict(), saved_weight_dir)
                            print('model saved to ' + saved_weight_dir)
                            torch.save(model2.state_dict(), saved_weight_dir2)
                            print('model2 saved to ' + saved_weight_dir2)
                    
train()
