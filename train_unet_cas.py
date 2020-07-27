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


#subset of the dataset
# indices = list(range(len(labeled_dataset)//4))
# labeled_dataset = Subset(labeled_dataset, indices)
# train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

saved_weight_dir = './check_points/weights_unet_5_2_cas.pth'

train_csv = './csv/7415_5_2.csv'
validation_csv = './csv/fusion_label_valid_927.csv'

# training_size = len(labeled_dataset)
# writer = SummaryWriter(tensorboard_file)

model = UNet().cuda()
# load_weights = './check_points/weights_7415_3_1.pth'
# model.load_state_dict(torch.load(load_weights))

labeled_dataset = WeldingDatasetToTensor(data_root='all_images', csv_file=train_csv, root_dir='./')
val_dataset = WeldingDatasetToTensor(data_root='all_images', csv_file=validation_csv, root_dir='./')

train_loader = DataLoader(labeled_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=40, num_workers=4, shuffle=False)
# dataloader_iterator = iter(train_loader)

criterion = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def train():
    #cross validation
    # order = np.random.RandomState().permutation(len(labeled_dataset))
    # train_dataset, val_dataset = split_dataset(labeled_dataset, int(training_size*0.9), order)
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


            x_delta = randint(-50, 50)
            y_delta = randint(-50, 50)
            coors_bc_delta = coors_bc + [x_delta, y_delta]
            # print("random x, y {}, {}".format(x_delta, y_delta))
            for index, out in enumerate(outputs):
                crop_image, crop_hmap, x_shift, y_shift = get_crop_hmap(origin_imgs[index], coors_bc_delta[index], x_delta, y_delta)
                # y, x = unravel_index(crop_hmap.argmax(), crop_hmap.shape)
                # print("label: {}".format(coors_bc[index]))
                # print("after random shift: {}".format(coors_bc_delta[index]))
                # print("hmap x, y: {}, {}".format(x, y))
                # print("x_shift: {}, y_shift: {}".format(x_shift, y_shift))
                # print("restate x: {}, y: {}".format(x+x_shift, y+y_shift))

                crop_image = crop_image.unsqueeze(0).unsqueeze(0).float()
                crop_hmap = torch.from_numpy(crop_hmap).unsqueeze(0).unsqueeze(0)
                if index == 0:
                    stacked_crop_image = crop_image
                    stacked_crop_hmap = crop_hmap
                else:
                    stacked_crop_image = torch.cat([stacked_crop_image, crop_image], dim=0)
                    stacked_crop_hmap = torch.cat([stacked_crop_hmap, crop_hmap], dim=0)

            
            outputs_crop = model(stacked_crop_image.cuda())
            loss = criterion(outputs_crop.float(), stacked_crop_hmap.cuda().float())

            loss.backward()
            optimizer.step()




            if (batch_index+1) % 5 == 0:    # every 20 mini-batches...
                # e_distance = 0
                # for index, out in enumerate(outputs):
                    # x, y = heatmap_to_coor(out.reshape(224, 224).cpu().detach().numpy())
                    # e_distance += ((int(x/224*1280)-coors_bc[index][0])**2 + \
                                    # (int(y/224*1024)-coors_bc[index][1])**2)**0.5

                # torch.save(model, './empty.pth')
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


                        outputs_crop = model(stacked_crop_image.cuda())
                        outputs_crop = outputs_crop.cpu().detach().numpy()




                        loss = criterion(outputs, labels)

                        outputs = outputs.cpu().detach().numpy()
                        labels = labels.cpu().detach().numpy()
                        
                        sum_acc_x, sum_acc_y, _, _ = accuracy_sum(outputs, coors_bc)
                        total_acc_x += sum_acc_x
                        total_acc_y += sum_acc_y

                        for index, out in enumerate(outputs_crop):
                            # x, y = heatmap_to_coor(out.reshape(224, 224))
                            # e_distance += ((int(x/224*1280)-coors_bc[index][0])**2 + \
                                            # (int(y/224*1024)-coors_bc[index][1])**2)**0.5

                            coor_pred = np.array(heatmap_to_coor(out.squeeze()))
                            shift = stacked_shift[index]
                            coor_pred = coor_pred + shift

                            # coor_pred = (coor_pred * [1280/224, 1024/224]).astype(int)
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
