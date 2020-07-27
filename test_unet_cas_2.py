from unet.unet_model import UNet
# from Student2 import Student2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from Dataset import WeldingDatasetToTensor
from pdb import set_trace
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
from PIL import Image
import matplotlib.pyplot as plt
from utils import heatmap_to_coor, accuracy_sum, get_crop_hmap
from tqdm import tqdm


def my_eval():
    ########################################    Transformed Dataset

    file_to_read = './csv/all/all.csv'
    file_to_write = "./csv/all/pred_all.csv"


    # file_to_read = './csv/pass_valid_tail_1000.csv'
    # file_to_write = "./csv/pred_pass_valid_tail_1000.csv"


    # file_to_read = './csv/fusion_927.csv'
    # file_to_write = './csv/fusion_927_pred.csv'


    # saved_weights = './check_points/weights_unet_200_1_cascade_2_1.pth'
    # saved_weights2 = './check_points/weights_unet_200_1_cascade_2_2.pth'
    saved_weights = './check_points/weights_100_1.pth'
    saved_weights2 = './check_points/weights_100_2.pth'

    batch_size = 40

    dataset = WeldingDatasetToTensor(csv_file=file_to_read, data_root='all_images')


    valid_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model = UNet().cuda()
    model.load_state_dict(torch.load(saved_weights))

    model2 = UNet().cuda()
    model2.load_state_dict(torch.load(saved_weights2))
    criterion = nn.MSELoss()

    # model.eval()
    valid_loss = 0
    f = open(file_to_write, "w")
    pbar = tqdm(total=len(valid_loader.dataset))
    with torch.no_grad():
        distances = []
        accuracy = []
        for i, batch in enumerate(valid_loader):
            inputs = batch['image'].float().cuda()
            labels = batch['hmap'].float().cuda()
            coors_bc = batch['coor_1'].numpy()
            img_names = batch['img_name']
            origin_imgs = batch['origin_img']
            outputs = model(inputs)

            valid_loss += criterion(outputs, labels)
            outputs = outputs.cpu().detach().numpy()


            for index, out in enumerate(outputs):
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


            for index, out in enumerate(outputs_crop):
                coor_pred = np.array(heatmap_to_coor(out.squeeze()))
                shift = stacked_shift[index]
                coor_pred = coor_pred + shift

                dist = np.sum((coor_pred - coors_bc[index]) ** 2) ** 0.5
                acc = [1, 1] - (np.absolute(coor_pred - coors_bc[index]) / [1280, 1024])

                f.write(img_names[index]\
                        + ',' + str(coors_bc[index][0]) \
                        + ',' + str(coors_bc[index][1]) \
                        + ',' + str(coor_pred[0]) \
                        + ',' + str(coor_pred[1])\
                        + '\n')
                distances.append(dist)
                accuracy.append(acc)

                pbar.update()

            
        e_distances = np.asarray(distances)
        accuracy = np.asarray(accuracy)
        accuracy = np.mean(accuracy, axis=0)

        

        print("=" * 30)
        print("mean acc_x = {:.10f}".format(accuracy[0]))
        print("mean acc_y = {:.10f}".format(accuracy[1]))
        print("Euclidean Distance: {}".format(np.mean(e_distances)))
        print("=" * 30)

        f.close()
        valid_loss = valid_loss / len(valid_loader.dataset)
        print('valid loss {}'.format(valid_loss))

if __name__ == "__main__":
    my_eval()

