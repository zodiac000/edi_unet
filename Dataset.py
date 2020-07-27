import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
from skimage import io, transform
from torchvision import utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from utils import CoorToHeatmap, cutout_image, crop_image
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.transforms.functional import crop
from PIL import Image
import matplotlib.pyplot as plt
from torch import cat
from random import randint

from pdb import set_trace


class WeldingDatasetToTensor(Dataset):
    """Welding dataset."""

    def __init__(self, data_root, csv_file, root_dir='.', size=224):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.data_root = data_root
        self.size = size

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, self.data_root,
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coor_1 = self.all_data.iloc[idx, 1:3]
        coor_1 = np.array(coor_1).astype(int)

        
        hmap = CoorToHeatmap(image_np.shape)(coor_1)
        hmap = torch.from_numpy(hmap).view(1, int(self.size), int(self.size))
        input_transform = Compose([Resize((self.size, self.size)), ToTensor()])

        image = input_transform(image_pil)

        sample = {
            'image': image,
            'hmap': hmap,
            'img_name': image_name,
            'coor_1': coor_1,
            'origin_img': image_np,
        }

        return sample




class CutoutDataset(Dataset):
    """Cutout image for classification."""

    def __init__(self, csv_file, root_dir='.', dist_lower_bound=10,
                 n_random_cutout=2):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.dist_lower_bound = dist_lower_bound
        self.n_random_cutout = n_random_cutout

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coors = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        # top_left = [max(0, x-cut_size//2) for x in coors[::-1]]
        # image_crop = crop(image_pil, *top_left, cut_size, cut_size)
        # image_crop = ToTensor()(image_crop)
        image_cutout = np.copy(image_np)
        cutout_image(image_cutout, coors)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image_cutout = input_transform(Image.fromarray(image_cutout))

        image = input_transform(image_pil)

        random_cutouts = []
        for i in range(self.n_random_cutout*4):
            while(True):
                random_delta = np.asarray(
                    [randint(-300, 300), randint(-300, 300)])
                random_coors = np.clip(coors + random_delta, 50, 950)
                dist = np.sum((random_coors - coors) ** 2) ** 0.5
                if (dist > self.dist_lower_bound):
                    image_random_cutout = np.copy(image_np)
                    cutout_image(image_random_cutout, random_coors)
                    image_random_cutout = input_transform(
                        Image.fromarray(image_random_cutout))
                    random_cutouts.append(image_random_cutout)
                    break
        random_cutouts = torch.stack(random_cutouts)

        # generating cutouts within certain Euclidean distance
        random_center_cutouts = []
        for i in range(self.n_random_cutout):
            while(True):
                random_delta = np.asarray([randint(-10, 10), randint(-10, 10)])
                random_coors = coors + random_delta
                dist = np.sum((random_coors - coors) ** 2) ** 0.5
                # print(random_coors, dist)
                if (dist < self.dist_lower_bound):
                    image_random_cutout = np.copy(image_np)
                    cutout_image(image_random_cutout, random_coors)
                    image_random_cutout = input_transform(
                        Image.fromarray(image_random_cutout))
                    random_center_cutouts.append(image_random_cutout)
                    break
        random_center_cutouts = torch.stack(random_center_cutouts)

        sample = {
            'image': image,
            'image_cutout': image_cutout,
            'random_cutouts': random_cutouts,
            'random_center_cutouts': random_center_cutouts,
        }

        return sample


class CutoutDataset_pred(Dataset):
    def __init__(self, csv_file, root_dir='.'):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coor_label = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        coor_pred = np.array(self.all_data.iloc[idx, 3:]).astype(int)
        class_gt = torch.ones(1) if coor_label[0] == -1 else torch.zeros(1)
        image_cutout = np.copy(image_np)
        cutout_image(image_cutout, coor_pred)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image_cutout = input_transform(Image.fromarray(image_cutout))
        image = input_transform(image_pil)

        sample = {
            'image_name': image_name,
            'coor_label': coor_label,
            'coor_pred': coor_pred,
            'image_cutout': image_cutout,
            'gt': class_gt,
        }

        return sample


class InvalidDataset(Dataset):
    """Invalid image for classification."""

    def __init__(self, csv_file, root_dir='.', n_random_cutout=2):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.n_random_cutout = n_random_cutout

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image = input_transform(image_pil)
        random_cutouts = []
        for i in range(self.n_random_cutout):
            random_coors = np.asarray([randint(100, 800), randint(100, 800)])
            image_random_cutout = np.copy(image_np)
            cutout_image(image_random_cutout, random_coors)
            image_random_cutout = input_transform(
                Image.fromarray(image_random_cutout))
            random_cutouts.append(image_random_cutout)

        random_cutouts = torch.stack(random_cutouts)

        sample = {
            'image': image,
            'random_cutouts': random_cutouts,
        }
        return sample


class CropDataset(Dataset):
    """Cutout image for classification."""

    def __init__(self, csv_file, root_dir='.', dist_lower_bound=10,
                 n_random_crop=1):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir
        self.dist_lower_bound = dist_lower_bound
        self.n_random_crop = n_random_crop

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coors = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        image_crop = np.copy(image_np)
        image_crop = crop_image(image_crop, coors)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image_crop = input_transform(image_crop)

        image = input_transform(image_pil)

        random_crops = []
        for i in range(self.n_random_crop):
            while(True):
                random_delta = np.asarray(
                    [randint(-300, 300), randint(-300, 300)])
                random_coors = np.clip(coors + random_delta, 50, 950)
                dist = np.sum((random_coors - coors) ** 2) ** 0.5
                if (dist > self.dist_lower_bound):
                    image_random_crop = np.copy(image_np)
                    image_random_crop = crop_image(image_random_crop, random_coors)
                    image_random_crop = input_transform(image_random_crop)
                    random_crops.append(image_random_crop)
                    break
        random_crops = torch.stack(random_crops)

        # # generating cutouts within certain Euclidean distance
        # random_center_cutouts = []
        # for i in range(self.n_random_crop):
            # while(True):
                # random_delta = np.asarray([randint(-10, 10), randint(-10, 10)])
                # random_coors = coors + random_delta
                # dist = np.sum((random_coors - coors) ** 2) ** 0.5
                # # print(random_coors, dist)
                # if (dist < self.dist_lower_bound):
                    # image_random_cutout = np.copy(image_np)
                    # cutout_image(image_random_cutout, random_coors)
                    # image_random_cutout = input_transform(
                        # Image.fromarray(image_random_cutout))
                    # random_center_cutouts.append(image_random_cutout)
                    # break
        # random_center_cutouts = torch.stack(random_center_cutouts)

        sample = {
            'image': image,
            'image_crop': image_crop,
            'random_crop': random_crops,
            # 'random_center_cutouts': random_center_cutouts,
        }

        return sample


class CropDataset_pred(Dataset):
    def __init__(self, csv_file, root_dir='.'):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coor_label = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        coor_pred = np.array(self.all_data.iloc[idx, 3:]).astype(int)
        class_gt = torch.zeros(1) if coor_label[0] == -1 else torch.ones(1)
        image_crop = np.copy(image_np)
        image_crop = crop_image(image_crop, coor_pred)
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image_crop = input_transform(image_crop)
        # image = input_transform(image_pil)

        sample = {
            'image_name': image_name,
            'image_crop': image_crop,
            'coor_label': coor_label,
            'coor_pred': coor_pred,
            'gt': class_gt,
        }

        return sample


class Hybrid(Dataset):
    def __init__(self, csv_file, root_dir='.'):
        self.all_data = pd.read_csv(csv_file, header=None)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_name = self.all_data.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, 'all_images/',
                                self.all_data.iloc[idx, 0])
        input_transform = Compose([Resize((224, 224)), ToTensor()])
        image_pil = Image.open(img_name)
        image_np = np.array(image_pil)
        coor_label = np.array(self.all_data.iloc[idx, 1:3]).astype(int)
        coor_pred = np.array(self.all_data.iloc[idx, 3:]).astype(int)

        # class_gt_cutout = torch.ones(1) if coor_label[0] == -1 else torch.zeros(1)
        class_gt_crop = torch.ones(1) if coor_label[0] != -1 else torch.zeros(1)

        image_cutout = np.copy(image_np)
        cutout_image(image_cutout, coor_pred)
        image_cutout = input_transform(Image.fromarray(image_cutout))

        image_crop = np.copy(image_np)
        image_crop = crop_image(image_crop, coor_pred)
        image_crop = input_transform(image_crop)

        sample = {
            'image_name': image_name,
            'coor_label': coor_label,
            'coor_pred': coor_pred,
            'image_cutout': image_cutout,
            'image_crop': image_crop,
            'class_gt_crop': class_gt_crop,
            # 'class_gt_cutout': class_gt_cutout,
        }

        return sample




if __name__ == '__main__':
    from torch.utils.data import DataLoader

    # dataset = InvalidDataset('csv/pass_invalid_85.csv')
    # dataset = CutoutDataset('csv/pass_valid_test_100.csv')
    dataset = CutoutDataset('csv/pass_valid_50.csv')
    # dataset = CropDataset('csv/pass_valid_50.csv')
    # dataset = CropDataset_pred('csv/pred_pass_valid_test_100.csv')

    dataloader = DataLoader(dataset, batch_size=1)
    iter_data = iter(dataloader)
    batch = next(iter_data)

    # ===============testing======================
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('tkagg')

    image = batch['image']
    plt.imshow(image[0].numpy().reshape(224, 224))
    plt.title('Original')
    plt.savefig('demo/original.png')

    # coor_cutout = batch['coor_cutout']
    # plt.imshow(coor_cutout[0].numpy().reshape(224, 224))
    # plt.title('Center Cutout')
    # plt.savefig('demo/center_cutout.png')

    # random_cutout = batch['random_cutouts']
    # plt.imshow(random_cutout.squeeze()[0].numpy().reshape(224, 224))
    # plt.title('Random Positive Cutout')
    # plt.savefig('demo/pos_cutout.png')

    # random_center_cutout = batch['random_center_cutouts']
    # plt.imshow(random_center_cutout.squeeze()[0].numpy().reshape(224, 224))
    # plt.title('Random Negative Cutout')
    # plt.savefig('demo/neg_cutout.png')
    # ====================================================
    
    # coor_crop = batch['coor_crop']
    # plt.imshow(coor_crop[0].numpy().reshape(224, 224))
    # plt.title('Center Crop')
    # plt.savefig('demo/center_crop.png')

    # random_crop = batch['random_crop']
    # plt.imshow(random_crop.squeeze(0)[0].numpy().reshape(224, 224))
    # plt.title('Random Crop')
    # plt.savefig('demo/random_crop.png')

    plt.show()
    # ===============testing======================
