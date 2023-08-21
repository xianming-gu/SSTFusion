import os
import cv2
import argparse
import torch.nn as nn
import numpy as np
import random
import torch.utils.data as data
import torchvision.transforms as transforms
from MyOption import args


class TrainData(data.Dataset):  # MS-COCO Dataset (partial ~10000 images)
    def __init__(self, transform=None):
        super(TrainData, self).__init__()
        self.img_dir = os.listdir('./coco-dataset/')

        self.patch_size = args.patch_size
        self.transform = transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        img = cv2.imread('./coco-dataset/' + self.img_dir[index], cv2.IMREAD_GRAYSCALE)
        img = img.squeeze()

        if self.transform:
            img = self.transform(img)

        return img  # 1,256,256

    def get_patch(self, img):
        h, w = img.shape[:2]
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img_p = img[y:y + stride, x:x + stride]

        return img_p


class TestData(data.Dataset):
    def __init__(self, transform=None):
        super(TestData, self).__init__()
        self.transform = transform
        self.dir_prefix = args.dir_test

        self.img1_dir = os.listdir(self.dir_prefix + args.img_type1)
        self.img2_dir = os.listdir(self.dir_prefix + args.img_type2)

    def __getitem__(self, index):
        self.img1_dir.sort()
        self.img2_dir.sort()
        img_name = str(self.img1_dir[index])
        if args.img_type1 == 'CT/':
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + args.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img_name, img1, img2  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256
        else:
            img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index])
            # img1 = cv2.imread(self.dir_prefix + args.img_type1 + self.img1_dir[index], cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(self.dir_prefix + args.img_type2 + self.img2_dir[index], cv2.IMREAD_GRAYSCALE)

            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)  # CT/PET/SPECT 256,256,3

            img1_Y = img1[:, :, 0:1]
            img1_CrCb = img1[:, :, 1:3].transpose(2, 0, 1)

            if self.transform:
                img1_Y = self.transform(img1_Y)
                img2 = self.transform(img2)

            return img_name, img1_Y, img2, img1_CrCb  # img1[YCrCb]:3,256,256  img2[Gray]:1,256,256

    def __len__(self):
        assert len(self.img1_dir) == len(self.img2_dir)
        return len(self.img1_dir)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5]),
                                    transforms.RandomCrop(args.patch_size, args.patch_size)])

    TrainDataset = TrainData(transform=transform)
    train_loader = data.DataLoader(TrainDataset,
                                   batch_size=1,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=2,
                                   pin_memory=True)
    i = 0
    for idx, data in enumerate(train_loader):
        # print(len(data))
        img = data
        print(img)
        print(img.shape)
        i = i + 1
    print(i)  # 133

    # TestDataset = TestData(transform=transform)
    # test_loader = data.DataLoader(TestDataset,
    #                               batch_size=1,
    #                               shuffle=True,
    #                               drop_last=True,
    #                               num_workers=2,
    #                               pin_memory=True)
    # j = 0
    # for idx, [img_name, data] in enumerate(test_loader):
    #     # print(len(data))
    #     j = j + 1
    # print(j)  # 24
