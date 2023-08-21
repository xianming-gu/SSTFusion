import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import warnings
from tqdm import trange, tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.loss import loss_sstfusion
from MyDataLoader import TrainData
from MyOption import args
# from model.SSTFusion import SSTFusion as net
from model.uformer import Uformer as net


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Mytrain(model_pretrain=None):
    # 设置随机数种子
    setup_seed(args.seed)
    model_path = args.model_save_path + '/' + str(args.epoch) + '/'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(args.img_save_dir, exist_ok=True)
    # os.makedirs('./modelsave')

    lr = args.lr

    # device handling
    if args.DEVICE == 'cpu':
        device = 'cpu'
    else:
        device = args.DEVICE
        # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # prepare model folder
    os.makedirs(args.temp_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.RandomCrop(args.patch_size, args.patch_size)])

    train_set = TrainData(transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=1,
                                               pin_memory=True)
    # model = net(img_size=args.imgsize, dim=256)
    # print('train datasets lenth:', len(train_loader))
    model = net(in_chans=1, dd_in=1)
    if model_pretrain is not None:
        model.load_state_dict(torch.load(model_pretrain, map_location=args.DEVICE))

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    loss_plt = []
    for epoch in range(0, args.epoch):
        # os.makedirs(args.result + '/' + '%d' % (epoch + 1), exist_ok=True)

        loss_mean = []
        for idx, data in enumerate(tqdm(train_loader, desc='[Epoch--%d]' % (epoch + 1))):
            # for idx, datas in tqdm(train_loader):

            # print(len(data))
            img = data
            img_down = F.max_pool2d(img, kernel_size=2, stride=2)
            img_tilde = F.upsample(img_down, scale_factor=2, mode='bilinear')
            # 训练模型
            model, loss_per_img = train(model, img, img_tilde, lr, device)
            loss_mean.append(loss_per_img)

        # print loss
        sum_list = 0
        for item in loss_mean:
            sum_list += item
        sum_per_epoch = sum_list / len(loss_mean)
        print('\tLoss:%.5f' % sum_per_epoch)
        loss_plt.append(sum_per_epoch.detach().cpu().numpy())

        # save info to txt file
        strain_path = args.temp_dir + '/temp_loss.txt'
        Loss_file = 'Epoch--' + str(epoch + 1) + '\t' + 'Loss:' + str(sum_per_epoch.detach().cpu().numpy())
        with open(strain_path, 'a') as f:
            f.write(Loss_file + '\r\n')

        # 每500epoch保存模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), model_path + str(epoch + 1) + '_' + args.model_save_name)
            print('model save in %s' % args.model_save_path)

    # 输出损失函数曲线
    plt.figure()
    x = range(0, args.epoch)  # x和y的维度要一样
    y = loss_plt
    plt.plot(x, y, 'r-')  # 设置输出样式
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(model_path + '/loss.png')  # 保存训练损失曲线图片
    plt.show()  # 显示曲线


def train(model, img1, img2, lr, device):
    model.to(device)
    model.train()

    img1 = img1.to(device)  # img
    img2 = img2.to(device)  # img_tilde

    opt = torch.optim.AdamW(model.parameters(), lr)

    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    img_hat = model(img2)
    img_hat = img_hat.to(device)

    # img_cat = torch.cat([img1, img2], dim=1)
    loss_total = loss_sstfusion(img_hat, img1)

    opt.zero_grad()
    loss_total.backward()
    opt.step()

    return model, loss_total


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    Mytrain()
