from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

FileIndex = 24
os.mkdir('./res110_pruning_fmValue/FM' + str(FileIndex) + '_value')

import resnet
import argparse

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))

print(model_names)

parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet110',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)


def get_image_info(image_dir):
    image_info = Image.open(image_dir).convert('RGB')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    image_info = transform(image_info)
    image_info = image_info.unsqueeze(0)
    return  image_info

if __name__ == '__main__':
    for num_fig in range(1,11):
        image_dir = '../Image/CIFAR-10/'+ str(num_fig) +'.png'

        global args
        args = parser.parse_args()
        model = resnet.__dict__[args.arch]()

        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_gpu else "cpu")
        model = nn.DataParallel(model)
        model.to(device)

        model_para = torch.load('./res110-ori.th')
        pretrained_dict = model_para['state_dict']
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        image_info = get_image_info(image_dir)
        image_info = image_info.to(device)

        x = image_info

        ff = open('./res110_pruning_fmValue/FM' + str(FileIndex) + '_value/' + str(num_fig) + '.txt', mode='w')

        with torch.no_grad():
            for index, layer in model.module._modules.items():
                x = layer(x)
                if index == 'bn1':
                    break

            for index, layer in model.module.layer1._modules.items():
                x = layer(x)
                if index == '10':
                    break

            for index, layer in model.module.layer1[11]._modules.items():
                x = layer(x)
                if index == 'bn1':
                    break

            for index, layer in model.module.layer1[11]._modules.items():
                if index == 'conv2':
                    print('... Extract feature map the k-th layer')
                    layer_tmp = layer
                    [s1, s2, s3, s4] = layer_tmp.weight.size()
                    x_now = torch.tensor(np.zeros((1, s1 * s2, x.size()[2], x.size()[3])))
                    x_now = x_now.cuda()
                    for output_num in range(0, s1):
                        for input_num in range(0, s2):
                            channel_tmp = nn.Conv2d(1, 1, kernel_size=s3, stride=1, padding=1, bias=None)
                            channel_tmp.weight = nn.Parameter(
                                layer_tmp.weight[output_num][input_num].reshape(1, 1, s3, s4))
                            x_now[0][s2 * ((output_num + 1) - 1) + (input_num + 1) - 1] = \
                                channel_tmp(x[0][input_num].reshape(1, 1, x[0][input_num].size()[0],
                                                                   x[0][input_num].size()[1]))

                            # save matrix
                            x_tmp = x_now[0][s2 * ((output_num + 1) - 1) + (input_num + 1) - 1].cpu().numpy()
                            print('...The size of output FMs: {},{} '.format(x_tmp.shape[0], x_tmp.shape[1]))
                            for x_tmp_col in range(0, x_tmp.shape[1]):
                                for x_tmp_row in range(0, x_tmp.shape[0]):
                                    x_save = x_tmp[x_tmp_row][x_tmp_col]
                                    ff.write('{} \n'.format(x_save))