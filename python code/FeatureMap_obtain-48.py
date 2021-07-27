from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

FileIndex = 48
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
                    ' (default: resnet110)')

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
                if index == 'layer1':
                    break

            for index, layer in model.module.layer2._modules.items():
                x = layer(x)
                if index == '4':
                    break

            for index, layer in model.module.layer2[5]._modules.items():
                x = layer(x)
                if index == 'bn1':
                    break

            for index, layer in model.module.layer2[5]._modules.items():
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