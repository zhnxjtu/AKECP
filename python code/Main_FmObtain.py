import os
os.mkdir('./res110_pruning_fmValue')

for layer_num in range(1,109):
    print('Calculate feature maps of the {}-th layer'.format(layer_num))

    # Here, we input 10 images as an example,
    # where the first image is used for pruning in the paper
    os.system('python FeatureMap_obtain-' + str(layer_num) + '.py')