import numpy as np
import cv2 as cv
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

# Load the model architecture
layers = get_lenet()
params = init_convnet(layers)

# Load the network
data = loadmat('../results/lenet.mat')
params_raw = data['params']

for params_idx in range(len(params)):
    raw_w = params_raw[0,params_idx][0,0][0]
    raw_b = params_raw[0,params_idx][0,0][1]
    assert params[params_idx]['w'].shape == raw_w.shape, 'weights do not have the same shape'
    assert params[params_idx]['b'].shape == raw_b.shape, 'biases do not have the same shape'
    params[params_idx]['w'] = raw_w
    params[params_idx]['b'] = raw_b

# Load data
fullset = False
xtrain, ytrain, xvalidate, yvalidate, xtest, ytest = load_mnist(fullset)
m_train = xtrain.shape[1]

batch_size = 1
layers[0]['batch_size'] = batch_size

rowNum = 4
colNum = 5
 
#Reference on how to plot well: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
output = convnet_forward(params, layers, xtest[:,0:1])
fig,axs = plt.subplots(rowNum, colNum)


testLayer = 1
for i in range(rowNum):
    for j in range(colNum):
        h = output[testLayer]['height']
        w = output[testLayer]['width']
        channel = output[testLayer]['channel']
        output_1 = np.reshape(output[testLayer]['data'], (h,w, channel), order='F')
        img = cv.rotate(output_1[:,:,i + j], cv.ROTATE_90_CLOCKWISE)
        img = cv.flip(img, 1)
        axs[i,j].imshow(img, cmap='gray')
fig.figure.savefig('../4_1_conv_layer')


testLayer = 2
for i in range(rowNum):
    for j in range(colNum):
        h = output[testLayer]['height']
        w = output[testLayer]['width']
        channel = output[testLayer]['channel']
        output_1 = np.reshape(output[testLayer]['data'], (h,w, channel), order='F')
        img = cv.rotate(output_1[:,:,i + j], cv.ROTATE_90_CLOCKWISE)
        img = cv.flip(img, 1)
        axs[i,j].imshow(img, cmap='gray') 
fig.figure.savefig('../4_1_relu_layer')


