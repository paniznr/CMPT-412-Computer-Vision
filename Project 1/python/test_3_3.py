import numpy as np
import cv2 as cv
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt

#Loading my images in

img = np.zeros((28*28, 5))

imgInit = cv.imread('../my_images/1.png')
imgGray = cv.cvtColor(imgInit, cv.COLOR_BGR2GRAY)
imgGray = cv.resize(imgGray, (28,28), interpolation=cv.INTER_LINEAR)
imgGray = imgGray.astype(np.float32)
imgGray = imgGray/255.0
imgShaped = np.reshape(imgGray, (28*28,-1))
img[:,0] = imgShaped.T

imgInit = cv.imread('../my_images/2.png')
imgGray = cv.cvtColor(imgInit, cv.COLOR_BGR2GRAY)
imgGray = cv.resize(imgGray, (28,28), interpolation=cv.INTER_LINEAR)
imgGray = imgGray.astype(np.float32)
imgGray = imgGray/255.0
imgShaped = np.reshape(imgGray, (28*28,-1))
img[:,1] = imgShaped.T

imgInit = cv.imread('../my_images/3.png')
imgGray = cv.cvtColor(imgInit, cv.COLOR_BGR2GRAY)
imgGray = cv.resize(imgGray, (28,28), interpolation=cv.INTER_LINEAR)
imgGray = imgGray.astype(np.float32)
imgGray = imgGray/255.0
imgShaped = np.reshape(imgGray, (28*28,-1))
img[:,2] = imgShaped.T

imgInit = cv.imread('../my_images/4.png')
imgGray = cv.cvtColor(imgInit, cv.COLOR_BGR2GRAY)
imgGray = cv.resize(imgGray, (28,28), interpolation=cv.INTER_LINEAR)
imgGray = imgGray.astype(np.float32)
imgGray = imgGray/255.0
imgShaped = np.reshape(imgGray, (28*28,-1))
img[:,3] = imgShaped.T

imgInit = cv.imread('../my_images/5.png')
imgGray = cv.cvtColor(imgInit, cv.COLOR_BGR2GRAY)
imgGray = cv.resize(imgGray, (28,28), interpolation=cv.INTER_LINEAR)
imgGray = imgGray.astype(np.float32)
imgGray = imgGray/255.0
imgShaped = np.reshape(imgGray, (28*28,-1))
img[:,4] = imgShaped.T

# Load the model architecture
layers = get_lenet()
layers[0]['batch_size'] = img.shape[1]
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

# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []

cptest, P = convnet_forward(params, layers, img, test=True)
pMax = np.argmax(P, axis=0)
all_preds.extend(pMax)
ytest = [1,2,3,4,5]

print("True:       ", ytest)
print("Prediction: ", all_preds)

