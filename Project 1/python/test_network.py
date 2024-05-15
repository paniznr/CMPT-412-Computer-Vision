import numpy as np
from utils import get_lenet
from load_mnist import load_mnist
from scipy.io import loadmat
from conv_net import convnet_forward
from init_convnet import init_convnet
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
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


# Testing the network
#### Modify the code to get the confusion matrix ####
all_preds = []

for i in range(0, xtest.shape[1], 100):
    cptest, P = convnet_forward(params, layers, xtest[:,i:i+100], test=True)
    pMax = np.argmax(P, axis=0)
    all_preds.extend(pMax)

confMat = confusion_matrix(np.transpose(ytest), all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=confMat)
disp = disp.plot()
disp.figure_.savefig('../3_2_confusion_Matrix')

