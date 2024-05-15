import numpy as np
#from https://learner-cares.medium.com/handwritten-digit-recognition-using-convolutional-neural-network-cnn-with-tensorflow-2f444e6c4c31#6798


def inner_product_forward(input, layer, param):
    """
    Forward pass of inner product layer.

    Parameters:
    - input (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """

    d, k = input['data'].shape
    n = param['w'].shape[1]

    ###### Fill in the code here ######

    outputData = np.zeros((n, k))
    res = np.zeros((n,k))

    #Loop through and use the Inner Product formula:
    #f(x) = w*x + b
    for i in range(k): 
        data = input['data'][:,i]
        data = data.reshape(-1,1) #to make my 1D array work
        outputData[:,i] = np.add(np.dot(np.transpose(data), param['w']), param['b'])

    # Initialize output data structure
    output = {
        "height": n,
        "width": 1,
        "channel": 1,
        "batch_size": k,
        "data": outputData # replace 'data' value with your implementation
    }

    return output 


def inner_product_backward(output, input_data, layer, param):
    """
    Backward pass of inner product layer.

    Parameters:
    - output (dict): Contains the output data.
    - input_data (dict): Contains the input data.
    - layer (dict): Contains the configuration for the inner product layer.
    - param (dict): Contains the weights and biases for the inner product layer.
    """
    param_grad = {}
    ###### Fill in the code here ######
    # Replace the following lines with your implementation.
    
    param_grad['b'] = np.zeros_like(param['b'])
    param_grad['w'] = np.zeros_like(param['w'])
    data = input_data['data']
    
    b = param_grad['b']
    b = np.sum(output['diff'],axis=1) 
    b = b.reshape(-1,1)
    b = np.transpose(b)
    param_grad['b'] = b

    param_grad['w'] = np.dot(data, (np.transpose(output['diff'])))

    input_od = np.dot(param['w'], output['diff'])

    return param_grad, input_od
