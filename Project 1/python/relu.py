import numpy as np

def relu_forward(input_data):
    output = {
        'height': input_data['height'],
        'width': input_data['width'],
        'channel': input_data['channel'],
        'batch_size': input_data['batch_size'],
    }

    ###### Fill in the code here ######
    # Replace the following line with your implementation.
    output['data'] = np.zeros_like(input_data['data'])
   
    data = input_data['data']
    result = np.zeros_like(input_data['data'])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            result[i,j] = np.maximum(data[i,j], 0)
    output['data'] = result
    return output



def relu_backward(output, input_data, layer):
    input_od = np.zeros_like(input_data['data'])
    data = input_data['data']

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (data[i,j] >= 0) :
                input_od[i,j] = output['diff'][i,j]
            else:
                input_od[i,j] = 0
    return input_od
