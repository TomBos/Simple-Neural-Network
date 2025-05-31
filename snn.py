import numpy as np

def generateNeuron(inputs, weights, bias):
    if len(inputs) != len(weights):
        return {
            "error": 1,
            "message": "Each input doesn't have its own weight!"
        }

    # Multiply elementwise each element at the same index for both vectors 
    # And add it all together
    return np.dot(weights, inputs) + bias


#====== START ======#

neuron_1_inputs = [1,2,3,2.5]
neuron_2_inputs = [2.0,5.0,-1.0,2.0]
neuron_3_inputs = [-1.5,2.7,3.3,-0.8]


# This is the importance (weight) of the parameter being passed into the NN
# Weights change the magnitude of the input impact
neuron_1_weights = [0.2, 0.8, -0.5, 1]
neuron_2_weights = [0.5, -0.91, 0.26, -0.5]
neuron_3_weights = [-0.26, -0.27, 0.17, 0.87]

# Similarly to weights, bias changes the ouput
# This time bias acts as an offset not as multiplier
neuron_1_bias = 2
neuron_2_bias = 3
neuron_3_bias = 0.5

output = [ generateNeuron(neuron_1_inputs, neuron_1_weights, neuron_1_bias), generateNeuron(neuron_2_inputs, neuron_2_weights, neuron_2_bias), generateNeuron(neuron_3_inputs, neuron_3_weights, neuron_3_bias) ] 

print(output)
