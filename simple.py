import numpy as np

inputs = [
    [1,2,3,2.5],
    [2,5,-1,2],
    [-1.5,2.7,3.3,-0.8]
]

weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

biases = [2,3,0.5]

weights_l2 = [
    [0.1,-0.14,0.5],
    [-0.5,0.12,-0.33],
    [-0.44,0.73,-0.13]
]

biases_l2 = [-1,2,-0.5]

layer1_output = np.dot(inputs, np.array(weights).T) + biases
layer2_output = np.dot(layer1_output, np.array(weights_l2).T) + biases_l2

# Parametrs are neurons * weighted inputs + biases
# example: 
#       first layer  => (3n *4wi) + 3b       => 15p
#       second layer => (3n *3wi) + 3b       => 12p
#       total        => l1p + l2p = 15 + 12  => 27 parameters

print(layer2_output)



