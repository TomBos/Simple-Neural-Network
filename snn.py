def generateNeuron(inputs, weights, bias):
    if len(inputs) != len(weights):
        return {
            "error": 1,
            "message": "Each input doesn't have its own weight!"
        }

    weighted_inputs = []

    for i, input in enumerate(inputs):
        weighted_inputs.append(input * weights[i])

    return sum(weighted_inputs) + bias


#====== START ======#

neuron_1_inputs = [1,2,3,2.5]
neuron_2_inputs = [1,2,3,2.5]
neuron_3_inputs = [1,2,3,2.5]


# This is the importance (weight) of the parameter being passed into the NN
neuron_1_weights = [0.2, 0.8, -0.5, 1]
neuron_2_weights = [0.5, -0.91, 0.26, -0.5]
neuron_3_weights = [-0.26, -0.27, 0.17, 0.87]

neuron_1_bias = 2
neuron_2_bias = 3
neuron_3_bias = 0.5

neuron_1 = generateNeuron(neuron_1_inputs, neuron_1_weights, neuron_1_bias)
neuron_2 = generateNeuron(neuron_2_inputs, neuron_2_weights, neuron_2_bias)
neuron_3 = generateNeuron(neuron_3_inputs, neuron_3_weights, neuron_3_bias) 


print(neuron_1, neuron_2, neuron_3)
