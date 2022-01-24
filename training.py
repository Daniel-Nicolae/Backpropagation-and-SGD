import network
from pandas import read_csv
from random import shuffle
import numpy as np


data_train = read_csv("image_data_training.csv")
data_train = np.array(data_train, np.float)

data_test = read_csv("image_data_test.csv")
data_test = np.array(data_test, np.float)


training = []
testing = []

for line, label in zip(data_train[:, :-1], data_train[:, -1]):
    instance = (line/255, int(label))
    training.append(instance)

for line, label in zip(data_test[:, :-1], data_test[:, -1]):
    instance = (line/255, int(label))
    testing.append(instance)


digits_net = network.Network([784, 12, 10])
shuffle(training)

# Training parameters
epochs = 10
batch_size = 10
learing_rate = 2.0

print(digits_net.test(testing))

grad_w = digits_net.weights*0
grad_b = digits_net.biases*0

for e in range(epochs):

    for k in range(0, len(training), batch_size):

        batch = training[k:k+batch_size]
        grad_w*=0
        grad_b*=0

        for image in batch:
            digits_net.recognise(image[0])
            grad_w_new, grad_b_new = digits_net.gradient2(network.generate_label(image[1]))

            grad_w+=grad_w_new
            grad_b+=grad_b_new
        
        grad_w/=batch_size
        grad_b/=batch_size
        digits_net.descend(grad_w, grad_b, learing_rate)

    print("Epoch:", e, digits_net.test(testing))




