import numpy as np

def separate_images(data):
    images = []
    number = 0
    for image_line in data:
        number += 1
        if number % 10 == 0:
            row = []
            for pixel in image_line[:-1]:
                if len(row) < 28:
                    row.append(pixel)
                else:
                    image.append(row)
                    row = [pixel]
            image.append(row)
            image = np.array(image)

            images.append(image)
            image = []
    images = np.array(images)

    return images


def sigmoid(x):
    return 1.0/(1+np.exp(-x))
    #return max(0, x)

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
    #return max(0, x)/x

def generate_label(n):
    label = np.zeros(10)
    label[n] = 1
    return label

class Network(object):

    def __init__(self, sizes):

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = np.array([np.random.randn(y) for y in sizes[1:]])
        self.activations = []
        self.z = []
        self.weights = np.array([np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])])

    def recognise(self, input):
        self.activations.append(input)
        for w, b in zip(self.weights, self.biases):
            self.z.append(np.dot(w,self.activations[-1])+b)
            self.activations.append(sigmoid(self.z[-1]))
        return self.activations[-1]

    def gradient(self, desire):

        L = self.num_layers-1

        grad_w = []
        dCda_old_array = []
        grad_w_layer = []
        grad_w_layer_row = []

        grad_b = []
        grad_b_layer = []

        while L > 0:

            for j in range(self.sizes[L]):

                dCda = 0
                dCda_new_array = []
                if L != self.num_layers-1:
                    for k in range(len(dCda_old_array)):
                        dCda+=dCda_old_array[k]*sigmoid_prime(self.z[L][k])*self.weights[L][k][j]

                else:
                    dCda = 2*(self.activations[-1][j]-desire[j])
                
                dCda_new_array.append(dCda)
                
                grad_w_layer_row = [dCda*(sigmoid_prime(self.z[L-1][j])*self.activations[L-1][i]) 
                                        for i in range(self.sizes[L-1])]

                grad_w_layer.append(grad_w_layer_row)
                grad_w_layer_row = []

                grad_b_layer.append(dCda*sigmoid_prime(self.z[L-1][j]))

            dCda_old_array = dCda_new_array

            grad_w.append(np.array(grad_w_layer))
            grad_w_layer = []

            grad_b.append(np.array(grad_b_layer))
            grad_b_layer = []

            L-=1
        grad_w = np.flip(np.array(grad_w), 0)
        grad_b = np.flip(np.array(grad_b), 0)
        return grad_w, grad_b
    
    def gradient2(self, desire):

        L = self.num_layers-1

        grad_w = []
        grad_b = []

        while L > 0:

            if (L == self.num_layers-1):
                #dCdz = 2*(self.activations[-1]-desire)*sigmoid_prime(self.z[-1])
                dCdz = (self.activations[-1]-desire)/self.activations[-1]/(1-self.activations[-1])*sigmoid_prime(self.z[-1])

            else:
                dCdz = np.dot(self.weights[L].transpose(), dCdz)*sigmoid_prime(self.z[L-1])


            grad_b.append(dCdz)
            dCdz_col = dCdz[:, np.newaxis]
            grad_w.append(dCdz_col*self.activations[L-1])

            L-=1

        grad_w = np.flip(np.array(grad_w), 0)
        grad_b = np.flip(np.array(grad_b), 0)

        return grad_w, grad_b

    def cost(self, desire):
        C = 0
        for a, d in zip(self.activations[-1], desire):
            C += (a-d)*(a-d)
        return C

    def descend(self, grad_w, grad_b, rate=0.2):
        for i in range(len(grad_w)):
            self.weights[i] -= rate*grad_w[i]
            self.biases[i]  -= rate*grad_b[i]

    def test(self, test_data):
        score = 0
        for data in test_data:
            result = self.recognise(data[0])
            result = result.tolist()
            if result.index(max(result)) == data[1]:
                score+=1
        return score

    def test2(self, test_data):
        score = 0
        for data in test_data:
            result = self.recognise(data[0])
            result = result.tolist()
            C = self.cost(generate_label(data[1]))
            if C < 0.2 and result.index(max(result)) == data[1]:
                score+=1
        return score





