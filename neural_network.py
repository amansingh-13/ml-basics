import numpy as np

training_input_data=np.array([[0,1],[1,0],[0,0],[1,1]])
training_output_data=np.array([[0],[0],[0],[1]])
# recreating AND

class NN101():
    def __init__(self):
        # understand its use later, kind of related to constructor ......
        random.seed(1)
        # initializing weights in range [-1,1] biases are ignored to avoid complexity
        # figure out what 2 and -1 do here .......
        self.weight_1 = 2 * random.random((4, 2)) - 1
        self.weight_2 = 2 * random.random((1, 4)) - 1
        self.nabla_weight_1 = 2 * random.random((4, 2)) - 1
        self.nabla_weight_2 = 2 * random.random((1, 4)) - 1
    
    def sigmoid (self,x,deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
    
    sigmoid_vector = np.vectorize(sigmoid)
    
    def operation (self, feed):
       #operates the weights on the input matrix, outputs number
       return self.sigmoid(np.matmul(self.weight_2, self.sigmoid_vector(np.matmul(self.weight_1, feed))))
    
    def train_and_backpropogate (self, training_input_data, training_output_data ):
        for iteration in range(1000):
            for i in range (4):
                
                cost=cost+(self.operation(training_input_data[i].tanspose())-training_output_data[i])**2
                for j in range (4):
                    nabla_weight_2.transpose()[j]=np.array([nabla_weight_2.transpose()[j]+
                                                  ((self.operation(training_input_data[i].tanspose())-training_output_data[i])*
                                                  self.sigmoid(self.operation(training_input_data[i].tanspose()),True)*
                                                  self.sigmoid(np.matmul(self.weight_1,training_input_data[i].tanspose())[j]))
                                                  ])
 
