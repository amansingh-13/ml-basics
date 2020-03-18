import numpy as np
training_input_data=np.array([[0,1],[1,0],[0,0],[1,1]])
training_output_data=np.array([[0],[0],[0],[1]])
def sigmoid (x, deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
    
sigmoid_vector = np.vectorize(sigmoid)
# recreating AND



class NN101():
    def __init__(self):
        # understand its use later, kind of related to constructor ......
        np.random.seed(1)
        # initializing weights in range [-1,1] biases are ignored to avoid complexity
        # 2*[0,1]-1=[-1,1]
        self.weight_1 = 2 * np.random.random((4, 2)) - 1
        self.weight_2 = 2 * np.random.random((1, 4)) - 1
        
    
    
    
    def operation (self, feed):
       #operates the weights on the input matrix, outputs number
        return sigmoid(np.dot(self.weight_2, sigmoid_vector(np.dot(self.weight_1, feed)))) # PROBLEM !!!!!
    
    def train_and_backpropogate (self, training_input_data, training_output_data ):
        for iteration in range(1000):
            cost=0
            nabla_weight_1 = np.array([[0,0],[0,0],[0,0],[0,0]]) 
            nabla_weight_2 = np.array([0,0,0,0])
            for i in range (4):
                cost=cost+(self.operation(training_input_data[i].transpose())-training_output_data[i])**2
                for j in range (4):
                    nabla_weight_2.transpose()[j]=np.add(nabla_weight_2.transpose()[j],
                                      np.array([(self.operation(training_input_data[i].transpose())-training_output_data[i])*
                                      sigmoid(self.operation(training_input_data[i].transpose()),True)*
                                      sigmoid(np.dot(self.weight_1,training_input_data[i].transpose())[j])
                                      ]))
                    
                    nabla_weight_1[j][0]=np.add(nabla_weight_1[j][0],
                                      np.array([(self.operation(training_input_data[i].transpose())-training_output_data[i])*
                                      sigmoid(self.operation(training_input_data[i].transpose()),True)*
                                      self.weight_2.transpose()[j]*
                                      training_input_data[i].transpose()[0]
                                      ]))
                    
                    nabla_weight_1[j][1]=np.add(nabla_weight_1[j][1],
                                      np.array([(self.operation(training_input_data[i].transpose())-training_output_data[i])*
                                      sigmoid(self.operation(training_input_data[i].transpose()),True)*
                                      self.weight_2.transpose()[j]*
                                      training_input_data[i].transpose()[1]
                                      ]))
                    
                    
                        
            cost=cost/125
            # updating weights
            self.weight_1=np.add(self.weight_1,cost*nabla_weight_1)
            self.weight_2=np.add(self.weight_2,cost*nabla_weight_2)
        

        
if __name__ == "__main__":
    nn=NN101()
    print(nn.weight_1)
    print(nn.weight_2)
    print("--------------")
    nn.train_and_backpropogate(training_input_data, training_output_data)
    print(nn.weight_1)
    print(nn.weight_2)
    print("--------------")
    # testing
    for m in range (4):
        print(nn.operation(training_input_data[m].transpose()))
