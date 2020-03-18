import numpy as np
training_input_data=np.array([[0,1],[1,0],[0,0],[1,1]])
training_output_data=np.array([[0],[0],[0],[1]])
# recreating AND
def sigmoid (x, deriv=False):
    if deriv:
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))
sigmoid_vector = np.vectorize(sigmoid)




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
        return sigmoid_vector(np.dot(self.weight_2, sigmoid_vector(np.dot(self.weight_1, feed))))[0][0]
    
    def train_and_backpropogate (self, training_input_data, training_output_data ):
        for iteration in range(1000):
            cost=0
            nabla_weight_1 = np.array([[0,0],[0,0],[0,0],[0,0]]) 
            nabla_weight_2 = np.array([0,0,0,0])
            for i in range (4):
                cost=cost+(self.operation((np.array([training_input_data[i]])).T)-training_output_data[i][0])**2
                for j in range (4):
                    # PROBLEM !!!!!
                    nabla_weight_2[0][j]+=(self.operation((np.array([training_input_data[i]])).T)-training_output_data[i][0])*\
                    sigmoid(self.operation((np.array([training_input_data[i]])).T),True)*\
                    sigmoid((np.dot(self.weight_1,((np.array([training_input_data[i]])).T)))[j][0])
                                      
                    nabla_weight_1[j][0]+=(self.operation((np.array([training_input_data[i]])).T)-training_output_data[i][0])*\
                    sigmoid(self.operation((np.array([training_input_data[i]])).T),True)*\
                    self.weight_2[0][j]*\
                    training_input_data[i][0]
                    
                    nabla_weight_1[j][1]+=(self.operation((np.array([training_input_data[i]])).T)-training_output_data[i][0])*\
                    sigmoid(self.operation((np.array([training_input_data[i]])).T),True)*\
                    self.weight_2[0][j]*\
                    training_input_data[i][1]
  
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
        print(nn.operation(np.array([training_input_data[m]]).T))
