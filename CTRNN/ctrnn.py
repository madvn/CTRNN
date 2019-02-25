import numpy as np
from scipy.sparse import csr_matrix

class CTRNN:

    def __init__(self,size=2,step_size=0.1):
        '''
        Constructer that initializes a random network
        with unit time-constants and biases
        args = size:integer = network size
               step_size = Euler integration step size
        '''
        self.size = size
        self.step_size = step_size
        self.taus = np.ones(size)
        self.biases = np.ones(size)
        self.gains = np.ones(size)
        self.weights = csr_matrix(np.random.rand(size,size))
        self.states = np.random.rand(size)
        self.outputs = self.sigmoid(self.states)

    def randomize_states(self,lb,ub):
        '''
        Randomize states in range [lb,ub]
        args = lb:float = lower bound for random range
                ub:float = upper bound for random range
        '''
        self.states = np.random.uniform(lb,ub,size=(self.size))

    def randomize_outputs(self,lb,ub):
        '''
        Randomize outputs in range [lb,ub]
        args = lb:float = lower bound for random range
                ub:float = upper bound for random range
        '''
        self.outputs = np.random.uniform(lb,ub,size=(self.size))

    def euler_step(self,external_inputs):
        '''
        Euler stepping the network by self.step_size with provided inputs
        args = external_inputs:array[size,] = one float input per neuron
        '''
        if len(external_inputs) != self.size:
            raise Exception("Size mismatch - len(external_inputs) != network_size")
        external_inputs = np.asarray(external_inputs)
        total_inputs = external_inputs + self.weights.dot(self.outputs)
        self.states += self.step_size*self.taus* [total_inputs - self.states][0]
        self.outputs = self.sigmoid(self.gains*(self.states+self.biases))

    def sigmoid(self,s):
        '''
        Computes the sigmoid function on input array
        args = s:array of any Size
        output = sigmoid(s):array of same size as input
        '''
        return 1/(1+np.exp(-s))

    def inverse_sigmoid(self,o):
        '''
        Computes the inverse of the sigmoid function
        args = o:array of any size
        returns = inverse_sigmoid(o):array same size as o
        '''
        return np.log(o/(1-o))

