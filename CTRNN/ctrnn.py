import numpy as np
from scipy.sparse import csr_matrix

class CTRNN:

    def __init__(self,size=2,step_size=0.1):
        '''
        Constructer that initializes a random network
        with unit time-constants and biases
        ARGS:
        size: integer = network size
        step_size:float = euler integration step size
        '''
        self.size = size
        self.step_size = step_size
        self.taus = np.ones(size)
        self.biases = np.ones(size)
        self.gains = np.ones(size)
        self.weights = csr_matrix(np.random.rand(size,size))
        self.states = np.random.rand(size)
        self.outputs = self.sigmoid(self.states)

    @property
    def taus(self): return self.__taus

    @property
    def biases(self): return self.__biases

    @property
    def gains(self): return self.__gains

    @property
    def states(self): return self.__states

    @property
    def outputs(self): return self.__outputs

    @taus.setter
    def taus(self,ts):
        '''
        Set time-constants
        args = ts:array[size,] = time-constant for each neuron
        '''
        if len(ts) != self.size:
            raise Exception("Size mismatch error - len(taus) != network_size")
        self.__taus = np.asarray(ts)

    @biases.setter
    def biases(self,bis):
        '''
        Set biases
        args = bis:array[size,] = bias for each neuron
        '''
        if len(bis) != self.size:
            raise Exception("Size mismatch - len(biases) != network_size")
        self.__biases = np.asarray(bis)

    @gains.setter
    def gains(self,gs):
        '''
        Set gains
        args = gs:array[size,] = gain for each neuron
        '''
        if len(gs) != self.size:
            raise Exception("Size mismatch - len(gains) != network_size")
        self.__gains = np.asarray(gs)

    @states.setter
    def states(self,s):
        '''
        Set states
        args = s:array[size,] = state for each neuron
        '''
        if len(s) != self.size:
            raise Exception("Size mismatch - len(states) != network_size")
        self.__states = np.asarray(s)
        self.__outputs = self.sigmoid(s)

    @outputs.setter
    def outputs(self,o):
        '''
        Set outputs
        args = o:array[size,] = output for each neuron
        '''
        if len(o) != self.size:
            raise Exception("Size mismatch - len(outputs) != network_size")
        self.__outputs = np.asarray(o)
        self.__states = self.inverse_sigmoid(o)/self.gains - self.biases

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
        self.states += self.step_size*(1/self.taus)* (total_inputs - self.states)
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
        inverse_sig = np.log(o/(1-o))
        #inverse_sig[np.isinf(inverse_sig)] = 0.
        return inverse_sig

#CTRNN = CTRNN()
if __name__ == "__main__":
    print('CTRNN python package.')
