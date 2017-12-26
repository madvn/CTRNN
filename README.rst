CTRNN
=========================
Python package that implements Continuous Time Recurrent Neural Networks (CTRNNs)

See Beer, R.D. (1995). On the dynamics of small continuous-time recurrent neural networks. Adaptive Behavior 3:469-509. for a study of CTRNNs. 

Installation instructions
-------------------------
        $ pip install CTRNN



Usage
-----

The CTRNN class has the following functions::

         |  __init__(self, size=2, step_size=0.1)
         |      Constructer that initializes a random network
         |      with unit time-constants and biases
         |      args = size:integer = network size
         |
         |  euler_step(self, external_inputs)
         |      Euler stepping the network by self.step_size with provided inputs
         |      args = external_inputs:array[size,] = one float input per neuron
         |
         |  inverse_sigmoid(self, o)
         |      Computes the inverse of the sigmoid function
         |      args = o:array of any size
         |      returns = inverse_sigmoid(o):array same size as o
         |
         |  randomize_outputs(self, lb, ub)
         |      Randomize outputs in range [lb,ub]
         |      args = lb:float = lower bound for random range
         |              ub:float = upper bound for random range
         |
         |  randomize_states(self, lb, ub)
         |      Randomize states in range [lb,ub]
         |      args = lb:float = lower bound for random range
         |              ub:float = upper bound for random range
         |
         |  sigmoid(self, s)
         |      Computes the sigmoid function on input array
         |      args = s:array of any Size
         |      output = sigmoid(s):array of same size as input

Example
-------

The following code creates a 2-neuron CTRNN sinusoidal oscillator::

        # imports
        import numpy as np
        import matplotlib.pyplot as plt
        # importing the CTRNN class
        from CTRNN import *

        # params
        run_duration = 250
        net_size = 2
        step_size = 0.01

        # set up network
        network = CTRNN(size=net_size,step_size=step_size)
        network.taus = [1.,1.]
        network.biases = [-2.75,-1.75]
        network.weights[0,0] = 4.5
        network.weights[0,1] = 1
        network.weights[1,0] = -1
        network.weights[1,1] = 4.5

        # initialize network
        network.randomize_outputs(0.1,0.2)

        # simulate network
        outputs = []
        for _ in range(int(run_duration/step_size)):
            network.euler_step([0]*net_size) # zero external_inputs
            outputs.append([network.outputs[0],network.outputs[1]])
        outputs = np.asarray(outputs)

        # plot oscillator output
        plt.plot(np.arange(0,run_duration,step_size),outputs[:,0])
        plt.plot(np.arange(0,run_duration,step_size),outputs[:,1])
        plt.xlabel('Time')
        plt.ylabel('Neuron outputs')
        plt.show()

Output

.. image:: https://raw.githubusercontent.com/madvn/CTRNN/master/osc.png
