import numpy as np
from CTRNN import CTRNN

test_net_size = 3

def show_exception(message,e):
    print(message)
    print(e)

def test_initialization():
    '''
    Inits a CTRNN and returns True if succesful
    '''
    try:
        print("Initing network of size 3...")
        test_net_size = 3
        ns = CTRNN(test_net_size)
        print("Done!")
        return True
    except Exception as e:
        show_exception("Raised exception with ns = CTRNN(test_net_size)",e)
        return False

def test_modification():
    '''
    Tests different kinds of modification and returns any(results)
    '''
    results = []
    # init first
    test_net_size = 3
    ns = CTRNN(test_net_size)

    print("**Testing taus and biases")
    print("Before",ns.taus,ns.biases)
    try:
        ns.taus = np.random.rand(test_net_size)
        results.append(True)
    except Exception as e:
        show_exception("Raised exception with ns.taus = np.random.rand(test_net_size)",e)
        results.append(False)
        pass
    try:
        ns.biases = np.random.rand(test_net_size)
        results.append(True)
    except Exception as e:
        show_exception("Raised exception with ns.biases = np.random.rand(test_net_size)",e)
        results.append(False)
        pass
    print("After",ns.taus,ns.biases)

    try:
        ns.taus = np.random.rand(test_net_size-1)
        results.append(False)
    except Exception as e:
        show_exception("Raised exception for ns.taus = np.random.rand(test_net_size-1)",e)
        results.append(True)

    try:
        ns.biases = np.random.rand(test_net_size-1)
        results.append(False)
    except Exception as e:
        show_exception("Raised exception for ns.biases = np.random.rand(test_net_size-1)",e)
        results.append(True)

    print("**Testing weights")
    #ns.weights = np.random.rand([test_net_size,test_net_size])
    print("Before ",ns.weights[1,2],type(ns.weights))
    try:
        ns.weights[test_net_size-1,test_net_size-1] = 3.45
        results.append(True)
    except Exception as e:
        show_exception("Raised exception with ns.weights[test_net_size-1,test_net_size-1] = 3.45",e)
        results.append(False)
        pass
    print("After ",ns.weights[1,2],type(ns.weights))
    try:
        ns.weights[test_net_size+1,test_net_size-1] = 3.45
        results.append(False)
    except Exception as e:
        show_exception("Raised exception for ns.weights[test_net_size+1,test_net_size-1] = 3.45",e)
        results.append(True)

    print("**Testing states")
    print("Before ",ns.states,ns.outputs)
    try:
        ns.states = np.ones(test_net_size)*0.5
        results.append(True)
    except Exception as e:
        show_exception("Raised exception with ns.states = np.ones(test_net_size)*0.5",e)
        results.append(False)
        pass
    print("After ",ns.states,ns.outputs)
    try:
        ns.states = np.random.rand(test_net_size-1)
        results.append(False)
    except Exception as e:
        show_exception("Raised exception with ns.states = np.random.rand(test_net_size-1)",e)
        results.append(True)

    print("Before ",ns.states,ns.outputs)
    try:
        ns.randomize_states(0.5,0.6)
        results.append(True)
    except Exception as e:
        show_exception("Raised exception with ns.randomize_states(0.5,0.6)",e)
        results.append(False)
        pass
    print("After ",ns.states,ns.outputs)

    print("**Testing outputs")
    print("Before ",ns.states,ns.outputs)
    try:
        ns.outputs = np.ones(test_net_size)*0.5
        results.append(True)
    except Exception as e:
        show_exception("Raised exception with ns.outputs = np.ones(test_net_size)*0.5",e)
        results.append(False)
        pass
    print("After ",ns.states,ns.outputs)
    try:
        ns.outputs = np.random.rand(test_net_size-1)
        results.append(False)
    except Exception as e:
        show_exception("Raised exception with ns.outputs = np.random.rand(test_net_size-1)",e)
        results.append(True)

    print("Before ",ns.states,ns.outputs)
    try:
        ns.randomize_outputs(0.5,0.6)
        results.append(True)
    except Exception as e:
        show_exception("Raised exception with ns.randomize_outputs(0.5,0.6)",e)
        results.append(False)
        pass
    print("After ",ns.states,ns.outputs)

    return any(results)

def test_simulation():
    '''
    Euler steps a CTRNN and returns True if no exceptions were raised
    '''
    ns = CTRNN(test_net_size)
    print("**Stepping network for 50 time steps")
    print("Before ",ns.states,ns.outputs)
    try:
        for _ in range(50):
            ns.euler_step(np.random.rand(test_net_size))
        result = True
    except Exception as e:
        show_exception("Simulation error:",e)
        result = False

    print("After ",ns.states,ns.outputs,result)
    return result

def run_basic_test():
    '''
    Main test function - runs each sub-test
    '''
    print("************************************** Basic test of CTRNN.py **************************************")
    print("Contact madcanda@indiana.edu if test fails or other bugs are discovered")

    print("\n## Testing Initialization")
    assert test_initialization(),"CTRNN Initialization Test Failed"
    print("**Initialization Test Passed**")

    print("\n## Testing Modification")
    assert test_modification(),"CTRNN Modification Test Failed"
    print("**CTRNN Modification Test Passed**")

    print("\n## Testing Simulation")
    assert test_simulation(),"CTRNN Simulation Test Failed"
    print("**CTRNN Simulation Test Passed**")

    print("\nALL TESTS PASSED!!")

if __name__ == "__main__":
    run_basic_test()
