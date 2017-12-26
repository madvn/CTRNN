from CTRNN import CTRNN

print("************************************** Basic test of CTRNN.py **************************************")
results = []

print("Initing network of size 3...")
size = 3
ns = CTRNN(size)
print("Done!")

print("\n\n**Testing taus and biases")
print("Before",ns.taus,ns.biases)
try:
    ns.taus = np.random.rand(size)
    results.append(True)
except:
    results.append(False)
    pass
try:
    ns.biases = np.random.rand(size)
    results.append(True)
except:
    results.append(False)
    pass
print("After",ns.taus,ns.biases)

try:
    ns.taus = np.random.rand(size-1)
    results.append(False)
except Exception as e:
    print("Raised exception for ns.taus = np.random.rand(size-1)")
    print(e)
    results.append(True)

try:
    ns.biases = np.random.rand(size-1)
    results.append(False)
except Exception as e:
    print("Raised exception for ns.biases = np.random.rand(size-1)")
    print(e)
    results.append(True)

print("\n\n**Testing weights")
#ns.weights = np.random.rand([size,size])
print("Before ",ns.weights[1,2],type(ns.weights))
try:
    ns.weights[1,2] = 3.45
    results.append(True)
except:
    results.append(False)
    pass
print("After ",ns.weights[1,2],type(ns.weights))
try:
    ns.weights[1,3] = 3.45
    results.append(False)
except Exception as e:
    print("Raised exception for ns.weights[1,3] = 3.45")
    print(e)
    results.append(True)

print("\n\n**Testing states")
print("Before ",ns.states,ns.outputs)
try:
    ns.states = np.ones(size)*0.5
    results.append(True)
except:
    results.append(False)
    pass
print("After ",ns.states,ns.outputs)
try:
    ns.states = [5,3]
    results.append(False)
except Exception as e:
    print("Raised exception with ns.states = [5,3]")
    print(e)
    results.append(True)

print("Before ",ns.states,ns.outputs)
try:
    ns.randomize_states(0.5,0.6)
    results.append(True)
except:
    results.append(False)
    pass
print("After ",ns.states,ns.outputs)

print("\n\n**Testing outputs")
print("Before ",ns.states,ns.outputs)
try:
    ns.outputs = np.ones(size)*0.5
    results.append(True)
except:
    results.append(False)
    pass
print("After ",ns.states,ns.outputs)
try:
    ns.outputs = [5,3]
    results.append(False)
except Exception as e:
    print("Raised exception with ns.outputs = [5,3]")
    print(e)
    results.append(True)

print("Before ",ns.states,ns.outputs)
try:
    ns.randomize_outputs(0.5,0.6)
    results.append(True)
except:
    results.append(False)
    pass
print("After ",ns.states,ns.outputs)

print("\n\n**Stepping network for 50 time steps")
print("Before ",ns.states,ns.outputs)
try:
    for _ in range(50):
        ns.euler_step(np.random.rand(size))
    results.append(True)
except:
    results.append(False)

print("After ",ns.states,ns.outputs)

print("\n\nTest can be considered passed if 5 exceptions were raised and there were no Nans or other errors")
if any(results) == False: print("Test result: FAIL")
else: print("Test result: PASS")
print("Contact madcanda@indiana.edu if test fails or other bugs are discovered")
print("************************************** End of test! **************************************")
