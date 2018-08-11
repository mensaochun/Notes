import numpy as np
import matplotlib.pyplot as plt
import pylab
# Create a probability distribution

sum_x=np.zeros(100000)
for i in range(100000):
    x=np.random.rand(1000)
    sum_x[i]=np.mean(x)
pylab.hist(sum_x,200)