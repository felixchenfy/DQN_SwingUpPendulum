
import numpy as np
import matplotlib.pyplot as plt

import sys, os
CURRENT_PATH=os.path.join(os.path.dirname(__file__))

def plot_cost(cost_his):
    plt.plot(np.arange(len(cost_his)), cost_his)
    plt.ylabel('Cost')
    plt.show()
    
filename="costhist_2018-12-06 10:15:41.txt"
cost_history=np.loadtxt(CURRENT_PATH+"/"+filename, delimiter=" ")

plot_cost(cost_history)
