
import numpy as np
import matplotlib.pyplot as plt

import sys, os
CURRENT_PATH=os.path.join(os.path.dirname(__file__))

def plot_cost(cost_his):
    store_interval_time=5.0
    plt.plot(np.arange(len(cost_his))*store_interval_time, cost_his)
    plt.ylabel('Cost')
    plt.xlabel('Time (second, in simulation)')
    plt.yscale("linear")

strtitle=[
    "Inverted pendulum. Initial pose is upright.",
    "Inverted pendulum. Initial pose is random.",
    "Swing-up pendulum. Initial pose is random.",
    "Swing-up pendulum. Initial pose is hanging-down.",
]

stage_idx=3
filename="stage"+str(stage_idx)+"_cost_history"
cost_history=np.loadtxt(CURRENT_PATH+"/"+filename+".txt", delimiter=" ")

plot_cost(cost_history)
plt.title(filename+"\n"+strtitle[stage_idx-1])
plt.savefig(CURRENT_PATH+"/"+filename+".jpg")
plt.show()
