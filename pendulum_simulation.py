


import tkinter as tk
import numpy as np
import math
import os
from pendulum_window import myWindow
import time

'''
# Simulation

## Physical model
link: mass m, inertia i, length R, gravity g
angle: theta. When pendulum is at the bottom, theta=0. Positive direction is counter-clockwise.
friction: fFric=-cf*dtheta
noise: fNoise=Gaussian(mean=0, std)

## Dynamics
configuration variable: q=theta
Lag = 1/2*m*(R/2*q'[t])^2 + 1/2*i*(q'[t])^2 - m*g*(1 - Cos[q[t]])
Eular-Lagrange equations: D[D[Lag, q'[t]], t] - D[Lag, q[t]] == -cf*q'[t] + fNoise
by solving E-L eqs, we get q''[t]

## Update Law: Eular-Integration
ddq = compute_ddq()
dq[k+1]=dq[k]+ddq*dt
q[k+1]=q[k]+dq*dt
'''

class Pendulum(object):
    def __init__(self, dt):
        self.m=1.0 # mass
        self.R=1.0 # length
        self.i=1.0/3*self.m*self.R**2 # inertia 

        self.g=9.8
        self.cf=0.01 # coef of friction
        # self.friction = lambda cf, dq: -cf*dq
        self.fNoise=0.001
        self.q=0.0
        self.dq=0.0
        self.ddq=0.0

        # simulation params
        self.dt=0.01
        self.t0=time.time()
        self.t_=0.0

        # pre-compute some qualities
        self.I=4*self.i + self.m*self.R**2
        self.gmR=self.g*self.m*self.R

        return

    def compute_ddq(self):
        fNoise=self.fNoise*(np.random.random()*2-2)
        self.ddq = 4*(fNoise - self.gmR/2*math.cos(self.q) - self.cf*self.dq)/self.I
        return self.ddq

    def update_q_dq(self):
        self.dq=self.dq+self.ddq*self.dt
        self.q=self.q+self.dq*self.dt
        return self.q, self.dq

    def reset_time(self):
        self.t0=time.time()

    def get_time(self):
        return time.time()-self.t0

INTERVAL_PRINT=1.0 # print for every ?? seconds

def update():
    pendulum.reset_time()

    MAX_ITE=int(1000.0/dt_sim)
    ite=0
    while ite<MAX_ITE:
        ite+=1
        
        if ite%(dt_disp/dt_sim)==0:
            mw.canvas.update()

        if ite%(INTERVAL_PRINT/dt_sim)==0:
            print("t=%.2f, ddq=%.2f" %
                (pendulum.get_time(), pendulum.ddq))

        q_pre,dq_pre=pendulum.q,pendulum.dq
        pendulum.compute_ddq()
        pendulum.update_q_dq()
        q_cur,dq_curr=pendulum.q,pendulum.dq

        delta_theta=q_cur-q_pre
        output_for_display(delta_theta)

        # sleep
        t_sleep=ite*dt_sim-pendulum.get_time()
        if t_sleep>0:
            time.sleep(t_sleep)

def output_for_display(delta_theta):
    # the window's coordinate: x to right, y to down.
    # which is different from world.
    # So, I need to change coordinate
    delta_theta = - delta_theta

    mw.rotate_link1(delta_theta)

if __name__=="__main__":
    # b = tk.Button(window, text='move', command=moveit).pack()
    os.system('xset r on')

    dt_sim=0.01
    dt_disp=0.05
    pendulum=Pendulum(dt_sim)
    mw=myWindow()
    
    mw.window.after(10, update)
    tk.mainloop()