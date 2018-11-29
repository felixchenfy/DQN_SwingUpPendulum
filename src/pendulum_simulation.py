#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This scripts adds the physical model (dynamics) to the 1-link pendulum
# You can use keypress "j" and "k" to apply torque to control the pendulum.

# Note:
# In pendulum_window.py, configuration is called:
#       "self.link1_theta"
# However, here the parameter is called:
#       "self.q", "self.dq".
# I put it in this way to seperate the "dynamics" and "image display",
# because there units might be different (e.g., 1 pixel versus 1 meter).

import tkinter as tk
import numpy as np
import math
import time
from math import pi

import sys, os
PROJECT_PATH=os.path.join(os.path.dirname(__file__))+ "/../"
sys.path.append(PROJECT_PATH)

from lib.Lunge_Kutta import rK3  # Lunge-Kutta Integration method
from pendulum_window import myWindow


'''
# Description of Simulation 

## Physical model
link: mass m, inertia i, length R, gravity g
angle: theta. When pendulum is at the bottom, theta=0. Positive direction is counter-clockwise.
friction: fFric=-cf*dtheta
noise: fNoise=Gaussian(mean=0, std)

## Dynamics
configuration variable: q=theta
LLagrange equation: 1/2*m*(R/2*q'[t])^2 + 1/2*i*(q'[t])^2 - m*g*(1 - Cos[q[t]])
Eular-Lagrange equations: D[D[Lag, q'[t]], t] - D[Lag, q[t]] == -cf*q'[t] + fNoise
by solving E-L eqs, we get q''[t]

## Update Law:
1. Eular-Integration 
    ddq = compute_ddq()
    dq[k+1]=dq[k]+ddq*dt
    q[k+1]=q[k]+dq*dt
2. Lunge Kutta

'''

# Choose a scenario
Inverted_Pendulum=True
Swing_Up_Pendulum=not Inverted_Pendulum

# Main parameters
Friction_Coefficient=0.1
Torque_Maginitude=2
Force_Noise=0.01

# trans angle to [0, 2*pi]
def angle_trans_to_pi(theta):
    return theta % (2*pi)

# main class
class Pendulum(myWindow):
    def __init__(self, q1_init=-math.pi/2, dq1_init=0, dt_sim=0.001):

        self.dt_sim = dt_sim # simulation time period
        self.q1_init=q1_init # initial angle of link1
        self.dq1_init=dq1_init # initial angular velocity of link1

        super(Pendulum, self).__init__()

        try: # turn off continuous key input
            os.system('xset r off')
        except:
            None
    
    # reset link1 to (q1_init, dq1_init)
    def reset(self, q1_init=None, dq1_init=None):

        # first reset all
        super(Pendulum, self).reset() # call parent function to reset the canvas
        self.reset_vars() # reset the dynamics parameters

        # then, deal with the case if user want to reset pendulum to a specified position
        if q1_init is not None:
            self.q=q1_init
            self.dq=dq1_init
            self.assign_q_to_window()
            self.rotate_link1_to(self.link1_theta)

        # return the states after reset
        return self.get_states() # [q, dq]

    def get_states(self):
        q=self.q
        dq=self.dq
        return np.array([q, dq])

    def reset_vars(self):
        super(Pendulum, self).reset()

        self.m = 1.0  # mass
        self.R = 1.0  # length
        self.i = 1.0/3*self.m*self.R**2  # inertia

        self.g = 9.8
        self.cf = Friction_Coefficient  # coef of friction, where friction = - cf * dq
        self.fNoise = Force_Noise

        self.q = 0.0
        self.dq = 0.0
        self.ddq = 0.0

        # simulation params
        self.t_sim=0.0
        # self.t_real0 = time.time() # This should be set manully through self.reset_real_time()

        # user input
        self.torque = 0
        self.Torque_Maginitude=Torque_Maginitude # self.torque = 0, +, - Torque_Maginitude

        # pre-compute some qualities
        self.I = 4*self.i + self.m*self.R**2
        self.gmR = self.g*self.m*self.R
        
        # others
        self.reset_real_time()

    def update_q_from_window(self):
        self.q = -self.link1_theta

    def assign_q_to_window(self):
        self.link1_theta = -self.q

    def compute_ddq(self, q, dq):
        fNoise = self.fNoise*(np.random.random()*2-2)
        F = fNoise+self.torque-self.cf*dq # total external torque applied on joint 1
        ddq = 4*(F - self.gmR/2*math.cos(q))/self.I
        return ddq

    def update_states(self, sim_time_length): # integrate ddq to get q and dq
        n=int(sim_time_length/self.dt_sim)
        for i in range(n):
            self.update_q_and_dq_()
            self.t_sim+=self.dt_sim

    def update_q_and_dq_(self):
        q = self.q
        dq = self.dq
        ddq = self.compute_ddq(q, dq)
        dt = self.dt_sim

        if 0:  # 1st order: Euler-Integration
            q += dq*dt
            dq += ddq*dt

        else:  # 4th order: Runge-Kutta method
            if not hasattr(self, 'fa_'):
                # vec=(a,b,c)=(dq,q,t)
                # f(a,b,c)=(da,db,dc)=(ddq,dq,1)
                # fa: d(dq)/dt
                # fb: d(q)/dt
                # fa: d(t)/dt
                self.fa_ = lambda a, b, c: self.compute_ddq(q=b, dq=a)
                self.fb_ = lambda a, b, c: a
                self.fc_ = lambda a, b, c: 1
            a = dq
            b = q
            c = 0  # here ddq() is not depend on t, so let t=0
            dq, q, _ = rK3(a, b, c, self.fa_, self.fb_, self.fc_, dt)

        self.q = angle_trans_to_pi(q)
        self.dq = dq

    def reset_real_time(self):
        self.t_real0 = time.time()

    def get_real_time(self):
        return time.time()-self.t_real0
    def get_sim_time(self):
        return self.t_sim

    def set_torque(self, torque):
        self.torque=torque*self.Torque_Maginitude

    def event_KeyPress(self, e):  # overload
        c = e.char.lower()

        # rotate link
        dtheta = 0.1
        if c == "a":
            self.rotate_link1(-dtheta)
        elif c == "d":
            self.rotate_link1(dtheta)

        # apply torque
        if c == "j":
            self.set_torque(-1) # Clockwise
            self.canvas.itemconfig(self.light1, fill='red')
        elif c == "k":
            self.set_torque(1) # CounterClockwise
            self.canvas.itemconfig(self.light2, fill='red')

        # restart
        elif c == 'q':
            self.reset()
        return

    def event_KeyRelease(self, e):
        c = e.char.lower()
        # reset torque
        if c == "j":
            self.set_torque(0)
            self.canvas.itemconfig(self.light1, fill='black')
        elif c == "k":
            self.set_torque(0)
            self.canvas.itemconfig(self.light2, fill='black')
        return

    def display_text(self): # overload
        str_text = "User input:\n"
        # str_text += "A,D: rotate link1\n"
        # str_text += "W,S: rotate link2\n"
        str_text += "J,K: apply torque to link1\n"
        str_text += "Q: reset game\n"
        # str_text+="E/R: enable/disable simulation\n" # the sim engine is in "pendulum_env"
        self.display_text_(str_text)

    def run_simulation(self):
    
        PRINT_INTERVAL = 1.0  # print for every PRINT_INTERVAL seconds
        dt_disp = 0.01
        dt_sim = self.dt_sim

        self.reset_real_time()

        MAX_ITE = int(1000000.0/dt_sim)
        ite = 0
        while ite < MAX_ITE:
            ite += 1
            self.render()

            # print time and q
            if ite % (PRINT_INTERVAL/dt_disp) == 0:
                print("t={:.2f}, q={:.2f}, dq={:.2f}, ddq={:.2f}".format
                    (self.t_sim, self.q, self.dq, self.ddq))

            # read current q from window
            # (because user might reset the link's position)
            self.update_q_from_window()

            # do simulation
            self.update_states(sim_time_length=dt_disp)

            # output to the window
            self.assign_q_to_window()
            self.rotate_link1_to(self.link1_theta)

            # sleep
            t_sleep = self.t_sim-self.get_real_time()
            if t_sleep > 0:
                time.sleep(t_sleep)


if __name__ == "__main__":
    # b = tk.Button(window, text='move', command=moveit).pack()
 
    # pendulum game
    if Swing_Up_Pendulum:
        q_init=-pi/2
    elif Inverted_Pendulum:
        q_init=pi/2
    pendulum = Pendulum(q1_init=q_init, dq1_init=0, dt_sim = 0.001)

    pendulum.after(10, pendulum.run_simulation) # after 10ms, run run_simulation
    pendulum.mainloop()
    
    try:
        os.system('xset r on')
    except:
        None


