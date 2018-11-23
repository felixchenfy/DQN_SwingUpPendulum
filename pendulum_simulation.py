
import tkinter as tk
import numpy as np
import math
import os
from pendulum_window import myWindow
import time
from Lunge_Kutta import rK3  # Lunge-Kutta Integration method

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


class Pendulum(myWindow):
    def __init__(self, dt_sim):
        super(Pendulum, self).__init__()

        self.m = 1.0  # mass
        self.R = 1.0  # length
        self.i = 1.0/3*self.m*self.R**2  # inertia

        self.g = 9.8
        self.cf = 0.1  # coef of friction, where friction = - cf * dq
        self.fNoise = 0.001

        self.q = 0.0
        self.dq = 0.0
        self.ddq = 0.0

        # simulation params
        self.dt_sim = dt_sim
        self.t0 = time.time()
        self.t_ = 0.0

        # user input
        self.torque = 0
        # self.torque_effect_time=0.1

        # pre-compute some qualities
        self.I = 4*self.i + self.m*self.R**2
        self.gmR = self.g*self.m*self.R

    def update_q_from_window(self):
        self.q = -self.link1_theta

    def assign_q_to_window(self):
        self.link1_theta = -self.q

    def compute_ddq(self, q, dq):
        fNoise = self.fNoise*(np.random.random()*2-2)
        F = fNoise+self.torque-self.cf*dq # total external torque applied on joint 1
        ddq = 4*(F - self.gmR/2*math.cos(q))/self.I
        return ddq

    def update_states(self): # integrate ddq to get q and dq
        self.update_q_and_dq_()

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

        self.q = q
        self.dq = dq

    def reset_time(self):
        self.t0 = time.time()

    def get_time(self):
        return time.time()-self.t0

    def set_torque(self, torque):
        self.torque=torque

    def event_KeyPress(self, e):  # overload
        c = e.char.lower()

        # rotate link
        dtheta = 0.1
        if c == "a":
            self.rotate_link1(-dtheta)
        elif c == "d":
            self.rotate_link1(dtheta)

        # apply torque
        torque_maginitude=1
        if c == "j":
            self.set_torque(-torque_maginitude)
        elif c == "k":
            self.set_torque(torque_maginitude)

        # restart
        elif c == 'q':
            self.restart()
        return

    def event_KeyRelease(self, e):
        c = e.char.lower()
        # reset torque
        if c == "j":
            self.set_torque(0)
        elif c == "k":
            self.set_torque(0)
        return

    def display_text(self): # overload
        str_text = "User input:\n"
        # str_text += "A,D: rotate link1\n"
        # str_text += "W,S: rotate link2\n"
        str_text += "J,K: apply torque to link1\n"
        str_text += "Q: reset game\n"
        # str_text+="E/R: enable/disable simulation\n" # the sim engine is in "pendulum_env"
        self.display_text_(str_text)

PRINT_INTERVAL = 1.0  # print for every PRINT_INTERVAL seconds


def update():
    pendulum.reset_time()

    MAX_ITE = int(1000.0/dt_sim)
    ite = 0
    while ite < MAX_ITE:
        ite += 1

        # update window
        if ite % (dt_disp/dt_sim) == 0:
            pendulum.canvas.update()

        # print time and q
        if ite % (PRINT_INTERVAL/dt_sim) == 0:
            print("t=%.2f, ddq=%.2f" %
                  (pendulum.get_time(), pendulum.ddq))

        # read current q from window
        # (because user might reset the link's position)
        pendulum.update_q_from_window()

        # do simulation
        pendulum.update_states()

        # output to the window
        pendulum.assign_q_to_window()
        pendulum.rotate_link1_to(pendulum.link1_theta)

        # sleep
        t_sleep = ite*dt_sim-pendulum.get_time()
        if t_sleep > 0:
            time.sleep(t_sleep)


if __name__ == "__main__":
    # b = tk.Button(window, text='move', command=moveit).pack()
    try:
        os.system('xset r off')
    except:
        None

    dt_sim = 0.005
    dt_disp = 0.05

    pendulum = Pendulum(dt_sim)
    pendulum.window.after(10, update)
    pendulum.mainloop()
    
    try:
        os.system('xset r on')
    except:
        None


