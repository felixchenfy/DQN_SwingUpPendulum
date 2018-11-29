#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This is mainly copied from here:
# https://www.codeproject.com/Tips/792927/Fourth-Order-Runge-Kutta-Method-in-Python


import math

def rKN(x, fx, n, hs):
   k1 = []
   k2 = []
   k3 = []
   k4 = []
   xk = []
   for i in range(n):
       k1.append(fx[i](x)*hs)
   for i in range(n):
       xk.append(x[i] + k1[i]*0.5)
   for i in range(n):
       k2.append(fx[i](xk)*hs)
   for i in range(n):
       xk[i] = x[i] + k2[i]*0.5
   for i in range(n):
       k3.append(fx[i](xk)*hs)
   for i in range(n):
       xk[i] = x[i] + k3[i]
   for i in range(n):
       k4.append(fx[i](xk)*hs)
   for i in range(n):
       x[i] = x[i] + (k1[i] + 2*(k2[i] + k3[i]) + k4[i])/6
   return x


def rK3(a, b, c, fa, fb, fc, hs):
    a1 = fa(a, b, c)*hs
    b1 = fb(a, b, c)*hs
    c1 = fc(a, b, c)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(ak, bk, ck)*hs
    b2 = fb(ak, bk, ck)*hs
    c2 = fc(ak, bk, ck)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(ak, bk, ck)*hs
    b3 = fb(ak, bk, ck)*hs
    c3 = fc(ak, bk, ck)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(ak, bk, ck)*hs
    b4 = fb(ak, bk, ck)*hs
    c4 = fc(ak, bk, ck)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b, c


def test_3_dimension():

    # y¨=μ(1−y^2)y˙−y+Asin(ωt)
    # y¨=0.9(1−y^2)y˙−y+1*sin(0.5*t)

    # Let:x  =( dy,  y, wt)
    #  dx/dt =(ddy, dy,  w)

    # a: dy
    # b: y
    # c: t
    # fa: d(dy)/dt = equation
    # fb: d(y)/dt = a
    # fa: d(wt)/dt = w

    def fa2(a, b, c):
        return 0.9*(1 - b*b)*a - b + math.sin(c)

    def fb2(a, b, c):
        return a

    def fc2(a, b, c):
        return 0.5 # d(wt)/dt=w

    def VDP2():
        a, b, c, hs = 1, 1, 0, 0.05
        for i in range(20000):
            a, b, c = rK3(a, b, c, fa2, fb2, fc2, hs)
        return a,b,c

    a,b,c=VDP2()
    print(a,b,c)

def test_N_dimension():
    def fa1(x):
        return 0.9*(1 - x[1]*x[1])*x[0] - x[1] + math.sin(x[2])

    def fb1(x):
        return x[0]

    def fc1(x):
        return 0.5

    def VDP1():
        f = [fa1, fb1, fc1]
        x = [1, 1, 0]
        hs = 0.05
        for i in range(20000):
            x = rKN(x, f, 3, hs)
        return x

    x=VDP1()
    print(x)

if __name__=="__main__":

    test_3_dimension()
    # test_N_dimension()