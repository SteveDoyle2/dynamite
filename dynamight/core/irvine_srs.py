"""
program: srs.py
author: Tom Irvine
Email: tom@vibrationdata.com
version: 2.3
date: October 4, 2013
description:

Calculate the shock response spectrum for an SDOF system
The file must have two columns: time(sec) & accel(G)

The numerical engine is the Smallwood ramp invariant digital
recursive filtering relationship

"""
from __future__ import print_function
#import sys

#if sys.version_info[0] == 2:
    #print ("Python 2.x")
    #import Tkinter as tk
    #from tkFileDialog import asksaveasfilename

#if sys.version_info[0] == 3:
print ("Python 3.x")
import tkinter as tk
from tkinter.filedialog import asksaveasfilename

if 0:
    from tompy import read_two_columns_from_dialog
    from tompy import signal_stats, sample_rate_check
    from tompy import GetInteger2, GetInteger3, WriteData3
    from tompy import enter_damping
    from tompy import enter_float
    from tompy import time_history_plot
    from tompy import srs_plot_pn

from scipy.signal import lfilter
import numpy as np
from numpy import zeros,concatenate
from math import pi,exp,sqrt,cos,sin

import matplotlib.pyplot as plt


########################################################################

#
#  http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html
#
#  a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[nb]*x[n-nb]
#                        - a[1]*y[n-1] - ... - a[na]*y[n-na]
#

class SRS:
    def __init__(self, b: np.ndarray, dt: float, sr: float):

        self.b = b
        self.b_old=b
        self.fn=[]
        self.num_fn = 0
        self.damp = 0
        self.dt = dt
        self.sr = sr

        self.iacc = 0
        self.ipv = 0
        self.ird = 0


    @classmethod
    def srs_parameters(cls,self):
        print(" ")

        Q = 10.
        self.damp = 1 / (2 * Q)
        f1 = 1.
        noct = 2 # 1/6th
        if 0:
            self.damp, Q = enter_damping()

            print(" ")
            print(" Enter starting frequency (Hz) ")
            f1 = enter_float()

            print(" ")
            print(" Select octave spacing")
            print("   1= one-third")
            print("   2= one-sixth")
            print("   3= one-twelfth")

            noct = GetInteger3()

        if noct == 1:
            octave = 1./3.
        elif noct == 2:
            octave = 1./6.
        elif noct == 3:
            octave = 1./12.
        else:
            raise RuntimeError(noct)

        iresidual = 0
        self.iacc = 1 # True
        self.ipv = 2  #  False
        self.ird = 2  # False
        if 0:
            print(" ")
            print("  include residual:  1=yes 2=no ")

            iresidual = GetInteger2()

            print(" ")
            print(" Select output metrics")
            print(" ")
            print("  acceleration:  1=yes 2=no ")

            self.iacc = GetInteger2()

            print(" ")
            print("  pseudo velocity:  1=yes 2=no ")
            self.ipv = GetInteger2()

            print(" ")
            print("  relative displacement:  1=yes 2=no ")
            self.ird = GetInteger2()

#*******************************************************************************
        nt = int(0.8*(1/f1)/dt)

        if iresidual == 1:
            tz = zeros(nt)
            d = concatenate((self.b, tz))
            self.b = d
#*******************************************************************************

        self.fn.append(f1)
        for j in range(1,999):
            self.fn.append(self.fn[j-1]*(2.**octave))
            if  self.fn[j] > self.sr/8.:
                break

        self.num_fn = j
        fff=self.fn

        temp = fff[0:self.num_fn]
        del fff
        self.fn=temp
        del temp

    @classmethod
    def a_coeff(cls,omega,damp,dt):
        ac = zeros(3)
        ac[0] = 1
        omegad = omega*sqrt(1.-(damp**2))
        E = exp(-damp*omega*dt)
        K = omegad*dt
        C = E*cos(K)
        ac[1] = -2*C
        ac[2] = +E**2
        return ac

    def srs_plots(self):
        SRS.srs_parameters(self)

        pv_pos = zeros(self.num_fn)
        pv_neg = zeros(self.num_fn)
        rd_pos = zeros(self.num_fn)
        rd_neg = zeros(self.num_fn)

        if self.iacc == 1:
            print(" ")
            print(" fn(Hz)  Positive Accel(G)  Negative Accel(G) ")
            x_pos, x_neg = SRS.accel_SRS(self)

        for j in range(self.num_fn):
            print ("%6.4g  %6.4g  %6.4g" % (self.fn[j],x_pos[j],x_neg[j]))


        if self.ipv==1 or self.ird==1:
            rd_pos,rd_neg=SRS.rel_disp_SRS(self)

#*******************************************************************************
        #   Pseudo Velocity
        if self.ipv==1:
            for j in range(self.num_fn):
                omega=2.*pi*self.fn[j]
                pv_pos[j] = rd_pos[j] * omega
                pv_neg[j] = rd_neg[j] * omega

#*******************************************************************************

        idf = 2
        if 0:
            print(" ")
            print("  Export data files:  1=yes 2=no ")
            idf = GetInteger2()

        if idf==1:
            print (" ")
            print (" Find output dialog box")

            if self.iacc==1:
                root = tk.Tk(); root.withdraw()
                output_file_path = asksaveasfilename(parent=root,title="Save the acceleration SRS as...")
                output_file = output_file_path.rstrip('\n')
                WriteData3(self.num_fn,self.fn,x_pos,x_neg,output_file)

            if self.ipv==1:
                root = tk.Tk(); root.withdraw()
                output_file_path = asksaveasfilename(parent=root,title="Save the pseudo velocity SRS as...")
                output_file = output_file_path.rstrip('\n')
                WriteData3(self.num_fn,self.fn,pv_pos,pv_neg,output_file)

            if self.ird==1:
                root = tk.Tk(); root.withdraw()
                output_file_path = asksaveasfilename(parent=root,title="Save the relative displacement SRS as...")
                output_file = output_file_path.rstrip('\n')
                WriteData3(self.num_fn,self.fn,rd_pos,rd_neg,output_file)

#*******************************************************************************

        plot_time = False
        if plot_time:
            print(" ")
            print(" begin plots ")
            print(" ")
            time_history_plot(
                a, self.b_old, 1,
                'Time(sec)', 'Accel(G)',
                'Base Input', 'time_history')

#*******************************************************************************

        if self.iacc == 1:
            plt.figure(2)
            srs_plot_pn(1,1,self.fn,x_pos,x_neg,self.damp,'accel_srs_plot')

        if self.ipv == 1:
            plt.figure(3)
            srs_plot_pn(2,1,self.fn,pv_pos,pv_neg,self.damp,'pv_srs_plot')

        if self.ird == 1:
            plt.figure(4)
            srs_plot_pn(3,1,self.fn,rd_pos,rd_neg,self.damp,'rd_srs_plot')

#*******************************************************************************

        plt.show()


    def accel_SRS(self):
        pos=zeros(self.num_fn)
        neg=zeros(self.num_fn)
        bc = zeros(3)

        for j in range(0,self.num_fn):
            self.fn[j] = 10.
            omega = 2.*pi*self.fn[j]
            omegad = omega*sqrt(1.-(self.damp**2))

            #  bc coefficients are applied to the excitation
            E = exp(-self.damp*omega*self.dt)
            K = omegad*self.dt
            C = E*cos(K)
            S = E*sin(K)
            Sp = S/K

            bc[0] = 1.-Sp
            bc[1] = 2.*(Sp-C)
            bc[2] = E**2-Sp

            ac = SRS.a_coeff(omega, self.damp, self.dt)
            resp = lfilter(bc, ac, self.b, axis=-1, zi=None)
            plt.plot(time, resp, label='input')
            plt.plot(time, self.b, label='output')
            plt.show()
            pos[j] = max(resp)
            neg[j] = abs(min(resp))
        return pos, neg

    def rel_disp_SRS(self):
        rd_pos=zeros(self.num_fn)
        rd_neg=zeros(self.num_fn)
        ac=zeros(3)
        bc=zeros(3)

        for j in range(0,self.num_fn):

            omega=2.*pi*self.fn[j]
            omegad=omega*sqrt(1.-(self.damp**2))

            E =exp(  -self.damp*omega*self.dt)
            E2=exp(-2*self.damp*omega*self.dt)

            K=omegad*self.dt
            C=E*cos(K)
            S=E*sin(K)

            Omr=(omega/omegad)
            Omt=omega*self.dt

            P=2*self.damp**2-1

            b00=2*self.damp*(C-1)
            b01=S*Omr*P
            b02=Omt

            b10=-2*Omt*C
            b11= 2*self.damp*(1-E2)
            b12=-2*b01

            b20=(2*self.damp+Omt)*E2
            b21= b01
            b22=-2*self.damp*C

            bc[0]=b00+b01+b02
            bc[1]=b10+b11+b12
            bc[2]=b20+b21+b22

            bc=-bc/(omega**3*self.dt)

            ac=SRS.a_coeff(omega,self.damp,self.dt)

            resp=lfilter(bc, ac, self.b, axis=-1, zi=None)

            rd_pos[j]= 386*max(resp)
            rd_neg[j]= 386*abs(min(resp))

        return rd_pos,rd_neg

########################################################################

class ReadData:
    def __init__(self):
        pass

    @classmethod
    def check_data(cls,a,b,num,sr,dt):
        sample_rate_check(a,b,num,sr,dt)
        return sr,dt

    def read_and_stats(self):
        label="Enter the base acceleration time history..."
        a, b, num = read_two_columns_from_dialog(label)
        sr, dt, ave, sd, rms, skew, kurtosis, dur = signal_stats(
            a, b, num)
        sr, dt = ReadData.check_data(
            a, b, num, sr, dt)
        return a, b, num, sr, dt, dur

#######################################################################

if __name__ == '__main__':
    print(" ")

    print (" ")
    print ("SRS using the Smallwood ramp invariant digital recursive")
    print ("filtering relationship")
    print (" ")

    print ("The file must have two columns:")
    print (" time(sec) & accel(G)")

    if 0:
        a, b, num, sr, dt, dur = ReadData().read_and_stats()

    from dynamight.core.srs import half_sine_pulse
    ymax = 1.
    tmax = 0.35
    tpulse = 0.011
    ntimes = 10001
    time, b = half_sine_pulse(ymax, tmax, tpulse, ntimes)

    dt = time[1] - time[0]
    sr = 1 / time.max()
    srs = SRS(b, dt, sr)
    srs.srs_plots()
