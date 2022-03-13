# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 10:47:10 2022

@author: cdmeh
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 16:44:22 2022

@author: cdmeh
"""

# Imports necessary modules
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as scint

# Plots a series of comparison curves between CARRGO and
# the new Bertanlaffy model.
def main():
    
    x1,y1,z1,m1,n1 = NEWGO([0,0,0],[0.5,0,0,0.5,0])
    x2,y2,z2,m2,n2 = NEWGO([0,0,0],[0.5,0.5,0,0.5,0])
    x3,y3,z3,m3,n3 = NEWGO([0.25,0.25,0.25],[0.5,0.5,0,0.5,0])
    t = np.linspace(0,40,len(x1))
    label1 = ["CARRGO: No Dose","CARRGO: One Dose","CARRGO: Four Doses", "New: No Dose","New: One Dose","New: Four Doses"]
    plt.figure(1)
    plt.grid(True)
    plt.title("Possible Dynamics of Glioma/CAR T-Cell Interaction")
    plt.ylabel("Cancer Cell Count")
    plt.xlabel("Time Elapsed")
    plt.plot(t,x1, label = label1[0])
    plt.plot(np.linspace(0,40,len(x2)),x2, label = label1[1])
    plt.plot(np.linspace(0,40,len(x3)),x3, label = label1[2])
    plt.legend()
    plt.show()
    plt.close(fig = 1)
    plt.figure(2)
    plt.grid(True)
    plt.title("Possible Dynamics of Glioma/CAR T-Cell Interaction")
    plt.ylabel("CAR T-Cell Count")
    plt.xlabel("Time Elapsed")
    plt.plot(t,y1, label = label1[0])
    plt.plot(np.linspace(0,40,len(y2)),y2, label = label1[1])
    plt.plot(np.linspace(0,40,len(y3)),y3, label = label1[2])
    plt.legend()
    plt.show()
    plt.close(fig = 2)
    plt.figure(3)
    plt.grid(True)
    plt.title("Possible Dynamics of Glioma/CAR T-Cell Interaction")
    plt.ylabel("Cancer Cell Count")
    plt.xlabel("Time Elapsed")
    plt.plot(t,z1, label = label1[0])
    plt.plot(np.linspace(0,40,len(z2)),x2, label = label1[1])
    plt.plot(np.linspace(0,40,len(z3)),x3, label = label1[2])
    plt.legend()
    plt.show()
    plt.close(fig = 3)
    plt.figure(4)
    plt.grid(True)
    plt.title("Possible Dynamics of Glioma/CAR T-Cell Interaction")
    plt.ylabel("Cancer Cell Count")
    plt.xlabel("Time Elapsed")
    plt.plot(t,m1, label = label1[0])
    plt.plot(np.linspace(0,40,len(m2)),m2, label = label1[1])
    plt.plot(np.linspace(0,40,len(m3)),m3, label = label1[2])
    plt.legend()
    plt.show()
    plt.close(fig = 4)
    plt.figure(5)
    plt.grid(True)
    plt.title("Possible Dynamics of Glioma/CAR T-Cell Interaction")
    plt.ylabel("Cancer Cell Count")
    plt.xlabel("Time Elapsed")
    plt.plot(t,n1, label = label1[0])
    plt.plot(np.linspace(0,40,len(n2)),n2, label = label1[1])
    plt.plot(np.linspace(0,40,len(n3)),n3, label = label1[2])
    plt.legend()
    plt.show()
    plt.close(fig = 5)
    
    
    

def NEWGO(DOSE,z0):
    # Parameters of the simulation
    # Final Time
    tf = 40
    # Number of steps
    n = 1000
    # Number of plotted trajectories
    m = 15
    tspan = (0,tf)
    # System parameters
    P = (1.87,1.6,1,1,1,1,1,1,1,1,2,3,2)
    
    
    sol1 = scint.solve_ivp(dw,(0,tf//4),z0, args = P,max_step = tspan[-1]/n)
    z1 = [sol1.y[0][-1],sol1.y[1][-1]+DOSE[0],sol1.y[2][-1],sol1.y[3][-1],sol1.y[4][-1]]
    sol2 = scint.solve_ivp(dw,(0,tf//4),z1, args = P,max_step = tspan[-1]/n)
    z2 = [sol2.y[0][-1],sol2.y[1][-1]+DOSE[1],sol2.y[2][-1],sol2.y[3][-1],sol2.y[4][-1]]
    sol3 = scint.solve_ivp(dw,(0,tf//4),z2, args = P,max_step = tspan[-1]/n)
    z3 = [sol3.y[0][-1],sol3.y[1][-1]+DOSE[2],sol3.y[2][-1],sol3.y[3][-1],sol3.y[4][-1]]
    sol4= scint.solve_ivp(dw,(0,tf//4),z3, args = P,max_step = tspan[-1]/n)
    sx = np.hstack((sol1.y[0],sol2.y[0],sol3.y[0],sol4.y[0]))
    sy = np.hstack((sol1.y[1],sol2.y[1],sol3.y[1],sol4.y[1]))
    sz = np.hstack((sol1.y[2],sol2.y[2],sol3.y[2],sol4.y[2]))
    sm = np.hstack((sol1.y[3],sol2.y[3],sol3.y[3],sol4.y[3]))
    sn = np.hstack((sol1.y[4],sol2.y[4],sol3.y[4],sol4.y[4]))
    # Plots trajectories and boundary between regions of concavity
    
    return sx,sy,sz,sm,sn
def dw(t,z,A,B,C,D,E,G,H,I,J,K,L,M,N):
    F = np.power(A/B,1/3)
    dx = A*np.power(z[0],2/3) - B*z[0] - (C+D*z[3])*z[1]*np.power(z[0],2/3)
    dy = -1*E*(1 - np.power(z[0],2/3)/F)*z[1] + G*z[2]*np.power(z[0],2/3) + H*z[1]*np.power(z[0],2/3) - I*z[1]
    dz = E*(1 - np.power(z[0],2/3)/F)*z[1] - G*z[2]*np.power(z[0],2/3) - J*z[2]
    dm = (L - K)*z[3]*np.power(z[0],2/3) + M*z[3]*z[4] - N*z[3]
    dn = K*z[3]*np.power(z[0],2/3) - M*z[3]*z[4] - N*z[4]
    return [dx,dy,dz,dm,dn]

# Executes function
main()