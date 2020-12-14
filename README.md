# Double-Pendulum-Chaos-Theory-
double pendulum with position graph, phasespace , centre of mass, total energy, polar and cartesian coordinate and relative error estimation.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 03:18:02 2020

@author: sudevpradhan
CHAOTIC PENDULUM
"""

#packages inclusion
import numpy as np
from numpy import cos, sin, arange, pi
import matplotlib.pyplot as plt
from IPython.display import display, Image
import matplotlib.animation as animation
import matplotlib.cm as cm
from IPython.display import HTML





# initial value
h = 0.0002   #the chenge in runge kutta
figsize = 6
dpi = 1000
N = 10000 # iterations
L1=1    #length 1
L2=1.5  #lenth 2
m1=1.5 #mass of bob 1
m2=1    #mass of bob2
g = 9.81#gravity

# dw/dt function oft theta 1

def funcdwdt1(theta1,theta2,w1,w2):

    cos12 = cos(theta1 - theta2)#for wrirting the main equation in less complex manner
    sin12 = sin(theta1 - theta2)
    sin1 = sin(theta1)
    sin2 = sin(theta2)
    denom = cos12**2*m2 - m1 - m2
    ans = ( L1*m2*cos12*sin12*w1**2 + L2*m2*sin12*w2**2
            - m2*g*cos12*sin2      + (m1 + m2)*g*sin1)/(L1*denom)
    return ans

# dw/dt function oft thetas 2
    
def funcdwdt2(theta2,theta1,w1,w2):

    cos12 = cos(theta1 - theta2)
    sin12 = sin(theta1 - theta2)
    sin1 = sin(theta1)
    sin2 = sin(theta2)
    denom = cos12**2*m2 - m1 - m2
    ans2 = -( L2*m2*cos12*sin12*w2**2 + L1*(m1 + m2)*sin12*w1**2
            + (m1 + m2)*g*sin1*cos12  - (m1 + m2)*g*sin2 )/(L2*denom)
    return  ans2

# d0/dt function for theta 1

def funcd0dt1(w0):
    return w0

# d0/dt function for theta 2
    
def funcd0dt2(w0):
    return w0


# Runge kutta 4th order algorithm for "coupled ODE"

def Runjetheta(w1,w2, theta1,theta2):
    #here actually we are solving 2 coupled ode for theta 1 and theta2 so we took,
    #k1a,k2a,k3a,k4a, k1b, k2b ,k3b, k4b for theta 1
    #k1c,k2c,k3c,k4c, k1d, k2d ,k3d, k4d for theta 2
    k1a = h * funcd0dt1(w1)  # gives theta1
    k1b = h * funcdwdt1(theta1,theta2,w1,w2)  # gives omega1
    k1c = h * funcd0dt2(w2)  # gives theta2
    k1d = h * funcdwdt2(theta2,theta1,w1,w2)   # gives omega2

    k2a = h * funcd0dt1(w1 + (0.5 * k1b))
    k2b = h * funcdwdt1(theta1 + (0.5 * k1a),theta2,w1,w2)
    k2c = h * funcd0dt2(w2 + (0.5 * k1d))
    k2d = h * funcdwdt2(theta2 + (0.5 * k1c),theta1,w1,w2)

    k3a = h * funcd0dt1(w1 + (0.5 * k2b))
    k3b = h * funcdwdt1(theta1 + (0.5 * k2a),theta2,w1,w2)
    k3c = h * funcd0dt2(w2 + (0.5 * k2d))
    k3d = h * funcdwdt2(theta2 + (0.5 * k2c),theta1,w1,w2)

    k4a = h * funcd0dt1(w1 + k3b)
    k4b = h * funcdwdt1(theta1 + k3a,theta2,w1,w2)
    k4c = h * funcd0dt2(w2 + k3d)
    k4d = h * funcdwdt2(theta2 + k3c,theta1,w1,w2)

    #addidng the vakue aftyer the iterartions
    theta001 = theta1 + 1 / 6 * (k1a + 2 * k2a + 2 * k3a + k4a)   # gives change in theta1
    w001 = w1 + 1 / 6 * (k1b + 2 * k2b + 2 * k3b + k4b)             # gives change in omega1
    theta002 = theta2 + 1 / 6 * (k1c + 2 * k2c + 2 * k3c + k4c)     # gives change in theta2
    w002 = w2 + 1 / 6 * (k1d + 2 * k2d + 2 * k3d + k4d)              # gives change in omega2
    return theta001, w001, theta002, w002


# calling Runge Kutta function

def kutta(w1,w2, theta01,theta02):
    #initialising the initial theta 1 and theta 2 value and omega 1 and omega 2 value to the list  
    Theta1 = (N + 1) * [0] #initialising all the list value theta1, theta2, omega1, omega2 to 0
    Omega1 = (N + 1) * [0]
    Theta2 = (N + 1) * [0]
    Omega2 = (N + 1) * [0]

    Theta1[0] = theta01   #theta1 first value to the user speicified value
    Omega1[0] = w1        #omega1 first value to the user speicified value
    Theta2[0] = theta02   #theta2 first value to the user speicified value
    Omega2[0] = w2        #omega2 first value to the user speicified value
    
    #iterations for runnning the runge kutta N times
    for i in range(0, N):
        #appending the value of theta1, theta 2, omega1 and omega2 with iterating N no. of times runjekutta function
        Theta1[i + 1], Omega1[i + 1],Theta2[i+1], Omega2[i+1] = Runjetheta(Omega1[i],Omega2[i], Theta1[i],Theta2[i])

    return Theta1,Theta2, Omega1, Omega2
    
def polar(x1,y1,x2,y2):
    #this particular function will change the cartesian coordinate to polar coordinate
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar')) 
    fig.suptitle('Polar Cordinate')
    ax1.plot(x1, y1)
    ax2.plot(x1, y2)

    plt.show()
    
    
#function to plot the position graph of the bob1 and bob 2 trackinfg their position with respect to time

def position(x1, y1, x2, y2, theta_1, theta_2, t):
    
    
    plt.figure(figsize=(2*figsize, figsize), dpi=dpi)
    plt.style.use('seaborn')  #using seaborn package
    

    # theta t-plot to show the relation netween time and theta 1, time with theta 2
    ax = plt.subplot(2, 2, 2)
    ax.plot(t, theta_1, label=r"$\theta_1(t)$")
    ax.plot(t, theta_2, label=r"$\theta_2(t)$")
    plt.ylabel(r"$\theta$, [rad]")
    plt.xlabel(r"$t$, [s]")
    ax.legend()

    # omega t-plot to show the relation netween time and omega 1, time with tomega 2

    ax = plt.subplot(2, 2, 4)
    ax.plot(t, omega_1, label=r"$\omega_1(t)$")
    ax.plot(t, omega_2, label=r"$\omega_2(t)$")
    plt.ylabel(r"$\omega$, [rad/s]")
    plt.xlabel(r"$t$, [s]")
    ax.legend()

    plt.show()
    
    # the cartesian axis plot of x1 and y1 that indirectly tracks the position of both the bobs with respect to omega
    L = 1.1*(L1 + L2)

    plt.style.use('seaborn')
    plt.figure(figsize=(1.5*figsize, figsize), dpi=dpi )
    plt.plot(x1, y1,"orangered",label=r"Track $m_1$")
    plt.plot(x2, y2,"teal", label=r"Track $m_2$")
    plt.plot([0, x1[0], x2[0]], [0, y1[0], y2[0]], "-o", label="Initial position", c='k')
    plt.ylabel(r"$y/L$-->")
    plt.xlabel(r"$x/L$-->")
    plt.figtext(0.05,0.07, "$m_1$=1.5kg,$m_2$=1kg")
    plt.figtext(0.05, 0.04, "$l_1$=1mtr,$l_2$=1.5mtr")
    xlim=(-L, L)
    ylim=(-L, L)
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1,loc='lower left')
    plt.show()

def phasespace(theta1, w1, theta2, w2):
    #A phase space is a space in which all possible states of a system can be represented and each 
    #possible state can be seen corresponding to a single unique point in the phase space
    #The double pendulum has 4 degrees of freedom:angular velocity (ω) and the angle (θ) , so in general (θ1,θ2,ω1andω2)
    
    plt.figure(figsize=(2*figsize, figsize), dpi=dpi)
    plt.title(r"Phase-space diagram, $\theta_{1}=1^{\circ}$, $\theta_{2}=2^{\circ}$ " + r"$\omega_{1}=0$, $\omega_{2}=0$")
    plt.plot(theta1, w1,"steelblue", label=r"$theta1 vs omega 1$")
    plt.plot(theta2, w2,"coral", label=r"$theta2 vs omega 2$")
    plt.legend(loc='lower center',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.figtext(0.08,0.06, "$m_1$=1.5kg , $m_2$=1kg")
    plt.figtext(0.08, 0.03, "$l_1$=1mtr , $l_2$=1.5mtr")
    plt.xlabel(r"$\theta_i$, (rad)")
    plt.ylabel(r"$\omega_i$, (rad/s)")
    xlim = [np.min(theta1), np.max(theta1), np.min(theta2), np.max(theta2)]
    plt.xlim(np.min(xlim), np.max(xlim))
    plt.show()  


def cartesian(theta_1, omega_1, theta_2, omega_2, L1, L2):
    #Converting theta1,theta2, omega1 and omega 2 into cartesian co-ordinates
    x1 = L1 * sin(theta_1)
    #x1     : x-posision of mass 1
    y1 = -L1 * cos(theta_1)
    #y1     : y-posision of mass 1
    x2 = x1 + L2 * sin(theta_2)
    #x2     : x-posision of mass 2
    y2 = y1 - L2 * cos(theta_2)
    #y2     : y-posision of mass 2

    return x1, y1, x2, y2

def theta1v2(theta1, theta2 ):
  
    #plotting the graph between theta1 and theta 2 to study chaos
    fig=plt.figure(figsize=(7,4),dpi=600)
    plt.title(r"Theta1($\theta_1$) vs. Theta2($\theta_1$)")
    plt.plot(theta1, theta2,"indianred", label=r"variations in $\theta$")
    plt.legend(loc='lower right',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel(r"$\theta_1$-->")
    plt.ylabel(r"$\theta_2$-->")
    plt.figtext(0.05,0.04, "$m_1$=1.5kg,$m_2$=1kg")
    plt.figtext(0.05, 0.001, "$l_1$=1mtr,$l_2$=1.5mtr")
    plt.show() 

def omega1v2(omega1, omega2):
  
    #plotting the graph between omega1 and omega2 to study the trend.
    fig=plt.figure(figsize=(7,4),dpi=600)
    plt.title(r"Omega1($\omega_1$) vs. Omega2($\omega_1$)")
    plt.plot(omega1, omega2,color="steelblue", label=r"Variation in $\omega$")
    plt.legend(loc='lower left',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel(r"$\omega_1$-->")
    plt.ylabel(r"$\omega_2$-->")
    plt.figtext(0.05,0.04, "$m_1$=1.5kg,$m_2$=1kg")
    plt.figtext(0.05, 0.001, "$l_1$=1mtr,$l_2$=1.5mtr")
    plt.show() 
    
#Potential energy function    
def PotentialEnergy(theta1, theta2):

    return -1*m1*g*L1*( np.cos(theta1)) - m2*g*(L1*np.cos(theta1) + L2*np.cos(theta2))
#Kinetic energy function  
def KineticEnergy(w1,w2,theta1,theta2):
    preity= (0.5*m1*(L1**2)*(np.array(w1)**2)) + (0.5*m2*((L1**2)*(np.array(w1)**2) + (L2**2)*(np.array(w2)**2) +(2*L1*L2*cos(np.array(theta1) - np.array(theta2))*np.array(w1)*(np.array(w2)))))

    return preity


#graph for the conservation of energy
    
def draw_energy(theta1, theta2, omega1, omega2):
    #here we have plotted the total energy, kinteic energy and potential energy with respect to time on a comman axis
    fig=plt.figure(figsize=(7,4),dpi=600)
    plt.title(r"Mechanical energy")
    plt.plot(t, PotentialEnergy(theta1, theta2),"seagreen" ,label=r"Potential energy")
    plt.plot(t, KineticEnergy(omega1, omega2, theta1, theta2),"steelblue", label=r"Kinetic energy")
    plt.plot(t, PotentialEnergy(theta1, theta2) + KineticEnergy(omega1, omega2, theta1, theta2),"black", label=r"Total energy")
    plt.xlabel(r"$time$ (sec)")
    plt.ylabel(r"$Total Energy$ (E)")
    plt.legend(loc='center right',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.figtext(0.05,0.04, "$m_1$=1.5kg,$m_2$=1kg")
    plt.figtext(0.05, 0.001, "$l_1$=1mtr,$l_2$=1.5mtr")
    plt.show()

#centre of mass function
def C_O_M(x1, x2, y1, y2):
  
    U=((m1*x1) +(m2*x2) )/(m1+m2)
    cheli= ((m1*y1) +(m2*y2) )/(m1+m2)
     
    return U, cheli
#centre of mass plot
def draw_C_O_M(a,b):
  
    plt.style.use('seaborn')
    fig=plt.figure(figsize=(7,4),dpi=600)
    plt.title(r"Centre Of Mass")
    plt.plot(a,b, "teal", label=r"Position of $COM$")
    plt.legend(loc='lower left',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel(r"x -->")
    plt.ylabel(r"y -->")

    plt.figtext(0.05,0.04, "$m_1$=1.5kg,$m_2$=1kg")
    plt.figtext(0.05, 0.001, "$l_1$=1mtr,$l_2$=1.5mtr")

    plt.show() 
    
    #this plot include the position time plot which traces the position of the bob and here we have mixed the centre of mass plot to visualise easily
    L = 1.1*(L1 + L2)

    plt.style.use('seaborn')
    plt.figure(figsize=(1.5*figsize, figsize), dpi=dpi )
    plt.plot(x1, y1,"orangered",label=r"Track $m_1$")
    plt.plot(x2, y2,"teal", label=r"Track $m_2$")
    plt.plot(a,b, "sandybrown", label=r"Position of $COM$")
    plt.plot([0, x1[0], x2[0]], [0, y1[0], y2[0]], "-o", label="Initial position", c='k')
    plt.ylabel(r"$y/L$-->")
    plt.xlabel(r"$x/L$-->")
    plt.figtext(0.05,0.07, "$m_1$=1.5kg,$m_2$=1kg")
    plt.figtext(0.05, 0.04, "$l_1$=1mtr,$l_2$=1.5mtr")
    xlim=(-L, L)
    ylim=(-L, L)
    plt.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1,loc='lower left')
    plt.show()
    
def Error(theta1, theta2, w1, w2):
     #Computes the relative error to print the output
    E0 = -1*m1*g*L1*( np.cos(theta1[0])) - m2*g*(L1*np.cos(theta1[0]) + L2*np.cos(theta2[0])) + (0.5*m1*(L1**2)*(np.array(w1[0])**2)) + (0.5*m2*((L1**2)*(np.array(w1[0])**2) + (L2**2)*(np.array(w2[0])**2) +(2*L1*L2*cos(np.array(theta1[0]) - np.array(theta2[0]))*np.array(w1[0])*(np.array(w2[0])))))
    E1 = -1*m1*g*L1*( np.cos(theta1[1])) - m2*g*(L1*np.cos(theta1[1]) + L2*np.cos(theta2[1])) + (0.5*m1*(L1**2)*(np.array(w1[1])**2)) + (0.5*m2*((L1**2)*(np.array(w1[1])**2) + (L2**2)*(np.array(w2[1])**2) +(2*L1*L2*cos(np.array(theta1[1]) - np.array(theta2[1]))*np.array(w1[1])*(np.array(w2[1])))))
    return np.abs((E0 - E1)/E0)

#here we are trying to find relative error for each term from the first total energy as that is the true value entered by the user rest all value are generated by the function andwe are comparing it from that
def error_graph(b,theta1, theta2, w1, w2):
    a=b+1
    E0 = -1*m1*g*L1*( np.cos(theta1[0])) - m2*g*(L1*np.cos(theta1[0]) + L2*np.cos(theta2[0])) + (0.5*m1*(L1**2)*(np.array(w1[0])**2)) + (0.5*m2*((L1**2)*(np.array(w1[0])**2) + (L2**2)*(np.array(w2[0])**2) +(2*L1*L2*cos(np.array(theta1[0]) - np.array(theta2[0]))*np.array(w1[0])*(np.array(w2[0])))))
    E1 = -1*m1*g*L1*( np.cos(theta1[a])) - m2*g*(L1*np.cos(theta1[a]) + L2*np.cos(theta2[a])) + (0.5*m1*(L1**2)*(np.array(w1[a])**2)) + (0.5*m2*((L1**2)*(np.array(w1[a])**2) + (L2**2)*(np.array(w2[a])**2) +(2*L1*L2*cos(np.array(theta1[a]) - np.array(theta2[a]))*np.array(w1[a])*(np.array(w2[1])))))
    return np.abs((E0 - E1)/E0)

#plotting the error graph , where we can see the max erro is in the range og 10^-4, which is negligible
def show_error_graph():
    for i in range(1, N):
        Energy = (N) * [0]

    for i in range(0, N):    
        Energy[i]=error_graph(i, theta_1,theta_2, omega_1,omega_2)
    

    T=np.linspace(0,N,N)
    plt.style.use('seaborn')
    fig=plt.figure(figsize=(7,4),dpi=600)
    plt.title(r"Realtive Error Curve")
    plt.plot(T,Energy, "teal", label=r"Relative error")
    plt.legend(loc='best',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.xlabel(r"iteration or steps")
    plt.ylabel(r"Relative error (E0-E1)/E0 ")
    plt.figtext(0.05,0.04, "$m_1$=1.5kg,$m_2$=1kg")
    plt.figtext(0.05, 0.001, "$l_1$=1mtr,$l_2$=1.5mtr")
    plt.show() 
    


'''def create_animation(filename, x1, y1, x2, y2, tmax, L1, L2):

    
    fig = plt.figure(figsize=(4, 4), dpi=60)
    L = 1.1*(L1 + L2)
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-L, L), ylim=(-L, L))

    tail1, = ax.plot([],[],'r') # Tail for m2
    line, = ax.plot([], [], '-o', lw=2, c="k")
    time_template = r'$t = %.1fs$'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    ax.set_aspect('equal')
    ax.axis('off')
    FPS = 60
    framesNum = int(FPS*tmax)
    frames = np.floor(np.linspace(0, len(x1) - 1, framesNum)).astype(np.int)
    def init():
        line.set_data([], [])
        tail1.set_data([], [])
        time_text.set_text('')
        return line, time_text

    def animate(i):
        line.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])
        tail1.set_data(x2[:i], y2[:i])
        time_text.set_text(time_template % (i*0.1))
        return line, time_text, tail1

    anim = animation.FuncAnimation(fig, animate, frames=frames)
    anim.save("test.gif", writer='imagemagick', fps=FPS)
    plt.close(anim._fig)
   
    # Display the animation
    #HTML(anim.to_html5_video())
    with open("test.gif",'rb') as file:
        display(Image(file.read()))

tmax=50

        
create_animation("double_pendulum", x1, y1, x2, y2, tmax, L1, L2)   ''' 


#---------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL


#initialisnig first pair of theta
theta_01 = ((np.pi)/180)*1
theta_02 = ((np.pi)/180)*2
#for time axis in the plot where we run the function for 10 seconds with N+1 intervals
t=np.linspace(0,10,N+1)

#saving the list of the runge utta derived theta and omega into theta_1,theta_2, omega_1,omega_2
theta_1,theta_2, omega_1,omega_2 = kutta(0,0 ,theta_01,theta_02)
#changing theta 1, theta 2, omega1 and omega 2 into cartesian coordinate, x1,x2 ,y1 and y2 respectively
x1, y1, x2, y2 = cartesian(theta_1, omega_1, theta_2, omega_2, L1, L2)

#animation
#activate the code for animation 
'''fig, ax = plt.subplots()
l1, = ax.plot([], [])
l2, = ax.plot([],[])
ax.set(xlim=(-2, 2), ylim=(-2,2))
def animate(i):
	l1.set_data(x1[:i], y1[:i])
	l2.set_data(x2[:i], y2[:i])
	return l1,l2,

ani = animation.FuncAnimation(fig, animate, interval = 1, frames=len(x1))
ani.save('save.mp4', writer='ffmpeg')  #this will be saved as mp4 in your folder
#plt.show()'''

position(x1, y1, x2, y2, theta_1, theta_2, t)  #calling postion time function and printing position vs. time graph

phasespace(theta_1, omega_1, theta_2, omega_2)#calling Phasespace function and printing phasespce graph between theta and omega

theta1v2(theta_1, theta_2)#calling theta function and printing theta1 vs otheta2 graph

omega1v2(omega_1, omega_2) #calling omega function and printing omega1 vs omega2 graph
 
draw_energy(theta_1, theta_2, omega_1, omega_2) #calling Energy function and printing total energy  graph along with potential energy and kinetic energy
      
a,b= C_O_M(x1, x2, y1, y2)#calling Centre of mass function and printing COM graph
draw_C_O_M(a,b)  

polar(x1,y1,x2,y2) #calling polar function and printing poilar coordinate graph

show_error_graph() #plotting the error graph for each value of theta and omega , we plotted it in term of total energy(which used theta and omega)  and subtracted it from the initial energy as that energy is entered by the user which we kept as reference

#printing out the error
print("Relative error for 2 consecutive value of Energy:")
print("Theta1 = %0f and Theta2 = %0f: %0.4e"%((theta_01*180)/np.pi,(theta_02*180)/np.pi, Error(theta_1,theta_2, omega_1,omega_2)))



    


