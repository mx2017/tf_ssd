#!/usr/bin/env python  
# -*- coding: utf-8 -*-  
  
import numpy as np  
import matplotlib.pyplot as plt  
import math



def sigmoid(x,scale=1.0,gamma=1.0,x_offset=0,y_offset=0):
    a = []
    for item in x:
        val = 1.0/(1.0+math.exp(-gamma*(item+x_offset))) + y_offset
        if val <= 0:
            val = 0
        elif val > 1.0:
            val = 1.0
            
        a.append(val*scale)
    return a
    
def tanh(x,gamma):
    a = []
    for item in x:
        ex = math.exp(gamma*(item)) 
        e_x = math.exp(-gamma*(item)) 
        a.append((ex - e_x)/ (ex + e_x) + 0.5)
    return a
    
#t = np.arange(0.0, 1.01, 0.01)  
#s = np.sin(2*2*np.pi*t)  
  
#plt.fill(t, s*np.exp(-5*t), 'r')  
#plt.fill(t, t**2, 'r') 
ALPHA = 3
BETA = 0.0
GAMMA = 2

#plt.fill(t,(ALPHA*t+BETA)**2,t,(ALPHA*t+BETA)**2.5,t,(ALPHA*t+BETA)**1,t,(ALPHA*t+BETA)**1.5,t, (ALPHA*t+BETA)**0.5, 'r')
#plt.fill(t,(ALPHA*t+BETA)**GAMMA,'r')
x = np.arange(0., 1., 0.01)
'''
for i in range(6,7): 
    sig = sigmoid(x,12)
    plt.plot(x,sig)
'''

'''
sig = sigmoid(x,8)
plt.plot(x,sig)
'''
'''
sig = sigmoid(x,8,0.43)
plt.plot(x,sig)

sig = sigmoid(x,12,0.40)
plt.plot(x,sig)
'''

'''
sig = sigmoid(x,24,0.36)
plt.plot(x,sig)
'''

sig = sigmoid(x,scale=1.2,gamma=16,x_offset=-0.2,y_offset=-0.15)
plt.plot(x,sig)

'''
x1 = np.arange(-1., 1., 0.01)  
sig = tanh(x1,3)
plt.plot(x1,sig) 
x1 = np.arange(-1., 1., 0.01)  
'''
 
'''
x1 = np.arange(-1., 1., 0.01)  
for i in range(1,2): 
    sig = tanh(x1,6)
    plt.plot(x1,sig) 
'''

'''
sig = sigmoid(x,6)
plt.plot(x,sig)

sig = sigmoid(x,8)
plt.plot(x,sig)

sig = sigmoid(x,10)
plt.plot(x,sig)

sig = sigmoid(x,12)
plt.plot(x,sig)
'''


plt.plot(x,x)

plt.show()

#plt.fill(t,math.exp(-t),'r')

#plt.grid(True)  
  
#保存为PDF格式，也可保存为PNG等图形格式  
plt.savefig('test.pdf')  
#plt.show()  