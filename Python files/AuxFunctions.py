# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la

import Config as C

def Potential(r):
    '''
    Function to return the potential for a given distance from a particle.
    Input: double giving the distance.
    Output: double giving the potential.
    
    (double) -> (double)
    '''
    return 4 * ( r**(-12) - r**(-6)  ) 

def PotentialDeriv(r):
    '''
    Function to return the derivative of the potential for a given distance from a particle.
    Input: double giving the distance.
    Output: double giving the derivative of the potential.
    
    (double) -> (double)
    '''
    return -4 * (12 * r**(-13) - 6 * r**(-7))

def calckinenergy(framesvar):
    '''
    Function to calculate the kinetic energy by taking the norm of the velocities.
    Input: np.array with the positions and velocities for each timestep.
    Output: double giving the kinetic energy.
    
    (np.array) -> (double)
    '''
    velvecs = framesvar[-1,1]

    return 0.5 * (np.sum(la.norm(velvecs, axis = 1)**2))

def calcpotenergy(framesvar):
    '''
    Function to calculate the potential energy by calculating the potential for each particle.
    Input: np.array with the positions and velocities for each timestep.
    Output: double giving the potential energy.
    
    (np.array) -> (double)
    '''
    posvecs = framesvar[-1,0]
    distancevec = np.zeros(3)  
    
    potenergy = 0
        
    for i in range(0,C.n): 
        for j in range(i,C.n):  
            #expression from lecutre notes to calc the distance according to the min image conv.
            distancevec = (posvecs[i] - posvecs[j] + C.L/2) % C.L - C.L/2            
            distance = la.norm(distancevec)
                       
            if i != j:                
                potenergy +=  Potential(distance)
       
    return (potenergy) 
