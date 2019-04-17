# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la

import Config as C
import AuxFunctions as AF

def histogram(framesvar):
    '''
    Function to calculate the pair dist. function for 1 timestep.
    Input: np.array with the positions and velocities for each timestep.
    Output: np.array with for each bin the number of particles in that bin, normalised using the 
    prefactor as given in the report.
    
    (np.array) -> (np.array) 
    '''
    posvecs = framesvar[-1][0] 
    distancevec = np.zeros(3)  
           
    hist = np.zeros(int(np.ceil((0.5*C.L) / C.dr)) )
   
    #loop over all particles and bin them according to the distance to the reference particle.
    for i in range(0,C.n): 
        for j in range(i,C.n): 
            if i != j:
                distancevec = (posvecs[i] - posvecs[j] + C.L/2) % C.L - C.L/2       
                distance = la.norm(distancevec)
                
                if distance < 0.5*C.L and distance > C.dr:
                    hist[int(distance / C.dr)] += 1

    rvec = np.arange(C.dr, 0.5 * C.L + C.dr, C.dr)
       
    hist = (hist / rvec**2)
    
    hist = hist * (1 / (C.dr * 4 * np.pi)) * (2*C.L**3 / (C.n * (C.n - 1)))
    
    return np.asarray(hist)

def SpecificHeat(KinEnergy, temp):
    '''
    Function to calculate the specific heat at a given temperature using the definition from
    the report.
    Input: np.array with the kinetic energy for each timestep and an double which gives
    the temperature.
    Output: Double which gives the specific heat.
    
    (np.array, double) -> double
    '''
    squaredAverage = np.mean(np.square(KinEnergy))
    squareOfAverage = np.mean(KinEnergy)**2
    
    return (3*C.n*squareOfAverage)/((3*C.n + 2)*squareOfAverage - 3*C.n*squaredAverage)

def Difussion(frames):
    '''
    Function to calculate the mean diffusion for a timestep using the distance traveled for each
    particle from its starting position.
    Input: np.array with the positions and velocities for each timestep.
    Output: double which gives the average diffusion.
    
    (np.array) -> double
    '''
    posvecs = np.array([x[0] for x in frames[0:]])
    
    return np.mean(la.norm((posvecs - frames[0][0] + C.L/2) % C.L - C.L/2, axis = 2)**2, axis = 1)

def Pressure(frames, temp):
    '''
    Function to calculte the pressure for a temperature.
    Input: np.array with the positions and velocities for each timestep and a double giving 
    the temperature.
    Output: Double giving the pressure for that temperature.
    
    (np.array, double) -> (double)
    '''
    listt = []
    
    for t in range(len(frames)):
        posvecs = frames[t,0]
        
        listn = []
        
        for i in range(0,C.n): 
            for j in range(0,C.n):  
                #expression from lecture notes to calculate the distance according to the minimal
                #image convention.                
                distancevec = (posvecs[i] - posvecs[j] + C.L/2) % C.L - C.L/2       
                distance = la.norm(distancevec)
                    
                if i != j:
                    listn.append(distance * AF.PotentialDeriv(distance))
                    
        listt.append(1/2 * np.sum(listn))
                           
    P = np.mean(listt) 
    #P = (1/(6*C.L**3)) * P
    #P = (C.n * temp)/(C.L**3) - P
    P = (1/(3*C.n * temp)) * P
    P = 1 - P
    
    return P          