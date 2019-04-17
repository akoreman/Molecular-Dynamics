# -*- coding: utf-8 -*-
'''
todo: timing
'''

#dependencies
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


#import timeit
import AuxFunctions as AF
import Observables as O
import Config as C

def initarr(T, method):
    """
    Function to Initialize the velocities according to a maxwell distribution and the positions either
    randomly or on a cubic lattice.
    
    (float, float, int) -> (2d double array)
    
    input parameters: 
    float: mean of the normal distribution of the initial velocities
    float: standard deviation of the normal distribution of the initial velocities    
    int: Parameter to chose the method to initialize the postions. Choose 0 for random positions or 1 for a 
    face centered cubic lattice.  
    """
    
    velvecs =  np.random.normal(0,np.sqrt(T),(C.n,3))
    
    #velvecs = (2 * np.pi)**(1/2) * velvecs
    velvecs = velvecs - np.mean(velvecs, axis = 0)  
    
    if method == 0:
        posvecs = np.random.uniform(0,C.L,(C.n,3))
        
        return np.array([[posvecs,velvecs]])

    #using the unit cell of a FCC lattice construct the position vectors as an FCC lattice.
    if method == 1:
        unitCell = np.array([[0,1,1],[1,0,1],[1,1,0],[0,0,0]])*0.5
        posvecs = np.zeros((C.n,3))
                
        vec = np.arange(0,int(np.ceil(C.L/(1*C.gridspacing))))

        l = 0
        
        for i in range(vec.size):
            for j in range(vec.size):
                for k in range(vec.size):   
                    posvecs[l:l + 4] = np.asarray([x + np.array([i,j,k]) for x in unitCell])
                    l += 4
              
        posvecs = (posvecs + 0.25)*C.gridspacing
                        
        return np.array([[posvecs,velvecs]])
                           
def calcforce(framesvar): 
    '''
    Function to calculate the forces between each of the particles as a vector of the total force for 
    each of the particles.
    Input: np.array with the positions and velocities for each timestep.
    Output: np.array with the total force vec for each particle.
    
    (np.array) -> (np.array)
    '''
    posvecs = framesvar[-1,0]
    
    forcevecs = np.zeros((C.n,3)) 
    distancevec = np.zeros(3)

    distance = 0
      
    #loop over all particles and calculate the force between each pair, using the symmetry one can
    #run the second loop from i to n to minimize the loops      
    for i in range(0,C.n): 
        for j in range(i,C.n):  
            #expression from lecutre notes to calculate the distance according to the minimal
            #image convention.
            distancevec = (posvecs[i] - posvecs[j] + C.L/2) % C.L - C.L/2
            distance = la.norm(distancevec)
            
            if i != j:
                forcevecs[i] +=  -1 * AF.PotentialDeriv(distance) * distancevec/distance
                forcevecs[j] +=  AF.PotentialDeriv(distance) * distancevec/distance
               
    return forcevecs         
                             
def nextframe(framesvar, forcesvar, method):
    '''
    Function to calcute the next frame given the current frame and the forces on each particle. 
    The third parameter is an int to choose the integration method: 0 for euler, 1 for verlet and 2 
    for velocity verlet.
    Input: np.array with the positions and velocities for each timestep, np.array with the
    force vecs for each particle and an int to choose the integration method.
    
    (np.array, np.array, int) -> (np.array)
    '''
    posvecs = framesvar[-1,0]
        
    if method != 0 and len(framesvar) >= 2:
        posvecsold = framesvar[-2,0]
            
    velvecs = framesvar[-1,1]       
    forcevecs = forcesvar  
     
    if method == 0:                          
        posvecs = (posvecs + velvecs*C.dt)%C.L          
        velvecs += forcevecs*C.dt
        
        nextframe = np.array([[posvecs,velvecs]])
   
    if method == 1:
        if len(framesvar) < 2: #calculate the first frame using euler because verlet needs the previous position.                               
            posvecs = (posvecs + velvecs*C.dt)%C.L              
            velvecs += (forcevecs)*C.dt
                       
            nextframe = np.array([[posvecs,velvecs]])
        else: 
            posvecs = (2*posvecs - posvecsold + C.dt**2 * forcevecs)%C.L               
            velvecs = (posvecs - posvecsold)/(2*C.dt)
                            
            nextframe = np.array([[posvecs,velvecs]])
  
    if method == 2: 
        velvecs = velvecs + (C.dt/2) * forcevecs 
        posvecs = posvecs + C.dt * velvecs
        velvecs = velvecs + (C.dt/2) * calcforce(np.array([[posvecs,velvecs]]))

        nextframe = np.array([[posvecs,velvecs]])          
        
    return nextframe
    
def rescaleeq(framesvar,temp):
    '''
    Function to rescale the velocities according to the definition given in the report.
    Input: np.array with the positions and velocities for each timestep and double given the temperature.
    Output: np.array with the positions and rescaled velocities for each timestep.
    
    (np.array, double) -> (np.array)
    '''
    velvecs = framesvar[-1,1] 

    lab = np.sqrt( (3*(C.n-1)* temp) /(np.sum(pow(la.norm(velvecs, axis = 1),2))))
    
    framesvar[-1,1] = lab * velvecs
    
    return framesvar 

def bootStrapP(framesvar,T,bitr):
    '''
    Function to calculate the error in the pressure using the bootstrap method described in the report.
    Input: np.array with the positions and velocities for each timestep, double giving the temperature 
    and an int giving the number of bootstrap iterations.
    Output: double giving the error in the pressure at a given temperature.
    
    (np.array, double, int) -> (double)
    '''
    lst = np.zeros(bitr)    
    n = len(framesvar)
        
    for i in range(0,bitr):
        lst[i] = O.Pressure(framesvar[np.random.choice(framesvar.shape[0], size = n)], T)
    
    return np.std(lst)

def bootStrapCv(energyList,T,bitr):
    '''
    Function to calculate the error in the specific heat using the bootstrap method described in the report.
    Input: np.array with a list of the kinetic energy at each timestep, double giving the temperature 
    and an int giving the number of bootstrap iterations.
    Output: double giving the error in the specific heat at a given temperature.
    
    (np.array, double, int) -> (double)
    '''
    lst = np.zeros(bitr)    
    n = len(energyList)
        
    for i in range(0,bitr):
        lst[i] = O.SpecificHeat(energyList[np.random.choice(energyList.shape[0], size = n)], T)
    
    return np.std(lst)

def runsim(tempList, rho, M, bitr, nsteps):   
    '''
    Function to run the simulation at a temperature and densities.
    Input: list giving the temperatures, double giving the density, an integer giving 
    the number of unit cells per axis. an integer giving the number
    of steps to run and an int to give the number of bootstarp iterations.
    Output: void
    
    (list, double, int, int, int) -> ()
    '''
    CvList = []
    PList = [] 

    C.n = 4*(M)**3
    C.gridspacing = (4/rho)**(1/3)
    C.L = M*C.gridspacing     
         
    #loop over the temps you want to look at.
    for m in range(len(tempList)):
        print("calculating for T = " + str(tempList[m]) + " and rho = " + str(rho))
        
        T = tempList[m]            
        frames = initarr(T,1)             
        totenergy = []            
        PDFpert = []
        
        KinEnergyList = []
        PotEnergyList = []
        
        #loop over each timestep.
        for t in range (0,nsteps):  
            forces = calcforce(frames)
      
            frames = np.append(frames,nextframe(frames,forces,2),axis = 0)
            
            #code used to flip the velocities for the reversibility study.
            #if t == 175:
            #    frames[-1][1] = -1 * frames[-1][1]
            
            totenergy.append(AF.calckinenergy(frames) + AF.calcpotenergy(frames))    
            
            #start sampling the system 20 timesteps after the rescalling is done.
            if t > (C.rescalef * C.rescalen) + 20:
                plt.plot(O.histogram(frames))
                PDFpert.append(O.histogram(frames))
                
            KinEnergyList.append(AF.calckinenergy(frames))
            PotEnergyList.append(AF.calcpotenergy(frames))
                
            #rescale the velocities to the temperature we want to look at each rescalef frames.
            if t != 0 and t <= (C.rescalef * C.rescalen) and t%C.rescalef == 0:
               frames = rescaleeq(frames,T)
             
        '''
        Code to calculate and plot the observables.
        '''
        plt.xlabel('r [dimensionless units]') 
        plt.ylabel('g(r)')
        plt.show()
          
        #vector with the dimensionless time used for plotting against time
        tvec = np.arange(0, len(totenergy), 1) * C.dt
        
        plt.plot(tvec[:(nsteps - (C.rescalef * C.rescalen + 20 - 1))], O.Difussion(frames[(C.rescalef * C.rescalen) + 20:]), 'ro', markersize = 0.6)
        plt.xlabel('t [dimensionless units]')
        plt.ylabel(r'$\langle (x(t)-x(0))^2\rangle$ [dimensionless units]') 
        plt.savefig("Diff_T_" + str(T) + "_rho_" + str(rho) + ".pdf")
        plt.show()
        
        PDF = np.sum(PDFpert, axis = 0) / len(PDFpert)   
        #vector with the dimensionless distance used for plotting against distance
        rvec = np.arange(C.dr, 0.5 * C.L + C.dr, C.dr)
        
        plt.plot(rvec, PDF, 'bo', markersize = 0.6) 
        plt.xlabel('r [dimensionless units]')
        plt.ylabel('g(r)')
        plt.savefig("pair_dist_func_T_" + str(T) + "_rho_" + str(rho) + ".pdf")
        plt.show()
               
        CvList.append([O.SpecificHeat(KinEnergyList[(C.rescalef * C.rescalen) + 20:], T), bootStrapCv(np.array(KinEnergyList[(C.rescalef * C.rescalen) + 20:]), T, bitr)])     
        PList.append([O.Pressure(frames[(C.rescalef * C.rescalen) + 20:],T), bootStrapP(frames[(C.rescalef * C.rescalen) + 20:],T, bitr)])   
                
        plt.plot(tvec, KinEnergyList,'bo', markersize = 0.6)
        plt.plot(tvec, PotEnergyList,'ro', markersize = 0.6)
        plt.plot(tvec, totenergy,'go', markersize = 0.6)   
                
        plt.xlabel('t [dimensionless units]')
        plt.ylabel('E [dimensionless units]')         
        plt.savefig("energy_split_T_" + str(T) + "_rho_" + str(rho) + ".pdf")
        plt.show()
      
    print(CvList)
        
    Cv = [x[0] for x in CvList]
    CvErr = [x[1] for x in CvList]
    
    plt.errorbar(tempList, Cv, CvErr, fmt = 'ro')  
    plt.xlabel('T [dimensionless units]')
    plt.ylabel(r'$C_V$ [dimensionless units]')
    plt.savefig("specific_heat.pdf")    
    plt.show()
    
    print(PList)
    
    P = [x[0] for x in PList]
    PErr = [x[1] for x in PList]
    
    plt.errorbar(tempList, P, PErr, fmt = 'bo')  
    plt.xlabel('T [dimensionless units]')
    plt.ylabel('P [dimensionless units]')
    plt.savefig("pressure.pdf")    
    plt.show()  