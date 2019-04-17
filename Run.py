# -*- coding: utf-8 -*-
'''
Module to run the simulation at a range of temperatures and a given density. 
'''
import MolecularDynamics as MD
import numpy as np

tempList = np.array([0.5])

rho = 1.2
M = 4

nsteps = 300
bitr = 10

MD.runsim(tempList, rho, M, bitr, nsteps)