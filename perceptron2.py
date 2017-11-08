# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 13:22:58 2017

@author: andre.timm
"""
import numpy as np

entradas = np.array([1, 7, 5])
pesos = np.array([0.8, 0.1, 0])

def soma(e, p):
   return e.dot(p)
#dot product / produto escalar (realiza a funcao soma)
        
s = soma(entradas, pesos)

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0
    
r = stepFunction(s)
