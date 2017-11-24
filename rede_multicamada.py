# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:31:19 2017

@author: andre.timm
"""
import numpy as np

# Função de ativação
def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

#a = sigmoid(50)

entradas = np.array([[0, 0], 
                     [0, 1], 
                     [1, 0], 
                     [1, 1]])
saidas = np.array([[0],
                   [1],
                   [1],
                   [0]])
    
#pesos0 = np.array([[-0.424, -0.740, -0.961],
#                 [0.358, -0.577, -0.469]])
#pesos1 = np.array([[-0.017], [-0.893], [0.148]])

pesos0 = 2 * np.random.random((2,3)) - 1
pesos1 = 2 * np.random.random((3,2)) - 1

epocas = 10000
momento = 1
taxaAprendizagem = 0.9

for j in range(epocas):
    camadaEntrada = entradas
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    erroCamadaSaida = saidas - camadaSaida
    mediaAbsoluta = np.mean(np.abs(erroCamadaSaida))
    print("Erro : " + str(mediaAbsoluta))
    
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroCamadaSaida * derivadaSaida
    
    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)
    
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)
    
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovo0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 =(pesos0 * momento) + (pesosNovo0 * taxaAprendizagem)
    
print("Taxa de acerto : " + str((1 - mediaAbsoluta) * 100) + "%")