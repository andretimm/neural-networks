# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 12:22:53 2017

@author: andre.timm
"""
import numpy as np

#entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
#saidas = np.array([0, 0, 0, 1])

#entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
#saidas = np.array([0, 1, 1, 1])
# XOR
entradas = np.array([[0,0], [0,1], [1,0], [1,1]])
saidas = np.array([0, 1, 1, 0])
pesos = np.array([0.0, 0.0])
taxaAprendizagem = 0.1

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

def calculaSaida(registro):
    s = registro.dot(pesos)
    return stepFunction(s)

def treinar():
    erroTotal = 1
    while (erroTotal != 0):
        erroTotal = 0
        for i in range(len(saidas)):
            saidaCalculada = calculaSaida(np.asarray(entradas[i]))
            erro = abs(saidas[i] - saidaCalculada)
            erroTotal += erro
            for j in range(len(pesos)):
                pesos[j] = pesos[j] + (taxaAprendizagem * entradas[i][j] * erro)
                print("Peso atualizado : " + str(pesos[j]))
        print("Total de erros: " + str(erroTotal))
        
treinar()
print("Rede treinada")