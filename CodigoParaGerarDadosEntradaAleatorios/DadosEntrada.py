#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#import math
import numpy as np
from random import *
###-----------------------------------------------------------
## Construcao de banco de funcoes quadraticas:
## construcao de matrizes simetricas definida positiva -Karas

############### GERAR VETOR DE TAMANHO N ENTRE 0 E 1 ###########################
def gerarzeroum(valorn):
   tamanho = valorn
   resposta = [0.0] * tamanho
   for i in range(tamanho):
       resposta[i] = random()
   return resposta
################################################################################
# GERAR MATRIZ DIAGONAL EM QUE CADA ELEMENTO DA DIAGONAL VARIA ENTRE lambda1 e lambdan
n = int(input('Digite a ordem da matriz quadrada S desejada: '))
ve=gerarzeroum(n)
lambdamin = int(input('Digite o valor extremo esquerdo do intervalo: '))
lambdamax = int(input('Digite o valor extremo direito do intervalo: '))
print('-'*80)
print('O vetor de números aleatórios gerado, entre 0 e 1, de tamanho n é:\n')
print(' '*3, gerarzeroum(n),'\n')
print('O intervalo para autovalores vai de {} à {}.\n' .format(lambdamin, lambdamax))
################ GERANDO A MATRIZ DE FATO ######################################
y=[]
for x in ve:
  yi=lambdamin + ( (lambdamax - lambdamin)/ (max(ve)-min(ve)) )*(x-min(ve))
  y.append(yi)
yo=y.sort()                     #ordena o vetor y
diagy=np.diag(y)                #Constroi a matriz com y na diagonal
print('A matriz diagonal com elementos da diagonal variando de lambda1 à lambdan é:\n\n', diagy,'\n')
################################################################################
############## CONSTRUÇÃO DA MATRIZ ORTOGONAL Q ################################
M=100*np.random.rand(n,n)   #Obtenção de uma matriz aleatória nxn(entradas 0-100)
print('Matriz M:\n\n',M,'\n')
q,r=np.linalg.qr(M)         #Obtenção da fatoração QR de M
print('\nQ:\n\n', q,'\n')
print('\nR:\n\n', r,'\n')
################################################################################
####### MATRIZ SIMÉTRICA nxn COM AUTOVALORES ENTRE lambda1 E lambdan ###########
MS1=np.matmul( np.transpose(q), diagy)
MS=np.matmul(MS1,q)                     #MS=np.transpose(q)*diagy*q
print('Matriz Simetrica A:\n\n',MS,'\n')
################################################################################
### AUTOVALORES DA MATRIZ SIMÉTRICA GERADA #####################################
autovv = np.linalg.eigvals(MS)
print('Autovalores:\n\n',autovv, '\n')
###################GERANDO MATRIZ A ALEATÓRIA####################################
nlA = int(input('Digite o n° de linhas da matriz A desejada: '))
ncA = int(input('Digite o n° de colunas da matriz A desejada: '))
print('Linhas de A: ',nlA,' Colunas de A: ', ncA)
A=np.zeros((nlA, ncA))
for i in range(nlA):
    linhai = np.zeros(ncA)
    for j in range(ncA):
        A[i][j] = randint(-100, 100)
        #linhai[j]=A[i][j]
    #print('A linha {}:',i,linhai)
#print('A matriz A:\n', A)
###################GERANDO v, xk, lambdak e b ALEATÓRIOS####################################
v=[0]*len(MS)
for u in range(n):
        v[u]=randint(0, 100)
xk=[0]*len(MS)
for u in range(n):
        xk[u]=randint(0, 100)
lambdak=[0]*nlA
for u in range(nlA):
        lambdak[u]=randint(0, 100)
print('O lambdak: ', lambdak)
b=[0]*nlA
for u in range(nlA):
        b[u]=randint(0, 100)
c=randint(0, 100)
############ SALVANDO A MATRIZ E OS VETORES GERADOS ############################
with open('01-matriz-S.csv','a') as arquivo:
    np.savetxt('01-matriz-S.csv', MS, newline='\n')
with open('02-vetor-v.csv','a') as arquivo:
    np.savetxt('02-vetor-v.csv', v, newline='\n')
arquivo=open('03-constante-c.csv','w')
arquivo.write(str(c))
with open('04-matriz-A.csv','a') as arquivo:
    np.savetxt('04-matriz-A.csv', A, newline='\n')
with open('05-vetor-b.csv','a') as arquivo:
    np.savetxt('05-vetor-b.csv', b, newline='\n')
with open('06-vetor-xk.csv','a') as arquivo:
    np.savetxt('06-vetor-xk.csv', xk, newline='\n')
with open('07-vetor-lambdak.csv','a') as arquivo:
    np.savetxt('07-vetor-lambdak.csv', lambdak, newline='\n')
