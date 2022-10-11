#!/usr/bin/env python3
#-*- coding: utf-8 -*-

#import math
import numpy as np
#import math
# Para gráficos
import matplotlib.pyplot as plt
import time
# DEFININDO AS FUNÇÕES
# Definindo a quadrática
def f_quadratica(x, S , v, c):
    return 0.5*(x @ S @ x) +v @ x + c

# Definindo a restrição do tipo h(x)=0
def restricoes(x, A, b):
    return A @ x - b

# Gradiente da função quadrática num ponto (derivada de f(x)=S*xk+v)
def f_grad(x, S, v):
   return S @ x + v

# Gradiente das restrições num ponto é somente A (derivada de h(x)=A)
def h_gradT(A):
   return A.T

def h_grad(A):
   return A


# CARREGANDO OS DADOS DE ENTRADA
S=np.loadtxt('01-matriz-S.csv')
v=np.loadtxt('02-vetor-v.csv')
c=np.loadtxt('03-constante-c.csv')
A=np.loadtxt('04-matriz-A.csv')
b=np.loadtxt('05-vetor-b.csv')
xk=np.loadtxt('06-vetor-xk.csv')
lambdk=np.loadtxt('07-vetor-lambdak.csv')

# IMPRIMINDO OS DADOS DE ENTRADA
print('~'*26, 'ENTRADAS', '~'*26)
print('A matriz S é:\n', S)
print('O vetor v é:\n', v)
print('A constante c é:\n', c)
print('A matriz de restrições é:\n', A)
print('O vetor b é:\n', b)
print('O xk é:\n', xk)
print('O lambdak é:\n', lambdk)
fInicial=f_quadratica(xk,S,v,c)
print('O valor da função f em x^k é:\n',f_quadratica(xk,S,v,c))
print('O valor do gradiente de f em x^k é:\n',f_grad(xk,S,v))
print('~'*62)

# SALVANDO DADOS DE ENTRADA
with open('Resultados.csv', 'w') as arquivo:
    arquivo.write(str('~' * 80+'\n'))
    arquivo.write(str(' '*32 + 'DADOS DE ENTRADA' + ' '*32))
    arquivo.write(str('\n'+'~' * 80))
    arquivo.write('\nA matriz S é:\n'+str(S))
    arquivo.write('\nO vetor v é:\n'+str(v))
    arquivo.write('\nA constante c é:\n'+str(c))
    arquivo.write('\nA matriz A é:\n'+str(A))
    arquivo.write('\nO vetor b é:\n'+str(b))
    arquivo.write('\nO vetor xk inicial é:\n'+str(xk))
    arquivo.write('\nO vetor lambdak inicial é:\n'+str(lambdk))
#linhas, colunas = S.shape
#print("Linhas: ",linhas,' Colunas: ', colunas)
#print("Tipos - ",'S é: ',type(S),' v é: ',type(v),' c é: ',type(c),'\nA é: ',type(A),' b é: ',type(b),' xk é: ',type(xk),' lambdak: ', type(lambdk))

# DETERMINANDO O GRADIENTE E A HESSIANA \nabla^2 DA JUNÇÃO LAGRANGEANA EM xk
# Criando uma matriz nula
if np.size(A)> len(A):
    # pois existir mais de n elementos significa que l>1
    if len(A) > len(A[0]):
        M0 = np.zeros((len(A), len(A)))
    else:
        M0 = np.zeros((len(A), len(A)))
else:
    M0=np.zeros((1,1))
#linhasa, colunasa = M0.shape
#print("Linhasa: ", linhasa, ' Colunasa: ', colunasa)

# Definindo o vetor L1 como f'(xk) + (h'(x^k))^T̂ * lambdk, no caso da quadrática,
# por meio da junção de vetores
if np.size(A)> len(A):
    # pois existir mais de n elementos significa que l>1
    L1=np.block([S @ xk + v + (h_gradT(A)@lambdk), restricoes(xk,A,b)])
else:
    L1 = np.block([S @ xk + v + (h_gradT(A) * lambdk), restricoes(xk, A, b)])

# Definindo o vetor L2 como f''(x^k, \lambdk) no caso da quadrática por meio
# da junção de matrizes
if np.size(A)> len(A):
    L2_l1 = np.block([S, A.T])
    L2_l2 = np.block([A, M0])
    L2 = np.block([[L2_l1], [L2_l2]])
else:
    a = np.array([A])
    L2_l1=np.block([S, a.T])
    L2_l2=np.block([A, M0])
    L2=np.block([[L2_l1],[L2_l2]])

# DEFININDO O VETOR (xk,lambdk)^T POR MEIO DA JUNÇÃO DE VETORES
xklambdk=np.block([xk,lambdk])
xklambdkMatriz=np.array([xklambdk])
linhasxklambdak,colunasxklambdak=xklambdkMatriz.T.shape

# DETERMINANDO O VETOR xlambdk POR MEIO DA RESOLUÇÃO DO SISTEMA
#  L''(x^k,lambdk)@xlambdk=L''(x^k,lambdk)@xlambdk-L'(x^k,lambdk)
inicio = time.time()
xlambdk = np.linalg.solve(L2, L2 @ xklambdk.T -1*L1)
# Definindo o vetor solução x
x=[0]*len(xk)
if np.size(A)> len(A):
    for i in range(len(xlambdk)-len(A)):
        x[i]=xlambdk[i]
else:
    for i in range(len(xlambdk) - 1):
        x[i] = xlambdk[i]
x_conv = np.array(x)

# IMPRIMINDO OS RESULTADOS
print('~'*25, 'RESULTADOS', '~'*25)
print('O lambdk é:\n', lambdk)
print('O L2 é:\n', L2)
print('O vetor xklambdk é:\n',xklambdk)
print('O vetor xlambdk é:\n',xlambdk)
print('O vetor solução x (convertido) é:\n', x_conv)
print('O valor da função f em x é:\n',f_quadratica(x, S, v, c))
print('A função é:\n0.5*({}*x**2+1*{}*x*y+1*{}*y**2)+1*{}*x+1*{}*y+{}'.format(S[0][0],(S[0][1]+S[1][0]),S[1][1],v[0],v[1],c))
print('~'*62)
print('******COMENTE O CÓDIGO REFERENTE AOS GRÁFICOS EM PROBLEMAS COM\nMAIS DE DUAS VARIÁVEIS!!!\n')

# SALVANDO A FUNÇÃO
with open('Resultados.csv', 'a') as arquivo:
    arquivo.write(str('\n'+'~' * 80))
    arquivo.write('\nO valor inicial da função f em x é:\n' + str(fInicial) + '\n')
    arquivo.write(str('\n'+'~' * 80+'\n'))
    arquivo.write(str(' '*35 + 'RESULTADOS' + ' '*35))
    arquivo.write(str('\n'+'~' * 80))
    arquivo.write('\nO lambdk é:\n'+str(lambdk))
    arquivo.write('\nO L2 é:\n'+str(L2))
    arquivo.write('\nO vetor xklambdk é:\n'+str(xklambdk))
    arquivo.write('\nO vetor xlambdk é:\n'+str(xlambdk))
    arquivo.write('\nO vetor solução x é:\n'+str(x))
    arquivo.write('\nO valor da função f em x é:\n'+str(f_quadratica(x, S, v, c)))
    arquivo.write('\n'+'~'*80)
    arquivo.write(str('\n'+' '*27 + 'FACILITANDO OS RELATÓRIOS' + ' '*28))
    arquivo.write('\n'+'~'*80)
    arquivo.write(str('\nFUNÇÃO PARA O GEOGEBRA:'))
    arquivo.write(str('\n{}*x**2+1*{}*x*y+1*{}*y**2+1*{}*x+1*{}*y+{}'.format(S[0][0],S[0][1]+S[1][0],S[1][1],v[0],v[1],c)))
    arquivo.write(str('\n!! Se atente aos sinais quando for copiar!!!'))
    arquivo.write('\n' + '~' * 80)
    arquivo.write(str('\nFUNÇÃO EM LATEX'))
    arquivo.write(str('\n"Normal"'))
    arquivo.write(str('\n{}x_1^2+{}x_1x_2+{}x_2^2+{}x_1+{}x_2+{}'.format(S[0][0],S[0][1]+S[1][0],S[1][1],v[0],v[1],c)))
    arquivo.write(str('\nMatricial'))
    arquivo.write(str('\n {} & {}\ \ {} & {}'.format(S[0][0],S[0][1],S[1][0],S[1][1])))
    arquivo.write(str('\nSujeita a '))
    #arquivo.write(str('\n {}'.format(A[0][0])))
    arquivo.write('\n' + '~' * 80)
    arquivo.write('\nA matriz h_grad(A) é:\n'+str(h_grad(A)))
    arquivo.write('\nA matriz h_gradT(A) é:\n'+str(h_gradT(A)))
fim = time.time()
print('TEMPO: ', fim - inicio)
with open('Resultados.csv', 'a') as arquivo:
    arquivo.write(str('\n' + '~' * 80))
    arquivo.write('\nTEMPO GASTO:\n' + str(fim - inicio) + '(em segundos)')
# GRÁFICO
ax = plt.axes(projection='3d')
# Definindo o intervalo dos eixos e a quantia de pontos
xd,yd=np.mgrid[-5000:5000:15J,-5000:5000:15J]

# Definindo a função
Z=0.5*(S[0][0]*xd**2+2*S[0][1]*xd*yd+S[1][1]*yd**2)+v[0]*xd+v[1]*yd+c
# Plotando o gráfico
ax.plot_surface(xd,yd,Z,cmap='viridis')
# Plotando um ponto do gráfico
ax.scatter(x_conv[0],x_conv[1],f_quadratica(x_conv,S,v,c),color='red')
# Nomeando os eixos
ax.set_xlabel('Eixo x')
ax.set_ylabel('Eixo y')
ax.set_zlabel('Eixo z')

# Mostrando o gráfico
plt.show()
