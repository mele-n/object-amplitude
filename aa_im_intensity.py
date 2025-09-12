#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 23:51:35 2022

@author: nico

Agregado corrección por saltos que no ocurrieron
Agregado tren cada 10 saltos (teorico)
"""

import matplotlib.pyplot as plt
import numpy as np
import glob 
import lmfit as lm


#%% Vector de intensidad 
#===========================================================
frec = 1 #En Hz
frec_pulso = 0.2
frate = 25 #fps
pi = np.pi

# f01 = 125  #En caso de querer correr solo una parte de todas las imágenes 
#           #indicar frame de inicio (nombre del archivo -1)
# ff1 = 1305  #Lo mismo para el final
# f02 = 2660 #Inicio del segundo tren
# ff2 = 4052 #Fin del segundo tren

file = glob.glob("*.png")
file.sort()
# file2 = file[f01:ff1]+file[f02:ff2]
inten_r1 = [] # Mean de cada frame en canal rojo
inten_b1 = [] # Mean de cada frame en canal azul
inten_g1 = [] # Mean de cada frame en canal verde
inten  = [] # Mean de RGB
# frame1 = range(len(file[f01:ff1]))
# frame1 = np.array(frame1) + f01 
# frame2 = range(len(file[f02:ff2]))
# frame2 = np.array(frame2) + f02 

for k in file:
    BB = plt.imread(k)
    inten_r1.append(np.mean(BB[:,:,0]))
    inten_g1.append(np.mean(BB[:,:,1]))
    inten_b1.append(np.mean(BB[:,:,2]))
    inten.append(np.mean([inten_r1[-1], inten_g1[-1], inten_b1[-1]]))
    print(k)


#%% Linear fit
#==========================================
def linear(x, a, b):
    return a*x+b


prende = 0.0035 + min(inten_b1) #intensidad asignada para prender

inten_b1 = np.array(inten_b1)

A = inten_b1>prende
B = inten_b1<prende

j = np.where(A[1:]*B[:-1])[0] 

        
# j = np.delete(j, -1)
# j = np.delete(j, 1)
# j = np.delete(j, 0)
# Calculo del numero de eventos que pasan durante el lapso
#==================================================


#Fit de la recta completa 
#========================================
params = lm.Parameters()
params.add('a', value=1/frec*frate, vary=1)
params.add('b', value=0, vary=1)

modelo = lm.Model(linear, independent_vars='x', params=params)

#Define el eje de eventos segun la frecuencia de campo y de pulsos
xx = np.linspace(0,len(j)*frec/frec_pulso-frec/frec_pulso, len(j))

#Correcciones por saltos:
# for k in xx:
#     if k>17:
#         xx[k] += 1 

result = modelo.fit(j, x=xx, params=params)
print(lm.fit_report(result))
a = result.params['a'].value
b = result.params['b'].value

yy0 = 1/frec*frate*xx + 0
yy = a*xx+b

fase = - 2*pi*b/a  #Calculo de Gus: https://docs.google.com/document/d/1TF0Ypa66JRQ-ClWuT1f-AhAJcUnSwU9SfemTysnOwf4/edit

fase2 = np.mod(fase, 2*pi)

frame = range(len(inten_b1))
#Guardar datos
frec_s = str(frec)



#%% PLOTS

np.savetxt('jota_'+frec_s+'Hz.csv', np.transpose([xx, j, yy]), fmt='% d'
           , header="Ciclo   Frame_Salto  Y_Ajuste ")
np.savetxt('frame_int_'+frec_s+'.csv', np.transpose([frame, inten_r1, inten_g1, inten_b1]),
           fmt='%1.5f', header="Frame   Inten R  Inten G   Inten B ")

fi = 0 #En caso de haber cargado todas las fotos y querer analizar sólo un 
       #tramo indicar el inicio de tramo
ff = -1 #Idem final de tramo

frame = range(len(inten_b1))


# PLOT Intensidad
plt.figure(55)
plt.cla()

# plt.plot(frame[fi:ff], inten_r1[fi:ff] - min(inten_r1[fi:ff]), 'or', label='red')
# plt.plot(frame[fi:ff], inten_g1[fi:ff] - min(inten_g1[fi:ff]), 'og', label='green')
plt.plot(frame[fi:ff], inten_b1[fi:ff] - min(inten_b1[fi:ff]), '-o', label='blue')
# plt.plot(frame[fi:ff], inten_b1[fi:ff], 'o', label='Mean RGB')
# plt.plot(frame[j], inten_b1[j], 'o')
plt.legend()
plt.xlabel('Frame')
plt.ylabel('Color Intensity - min')

#Buscar en qué índices de frames ocurren los saltos

ysen = np.zeros(len(j))
plt.figure(66)
plt.plot(j, ysen, 'o')


    
plt.figure(28)
plt.cla()
plt.plot(xx, j, 'o')
plt.plot(xx, yy, '-r')
plt.plot(xx, yy0, '-k')


plt.text(0, max(j), 'Frecuencia: {f:1.5f}'.format(f=1/a*frate))
plt.text(0, max(j)-0.1*max(j), 'Fase: {fa:1.5f}'.format(fa=fase2))
plt.ylabel('Indice frame salto positivo')
plt.xlabel('#')
print('Frecuencia: {f:1.5f}'.format(f=1/a*frate))
print('Fase: {fa:1.5f}'.format(fa=fase2))


#%% TEST CADA DIEZ
#========================================================================

# j2 = j[::10]
# x2 = xx[::10]
# [a2, b2] = np.polyfit(x2, j2, 1)

# xx2 = np.linspace(0, len(j2), 100)
# yy2 = a2*xx2 + b2

# plt.figure(29)
# plt.cla()
# plt.plot(x2, j2, 'o')
# plt.plot(xx2, yy2, '-r')

# fase_10 = - 2*pi*b2/a2  #Calculo de Gus: https://docs.google.com/document/d/1TF0Ypa66JRQ-ClWuT1f-AhAJcUnSwU9SfemTysnOwf4/edit

# fase_10b = np.mod(fase_10, 2*pi)

# plt.text(0, max(j2), 'Frecuencia: {f:1.5f}'.format(f=1/a2*frate))
# plt.text(0, max(j2)-0.1*max(j2), 'Fase: {fa:1.5f}'.format(fa=fase_10b))
# plt.ylabel('Indice frame salto positivo')
# plt.xlabel('#')
# print('Frecuencia: {f:1.5f}'.format(f=1/a2*frate))
# print('Fase: {fa:1.5f}'.format(fa=fase_10b))
