#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 15:06:24 2022

@author: nico mele

The objective is to obtain information on the angular position of objects identified 
using the ImageJ tool "Analyze Particles." A list is obtained for each acquired video 
frame. The main idea is to reconstruct the temporal tracking of these objects and then
fit functions according to their type of movement, comparing it with the original phase
of the magnetic field which is promoting the movement. From now on, the objects to be 
identified may be called Aggregates or particles, since it was originally implemented 
to identify aggregates of magnetic microparticles.

Analyzes video frames, identifying objects in order to establish a temporal 
correlation between each one. Each frame was previously analyzed with ImageJ to 
obtain information about each object. The original objective was to study the change 
in the angular position of magnetic microparticles and compare their motion relative 
to the magnetic field.

Names of some variables and functions were set in spanish. Sorry for the confusion. 

v 2.07
"""

import numpy as np;
# import scipy.optimize as opt;
import matplotlib.pyplot as plt
# import uncertainties as un
# from scipy.io import wavfile
import lmfit as lm
import glob 
# import pandas as pd
# from scipy.io import wavfile
from scipy import signal
import matplotlib.cm as cm
from tqdm import tqdm
import os 

pi = np.pi
# fname = ['frame000%2d'%k for k in range(30, 60)]

fname = glob.glob("*.csv")
fname.sort()
# fname = 'frame00030.png (red)'
ext = '.csv'

# frec_redo = 0.1 #Frecuencia en Hertz
# frec = 0.09998195582770436 #calculada a partir de aa_im_intensity.py o aa_load_int.py
# frate = 25 #frame rate
# f_ref = 0 #frame que se tomará como inicial para el MAIL LOOP
# f0 = 705 #número del frame del video completo del cual se comenzó a estudiar menos 1
# # audioname = 'frag5.wav'  # Archivo de audio extraido del video
# step = 0.1 #step del vector creado para el fiteo (respecto al num de frames)
# fase_campo = 1.16409 + pi # fase calculada a partir de aa_im_intensity.py o aa_load_int.py
# #El pi agregado en la fase de campo es debido a un error en la interpretación
# #Los cruces por cero asignados son cuando el campo se hace negativo (debe ser
# #la conexion del LED invertida)
# # fase_campo = 4.4083
# pop = [13, 4, 3, 1] #Colocar en orden descendente
# #Restricciones en curvas: 
len_pt = 300
dif_max = 90
dif_min = 60

curvas_rec = False  # Si True, guarda un .csv de cada curva de agregados

# params = lm.Parameters()
# params.add('A', value=1,  vary=0, min=0)
# params.add('f', value=frec/frate,  vary=0, min=0)
# params.add('phi0', value=fase_campo, vary=True)
# params.add('C', value=0,   vary=True)
# params.add('acdc', value=4.2/7.1, vary=1, min=0, max=10)
# params.add('ac', value=4.2, vary=1, min=0, max=10)

ua = 0.8 #Umbral porcentual del área rango [0:1]
uc = 20 #umbral de distancia al centroide (unidad: pixel)
nfig = 15 #Numero de figura preliminar


#%% CLASS DEF
#===========================================================================

# def seno(x, acdc, f, phi0, C):
#     return acdc*np.sin(x*f*2*np.pi+phi0) + C

def seno(x, acdc, f, phi0, C):
    return acdc*np.sin(x*f*2*np.pi+phi0) + C

def cuadrada(x, A, f, phi0, C):
    return A*signal.square(x*f*2*np.pi+phi0) + C

def tansen(x, A, f, phi0, C, acdc):
    return A*np.arctan(acdc*np.sin(x*f*2*np.pi+phi0))*180/pi + C

# gmodel = lm.Model(tansen, independent_vars='x', params=params)

def set_params(fase_campo, frec, frate, model='tansen'):
    ''' Setting and initializing parameters for lmfit implementation
    Arguments:
    fase_campo: float. Initial phase of magnetic field. Used to initialize lmfit params
    frec: float. frecuency of magnetic field. Used to initialize lmfit params
    frate: float. framerate of the camera used to record the videos. Initializing lmfit params
    model: 'seno' or 'tansen'. Function to be used to fit the aggregate movement.

    Return:
    params: lmfit class Parameters with a dictionary of lmfit.Parameter objects.
    gmodel: lmfit class which is a mathematical function used to describe and fit experimental data.
    '''
    params = lm.Parameters()
    if model=='tansen':
        params.add('A', value=1,  vary=0, min=0)
        params.add('f', value=frec/frate,  vary=0, min=0)
        params.add('phi0', value=fase_campo, vary=True)
        params.add('C', value=0,   vary=True)
        params.add('acdc', value=4.2/7.1, vary=1, min=0, max=10)
        gmodel = lm.Model(tansen, independent_vars='x', params=params)

    elif model=='seno':
        print('Modo seno')
        params.add('acdc', value=4.2/7.1, vary=1, min=0, max=90)
        params.add('f', value=frec/frate,  vary=0, min=0)
        params.add('phi0', value=fase_campo, vary=True)
        params.add('C', value=0,   vary=True)
        gmodel = lm.Model(seno, independent_vars='x', params=params)
    
    else: 
        #print('Elija modos entre "tansen" y "seno" ')
        raise Exception('The mode can only be string type "tansen" or "seno" ')
    return params, gmodel

class Aggregate():
    ''' Class to characterize each individual aggregate. 
    This class is used to compare the objects in different frames to identify the temporal correlation.
     '''
    def __init__(self, x, y, a, ang, ua=ua, uc=uc, rM=None, rm=None):
        ''' This data cames from the load data.
        Args:
        x, y = (float, float). Coordinates of the center of the aggregate
        a: float. Area of the aggregate.
        ang: float. Angle of the major radius of the ellipse (fitted by ImageJ) 
        ua: float. Area threshold (units related with input data lists). When comparing aggregates, the variance allowed to be considered to be the same aggregate.
        uc: float. Centroid threshold (units related with input data lists). Same as ua, but with the position of the centroid. 
        rM: float. Mayor radius of fitted ellipse.
        rm: float. Minor radius of fitted ellipse. 
         '''
        self.x   = x
        self.y   = y
        self.a   = a
        self.rM  = rM
        self.rm  = rm
        # self.ang = ang
        self.ang = self.angcorrection(ang)
        self.ua = ua #umbral de area
        self.uc = uc #umbral de centroide
        
    def compare(self, p_ext):
        '''
        Args:  
        p_ext: object Aggregate.  

        returns:
        igual: bool. Variable that indicates if the particles p_ext is the same (but in other frame) as self
        '''
        igual = True #same = True
        centroide = ((self.x - p_ext.x)**2 + (self.y-p_ext.y)**2) #comparison of centroids
        area = (self.a - p_ext.a)**2 #comparison of areas
        # print(np.sqrt(centroide))
        # print(np.sqrt(area))
        if centroide > self.uc**2:
            igual = False
        if area > (self.ua*self.a)**2:
            igual = False
        # print(igual)    
        return igual
    
    def isinlist(self, plist, printer=False):
        """Routine to check if there are matches between self and a list of Aggregate() 
        type objects using routine compare()
        Args:
        plist: list of all the Aggregate() of one frame.
        printer: bool. Warning message that more than one Aggregate() are similar to self. In this case, both are ignore.

        returns: 
        plist[j]: Aggregate(). This happens only when there is one coincidence between self and plist.
         """ 

        comp = [self.compare(k) for k in plist]
        if sum(comp) == 0:
            return False
        if sum(comp) > 1:
            if printer: print('Más de una partícula')
            return False
            # raise ValueError('Más de una partícula encontrada')
            # print('Encontro mas de una particula')
            # print('encontró %d particulas'%sum(comp))
            # return False
        j = comp.index(True)
        # print(plist[j])                                
        return plist[j]
    
    def angcorrection(self, ang):
        #Change of variables. The angle domain changes from [0, 180] to [-90, 90]
        if ang>90:
            self.ang = ang-180
        else: 
            self.ang = ang
        return self.ang
    
    
class AggregateTime():
    '''
    Class to save objects of class Aggregate that corresponds to the same particles
    in the video.  

    '''
    def __init__(self, f0, label=None, plist=None, flist=None, area=None,
                 rM=None, rm=None):
        '''
        Args:
        f0: initial frame of the analysis. User can choose the initial frame to avoid unwanted scenes
        label: string. Name of the particle/aggregate.
        plist: list of Aggregate() that correspond to the same particle/agg. It initializes as None. 
        flist: list of int or float. Time axis. Every time an Aggregate() is added to plist, the frame number to which it corresponds will be saved here. 

        =for debug=:
        area: list of float. Every time an Aggregate() is added to plist, the the area of the aggregate will be saved here.
        rM, rm: float, float. Mayor and minor radius of corresponding new aggregate, as in flist and area.
        '''
        if plist == None:
            plist = []
            flist = []
            area = []
            rM = []
            rm = []
        self.label = label
        self.plist = plist
        self.flist = flist
        self.area = area
        if len(self.flist)>3:
            self.pavrg = self.avrgpart()
        self.ang_t = [k.ang for k in self.plist]
        self.f0 = f0
        self.rM = rM
        self.rm = rm
        # if self.flist:
             # self.result = gmodel.fit(self.ang_t, x=self.flist, params=params)
        # else:            
        #     self.result = False
        # self.result = gmodel.fit(self.ang_t, x=self.flist, params=a_params)
            
    def addp(self, part, frame, area, rM, rm):
        ''' Tool to add a particle to AggregateTime() object
        part: Aggregate() object to be added.
        frame: float or int. Frame of the Aggregate() part.
        area, rM, rm: float, float, float. Information of the Aggregate() part. See __init__

        '''
        self.plist.append(part)
        self.flist.append(frame+self.f0)
        self.ang_t = [k.ang for k in self.plist]
        self.area.append(area)
        self.rM.append(rM)
        self.rm.append(rm)
        
        if len(self.flist)>3:
            # self.result = gmodel.fit(self.ang_t, x=self.flist, params=params)
            self.pavrg = self.avrgpart()           
    
    def avrgpart(self):
        area = np.mean([p.a for p in self.plist])
        x    = np.mean([p.x for p in self.plist])
        y    = np.mean([p.y for p in self.plist])
        return Aggregate(x,y,area, 0)
        
    def info(self):
        print('--------------------')
        print('numero de frames %d'%len(self.flist))
        if len(self.flist)>0:
            print('Average: (%d,%d), area= %d'%(self.pavrg.x,
                                                self.pavrg.y,
                                                self.pavrg.a))
    
    def fit(self, gmodel, params, fase_campo, i0=0, i1=None, tang=False):
        ''' Using LMFIT libraries, a fit is performed using the information of plist and flist
        gmodel: LMFIT Model(). See set_params()
        params: LMFIT Parameters(). See set_params()
        fase_campo: float. Initial phase of magnetic field. To calculate it, aa_im_intensity.py can be used.
        i0, i1: int, int. Frame range to be used, from i0 to i1.
        tang: bool. If true, apply numpy.tan() to the angle axis ang_t[i0,i1].
        '''
        
        if tang:
            y = np.tan(self.ang_t[i0:i1])
            self.result = gmodel.fit(y, x=self.flist[i0:i1], params=params)
            print(40*"=")
            print(self.label)
            print(lm.fit_report(self.result,show_correl=0))
        else:
            self.result = gmodel.fit(self.ang_t[i0:i1], x=self.flist[i0:i1], params=params)
            print(40*"=")
            print(self.label)
            print(lm.fit_report(self.result,show_correl=0))
           

#%% FRAME SCAN
#=================================================================

def frame_scan(fname, ua, uc, debug=0):
    '''
    Args:
    fname: frame txt name of the list of particles return by ImageJ. 
    This code assume that columns 1 = area; 2 and 3 = x, y (centroid of ellipse); -1 = angle
    Can be optimize for pandas.
    Returns
    f_ag: a list of aggregates (Aggregate() class) from the frame
    '''
    if debug:
        print(fname, ua, uc)
    AA = np.loadtxt(fname, delimiter=',', skiprows=1);
    f_ag = []
            
    for i in range(len(AA)):
        if np.size(AA) > 7:
            x = AA[i,2]
            y = AA[i,3]
            a = AA[i,1]
            rM = AA[i, 4]
            rm = AA[i, 5]
            ang = AA[i,-1]
        else:
            # print(k)            
            x = AA[2]
            y = AA[3]
            a = AA[1]
            rM = AA[4]
            rm = AA[5]
            ang = AA[-1]
            break
        p = Aggregate(x, y, a, ang, ua, uc, rM, rm)
        f_ag.append(p) #particulas de cada frame
    
    return f_ag      
   
# todas = []

# barra1 = tqdm(fname)
# for a in barra1:
#     barra1.set_description('Frame')
#     f_ag = frame_scan(a, ua, uc)
#     todas.append(f_ag)
    
    
#%% MAIN LOOP
#=================================================================
    

def main_loop(todos, frec, f0, len_pt=len_pt, dif_max=dif_max,
              dif_min=dif_min, f_ref=0, curvas_rec=False):
    ''' Iterates through all frames to identify aggregates over time and save them in a AggregateTime() class. 
    
    todos = list of lists of Agreggate() (list of lists f_ag - See frame_scan)
    frec = float. Frequency of the analyzed data in Hz
    f0 = int. Initial frame to be studied minus 1. If the first frame to analyze is the Nth frame, then enter N-1.

    f_ref = int. Frame of reference. The study will be held considering this frame as reference. By default, the initial frame of the studied range is used. 
    
    Cuve filtering: 
    len_pt = int. minimum length of a curve to avoid being eliminated. That is, the AggregateTime() class must have at least len_pt particles to be considered. 
    dif_max = maximum angle difference to filter curves. Same as len_pt but with the peak to peak amplitude.  
    dif_min = minimum angle difference to filter curves. Same as len_pt but with the peak to peak amplitude.
    curvas_rec: bool. Discontinued. Used now in ang_plot()
    
    Returns:
    todos_t: list. Nested list of all the AggregateTime() that passed the filtering. 
    '''
    # p0 = todas[0][0]
    p_tang = []; frame = []; p_tang2 = []; todos_t = []; pmean=[]
    # CC = [] ; BB=[]; #cent=[]
    
    barra2 = tqdm(todos[f_ref])
    #Iterates, within a reference frame f_ref, all the particles in it
    for n, p in enumerate(barra2): #Elijo el frame de referencia inicial
        barra2.set_description('Main Loop')
        # cent.append([p0.x, p0.y])
        pframes = []
        ag_label = "Agregado {npart:d} área(0)={area:5.0f}".format(npart=n+1,area=p.a)
        
        pmean = AggregateTime(f0, label=ag_label)
        # print(pmean.label)
        
        for fr in range(len(todos)): #Iterate through each frame looking for a match

            pnext = p.isinlist(todos[fr])
            if pnext:  #If there is a match in the fr frame, data is saved
                pframes.append(pnext)
                p = pnext
                p_tang.append(p.ang)
                frame.append(fr)            
                pmean.addp(pnext, fr, p.a, p.rM, p.rm)

        for i in p_tang:  #Changing angles from [0:180] to [-90:90]
            if i>90:
                p_tang2.append(i-180)
            elif i<-90:
                p_tang2.append(i+180)
            else:
                p_tang2.append(i)
        #Curve filtering
        #==============================================================
        if len(p_tang2) < len_pt: #by lenght
            # print(len(p_tang2))
            frame = []; 
            p_tang = []; p_tang2 = []
            continue     
        if (max(p_tang2)-min(p_tang2))>dif_max:  #by amplitude (peak to peak). Too much
            frame = []; 
            p_tang = []; p_tang2 = []
            continue
        if (max(p_tang2)-min(p_tang2))<dif_min:  #by amplitude (peak to peak). Too less
            frame = []; 
            p_tang = []; p_tang2 = []
            continue
        # if min(p0_tang2)<-40:
        #     frame = []; 
        #     p0_tang = []; p0_tang2 = []
        #     continue 
              
        todos_t.append(pmean)
        # pmean = []
        plt.figure(nfig)
        frame = np.array(frame)
        # x = frame/frate + 1
        # plt.plot(x, p0_tang2, 'o', alpha=0.5, label=n+1)
        frame = []; #p_t = []
        p_tang = []; p_tang2 = []
        # plt.legend()
        
        #SAVER
        #================
        # frec_s = str(int(frec*1000))

        # save_folder = '/home/nico/Documentos/Medidas Doctorado/Microscopio/COVID/Campo Alterno/Capa de Cloroformo/DC AC/Alto DC/dil 1-1000/LED B/'+frec_s+'/curvas/'
        
    return todos_t

# todos_t = main_loop(todos, frec_redo)


#%% PLOT
#=================================================================

def ang_plot(gmodel, params, todos_t, len_todos, frec, frec_redo, f0,
             fase_campo, frate=25, Brel=0.59, resta_C=True, curvas_rec=False, time_unit='fr',
             model='tansen'):
    '''Plot from an AggregateTime() class info 
    time_units= 's' to plot in sec or 'fr' to do in frames 

    todos_t = list of AggregateTime elements. Output of main_loop(). Elements that have been id in time and survived the filtering
    
    Curves are separeted in groups of 10, so it can be easily visible if there are some to be discarded


    ''' 
    
    if todos_t==[]:
        #print('No se encontraron agregados coincidentes')
        raise Exception('The inserted list todos_t is empty. Probably no matches were found.')
        return len_todos
    
    plt.figure(1);    plt.cla
    plt.figure(10);    plt.cla
    plt.figure(20);    plt.cla
    plt.figure(100);    plt.cla

    # plt.figure(30);    plt.clf
    A_pt = []; phi0_pt = []; C_pt = []; f_pt = []; desfa = []; acdc_pt = []
    phi0_pterr = []; acdc_pterr = []; area_pt = [] ; area_pterr = []
    rM_pt = [] ; rM_pterr = []
    
    for p, i in enumerate(todos_t):
        # if len(i.flist)<4:
        #     continue
    
        i.fit(gmodel, params, fase_campo, tang=0)
        y_samp = i.ang_t

        #Defining time and magnetic field
        
        x = np.arange(0,len_todos, 0.1)+f0
        yf = 10*np.sin(x*frec*2*pi/frate + fase_campo)
            
        fit_curve = gmodel.eval(i.result.params, x=x) #Curva ajustada
        
        if resta_C:
            C_pt.append(i.result.params['C'].value)
        else:
            C_pt.append(0)

        if time_unit == 's':
            fr = np.array(i.flist)/25
            x = (np.arange(0,len_todos, 0.1)+f0)/25
        elif time_unit == 'fr':
            fr = np.array(i.flist) #+f0
        else:
            raise('time_unit can only have input "s" and "fr".')

    
        if p>=40:
            #print('Mandale filtro, Rey, que hay muchos agregados')
            print('Too much aggregates.')
            continue
        
        
        #Todas las curvas juntas
        plt.figure(100)
        plt.plot(fr, y_samp, 'o', label=i.label)  # Plot

        #Graficos de a 10 curvas para ver ajustes
        #Lo hago de a 10 por el cm.tab20.colors que tiene 20 colores
        if p>=30:
            plt.figure(30)
            plt.plot(fr, y_samp-C_pt[-1], 'o', label=i.label,    
                      color=cm.tab20.colors[2*(p-30)])  # Plot
            plt.plot(x, fit_curve-C_pt[-1], '-', linewidth=3, 
                        color=cm.tab20.colors[2*(p-30)+1])  #Fit Plot
        
        if p>=20 and p<30:
            plt.figure(20)
            plt.plot(fr, y_samp-C_pt[-1], 'o', label=i.label,    
                      color=cm.tab20.colors[2*(p-20)])  # Plot
            plt.plot(x, fit_curve-C_pt[-1], '-', linewidth=3, 
                        color=cm.tab20.colors[2*(p-20)+1])  #Fit Plot
    
        if p>=10 and p<20:
            plt.figure(10)
            plt.plot(fr, y_samp-C_pt[-1], 'o', label=i.label,    
                      color=cm.tab20.colors[2*(p-10)])  # Plot
            plt.plot(x, fit_curve-C_pt[-1], '-', linewidth=3, 
                        color=cm.tab20.colors[2*(p-10)+1])  #Fit Plot
        
        if p<10:
            plt.figure(1)
            plt.plot(fr, y_samp-C_pt[-1], 'o', label=i.label,    
                      color=cm.tab20.colors[2*p])  # Plot
            plt.plot(x, fit_curve-C_pt[-1], '-', linewidth=3, 
                        color=cm.tab20.colors[2*p+1])  #Fit Plot
        
        #Plot of magnetic field
        Ac = np.arctan(Brel)/pi*180
        yf = Ac*np.sin(x*frec*2*pi/frate + fase_campo)

        #Save parameters
        if model=='tansen':
            A_pt.append(i.result.params['A'].value)
        C_pt.append(i.result.params['C'].value)
        phi0_pt.append(i.result.params['phi0'].value)
        f_pt.append(i.result.params['f'].value)
        acdc_pt.append(i.result.params['acdc'].value) 
        
        phi0_pterr.append(i.result.params['phi0'].stderr)
        acdc_pterr.append(i.result.params['acdc'].stderr)
        
        area_pt.append(np.mean(i.area))
        area_pterr.append(np.std(i.area))

        rM_pt.append(np.mean(i.rM))
        rM_pterr.append(np.std(i.rM))
        
    
    #Curve of mean values of all aggregates
    if model=='tansen':
        y_mean = tansen(x, 1, np.mean(f_pt), np.mean(phi0_pt), 0,
                        np.mean(acdc_pt))
    if model=='seno':
        y_mean = seno(x, np.mean(acdc_pt), np.mean(f_pt),
                      np.mean(phi0_pt), np.mean(C_pt))
        
    #
    plt.figure(1)
    plt.plot(x, yf, '-k')  
    # plt.plot(x, y_mean, '-r')                
    plt.legend()
    plt.grid()
    plt.xticks(fontsize=12)
    # plt.xlabel('Frame', fontsize=15)
    if time_unit == 's':
        plt.xlabel('Tiempo [s]', fontsize=15)        
    if time_unit == 'fr':    
        plt.xlabel('Tiempo [frame]', fontsize=15)

    plt.ylabel('Ángulo respecto a DC', fontsize=15)   
    
    # plt.figure(1)
    # plt.plot(x, yf, '.b')  
    # plt.plot(x, y_mean, '-r')                    
    # plt.legend()
    
    plt.figure(10)
    plt.plot(x, yf, '.-k')            
    plt.plot(x, y_mean, '-r')
    plt.legend()
    
    plt.figure(20)
    plt.plot(x, yf, '.-k')            
    plt.plot(x, y_mean, '-r')
    plt.legend()
    
    plt.figure(30)
    plt.plot(x, yf, '.-k')            
    plt.plot(x, y_mean, '-r')
    plt.legend()

    for i in range(len(phi0_pterr)):
        if phi0_pterr[i] == None: 
            phi0_pterr[i] = 0
            
    for i in range(len(acdc_pterr)):
        if acdc_pterr[i] == None: 
            acdc_pterr[i] = 0
        

    #Uncertainty in the phase
    fas_std = np.sqrt(np.std(phi0_pt)**2 + np.mean(phi0_pterr)**2)
    delta_phi = np.array(phi0_pt) - fase_campo
    #La segunda resta de pi es debido al menos de yf
    desfa = np.mod(delta_phi + pi, 2*pi) - pi 
    # print(desfa)
    desfa_m = np.mean(desfa)

#     #Reportes
    #========================================================
    print("<AC/DC>: {acdc:2.2f}".format(acdc=np.mean(acdc_pt)))
    print("<f>: {f:2.6f} // <f> = {f_r:2.6f}".format(f=np.mean(f_pt), f_r=np.mean(f_pt)*frate))     
    print("<phi0>: {phi0:2.2f}+/-{fstd:2.2f}".format(phi0=np.mean(phi0_pt), fstd=fas_std))
    print("<C>: {C:2.2f}".format(C=np.mean(C_pt)))
    print("<desfa>: {des:2.2f}".format(des=desfa_m))
    
    acdc=np.mean(acdc_pt)
    d_acdc = np.sqrt(np.mean(acdc_pterr)**2 + np.std(acdc_pt)**2)
    A = np.arctan(acdc)*180/pi
    dA = 1/(1+acdc**2)*d_acdc  #Propagación de errores para A = arctan(acdc)
    
    print("A = {A:1.5f} +/- {dA:1.5f}".format(A = A, dA = dA))
    print("Número de curvas: {n:3d}".format(n=len(todos_t)))
    frec_s = str(int(frec_redo*1000))
    
    if model=='tansen':
        A_pt = np.arctan(np.array(acdc_pt))*180/pi
    
    
    
    #Saving Variables to retrieve later. 
    #=========================================================  
    # path = '\\'.join(os.getcwd().split('\\')[:-1])
    path = '/'.join(os.getcwd().split('/')[:-1])
    print(path)

    # np.savez(path+'\\out_'+frec_s, phi=desfa, amp=A_pt, area=area_pt)
    # np.savez(path+'/out_'+frec_s+model, phi=desfa, amp=A_pt, area=area_pt, rM=rM_pt)
    
    #Saving curves
    # #========================================================= 
    # if curvas_rec:
    #     frec_s = str(frec_redo)
    #     for k in todos_t:
    #         np.savetxt(path+'\\curvas\\'+k.label+'_'+frec_s+'Hz.csv', np.transpose([k.flist, k.ang_t]),
    #                    fmt='%1.4f', header="Frame   Angle   "+k.label)

    if curvas_rec:
        frec_s = str(frec_redo)
        for k in todos_t:
            np.savetxt(path+'/curvas/'+k.label+'_'+frec_s+'Hz.csv', np.transpose([k.flist, k.ang_t]),
                       fmt='%1.4f', header="Frame   Angle   "+k.label)    
# ang_plot(todos_t)

#%% SIGLE PLOT TESTER
#==================================================================
# Verificador de curvas individuales para limpiar las no adecuadas
def ag_popper(todos_t, ag, pop=False):
    '''Function to check and remove curves that are not suitable at first glance. todos_t = list of AggregateTime with all aggregates found.
    todos_t = list of AggregateTime elements. Output of main_loop(). Elements that have been id in time and survived the filtering
    ag = List of the number of aggregates to check or remove.
    (from todos_t[n].label, where n is an integer between 0 and len(todos_t)).
    pop = bool. If you want to remove aggregates, this must be True.
    '''
    
    plt.figure(114)
    plt.cla()
    
    # ag = [0]
    # ag = np.linspace(20, 28, 9)    

    
    for k in ag:
        k_s = str(k)
        for agreg in todos_t:
            #Booleando que busca coincidencia con el Agregado #k_s
            # if any("Agregado "+k_s in agreg.label):    
            if "Agregado "+k_s+' ' in agreg.label: 
                x = agreg.flist
                y = agreg.ang_t
                plt.plot(x, y, 'o', label=agreg.label)
                plt.legend()

                if pop:        
                    print(k)
                    todos_t.remove(agreg)

# ag = [17, 84, 37, 55, 47]
# ag_popper(todos_t, ag, pop=0)
# ang_plot(todos_t, len(todos))

