'''
In this example, this module is placed in the same folder as the .csv files of the aggregates.
The location of amp_tools_2 can by added using sys.path.append() or could be inside the current location.
'fase_campo' and 'frec' was obtained by the sync. module 'aa_im_intensity.py' (light sync)

Probando edits
'''

import sys
sys.path.append(r'folder')
import amp_tools_2_07 as amp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob
import lmfit as lm

pi = np.pi
brel = 0.59 #nominal relation of fields AC/DC
frec_redo = 1.0 #Frecuency in Hertz (rounded)
frec = 1.000363768643 #calculated from aa_im_intensity.py or aa_load_int.py
frate = 25 #frame rate of the video (for time convertion)
f_ref = 0 #frame que se tomará como inicial para el main_loop()
f0 = 135 #número del frame del video completo del cual se comenzó a estudiar menos 1
# audioname = 'frag5.wav'  # Archivo de audio extraido del video
step = 0.1 #step del vector creado para el fiteo (respecto al num de frames)
fase_campo = 0.035046262131 #+ pi # fase calculada a partir de aa_im_intensity.py o aa_load_int.py
#El pi agregado en la fase de campo es debido a un error en la interpretación
#Los cruces por cero asignados son cuando el campo se hace negativo (debe ser
#la conexion del LED invertida)
# fase_campo = 4.4083
# pop = [13, 4, 3, 1] #Colocar en orden descendente
#Restricciones en curvas: 
len_pt = 400
dif_max = 90
dif_min = 60

params = lm.Parameters()
params.add('A', value=1,  vary=0, min=0)
params.add('f', value=frec/frate,  vary=0, min=0)
params.add('phi0', value=fase_campo, vary=True)
params.add('C', value=0,   vary=True)
params.add('acdc', value=4.2/7.1, vary=1, min=0, max=10)
# params.add('ac', value=4.2, vary=1, min=0, max=10)

ua = 0.8 #Umbral porcentual del área rango [0:1]
uc = 20 #umbral de distancia al centroide (unidad: pixel)
nfig = 15 #Numero de figura preliminar

fname = glob.glob("*.csv")
fname.sort()

[params, gmodel] = amp.set_params(fase_campo, frec, frate)

todos = []
barra1 = tqdm(fname)
for a in barra1:
    barra1.set_description('Frame')
    f_ag = amp.frame_scan(a, ua=ua, uc=uc)
    todos.append(f_ag)

todos_t = amp.main_loop(todos, frec_redo, f0, len_pt=len_pt, dif_max=dif_max, dif_min=dif_min)


#%%
# amp.ang_plot(gmodel, params, todos_t, len(todos), frec, 
#              frec_redo, f0, fase_campo)

plt.close('all')
ag = [53]
amp.ag_popper(todos_t, ag, pop=1)
amp.ang_plot(gmodel, params, todos_t, len(todos), frec, 
             frec_redo, f0, fase_campo, curvas_rec=0, Brel=brel)

