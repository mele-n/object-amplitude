Aggregate Motion README

The objective is to obtain information of the angular position of objects previously identified using the ImageJ tool "Analyze Particles." A list is obtained for each acquired video frame. The main idea is to reconstruct the temporal tracking of these objects (over the successive frames) and then fit functions according to their type of movement, comparing it with the original phase of the magnetic field which is promoting the movement. From now on, the objects to be identified may be called Aggregates or particles, since it was originally implemented to identify aggregates of magnetic microparticles.

Names of some variables and functions were set in spanish. Sorry for the confusion. 

amp_tools_2_XX.py is a module of tools to obtain the information about the motion of the aggregates. The information is taken from .csv files (ImageJ output). 

a_frec_scan.py is an example of running implementation. Take some information from a_im_intensity.py

a_im_intensity.py returns the values of frequency and phase of the magnetic field (MF), from the sync protocol (light one). Shortly, the video was iluminated with a LED everytime the MF was zero. Knowing that the MF has a sine behavior, the main intensity of each frame is calculated in this script, and then, assuming a consecutive order of ligh-ups, the phase and frecuency are fit. This one have to be translated. 
