Aggregate Motion README

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

amp_tools_2_XX.py is a module of tools to obtain the information about the motion of the aggregates. The information is taken from .csv files (ImageJ output). 

a_frec_scan.py is an example of running implementation. 

a_im_intensity.py returns the values of frequency and phase of the magnetic field, from the sync protocol (light one). This one have to be translated. 
