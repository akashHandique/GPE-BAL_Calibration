# Surrogate-assisted Bayesian calibration


***
This repository contains the developed work for a Master's Thesis titled "Spatially Explicit, Depth-dependent and
Data-driven Roughness Calibration for
Numerical Modelling".
***

##Aim:
The main purpose of this repository is to exhibit the workflow of adapting Ferguson's approach in catagorised roughness 
zones of a hydrodynamic numerical model for the Lower Yuba River in Telemac-2D and calibrate the model using 
surrogate-assisted Bayesian inversion process.

##Folder description
###main
This folder contains the Telemac-2D simulation files for hydrodynamic model and 4 python scripts and input files necessary
to run the calibration process. [ppmodules](https://github.com/pprodano/pputils) consist of scripts used to amend the Telemac-2D results files.
Telemac-2D files: ```friction_calc.f```,```liquid_boundary_unsteady.liq```,```Q17.61.slf```,```rating_curve_2014_DPD.txt```,```roughness.tbl```,```yuba_geometry.slf```,```yuba_unsteady.cas```.
###results
Initial training data for the surrogate model are provided in this folder in formats that are acknowledged by the calibration algorithm.
`````.txt````` files are updated with further iterations of Bayesian active learning.
###Tif
All the rasters generated for each Bayesian active learning iterations are stored here.

###xyz
All the ```.xyz``` files utilised for the rasterisation process are stored in this folder.

###scripts
This folder contains the auxiliary functions for the Bayesian active learning and coupling of telemac-2D with the calibration process.
The same scripts are available in the main folder as well.

##Modules

```main_GPE_BAL_telemac.py```: This module executes the Bayesian calibration process.  

```auxiliary_functions_BAL.py```: Module for performing the Bayesian active learning calculation

```auxilliary_functions_telemac.py```: Module couple the Telemac-2D files with Bayesian calibration algorithm

```bea.py```: Module generates and post process the hydraulic variable rasters.

##Dependencies

To run the code, run the main_GPE_BAL_telemac.py file using the main folder as a current directory from a console/terminal in which Telemac-2D have already been compiled. It is not recommended to run the code from PyCharm as PyCharm uses a kind of additional virtual environment when it fires up its Terminal, and because Telemac has its own environment and APIs, those might be conflicting with PyCharm.

##References
[beatriznegreiros](https://github.com/beatriznegreiros/fuzzycorr)

[eduardoAcunaEspinoza](https://github.com/eduardoAcunaEspinoza/surrogated_assisted_bayesian_calibration)

