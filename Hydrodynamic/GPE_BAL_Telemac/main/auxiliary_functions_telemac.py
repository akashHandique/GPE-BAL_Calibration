'''
Auxiliary functions to couple the Surrogate-Assisted Bayesian inversion technique with Telemac. These functions are
specific of the parameters that wanted to be changed at the time, but they can be used as a base on how to modify
Telemac's input and output files

Contact: iamakash0123@gmail.com
'''

# Import libraries
import sys, os
import subprocess
import shutil
import numpy as np
import math
from datetime import datetime
import init
from ppmodules.selafin_io_pp import *
import pandas as pd
import rasterio as rio
from osgeo import gdal
from osgeo import ogr
import bea as pp

def update_steering_file(prior_distribution, parameters_name, friction_name,
                         telemac_name, result_name_telemac, n_simulation):
    """
    Function amends calibration parameters in the friction_calc.f subroutine and
    updating new strings to be included in the updated file by calling two functions.
    Parameters
    ----------
    prior_distribution : Numpy array
         New calibration values
    parameters_name : String
         Name of the parameters to be changed
    friction_name : String
         Name of the file
    telemac_name : string
         Name of the telemac file
    result_name_telemac : String
         Name of the telemac result
    n_simulation : Integar
         Number of simulation

    Returns updated file with new calibration parameter values
    -------
    updates the friction_calc.f subroutine
    """

    # Update A
    updated_values = np.round(prior_distribution[0], decimals=1)
    updated_string = create_string(parameters_name[0], updated_values)
    rewrite_parameter_file(parameters_name[0], updated_string, friction_name)

    # Update B
    updated_values = np.round( prior_distribution[1], decimals=1)
    updated_string = create_string(parameters_name[1], updated_values)
    rewrite_parameter_file(parameters_name[1], updated_string, friction_name)

    updated_values = np.round( prior_distribution[2], decimals=1)
    updated_string = create_string(parameters_name[2], updated_values)
    rewrite_parameter_file(parameters_name[2], updated_string, friction_name)

    updated_values = np.round( prior_distribution[3], decimals=1)
    updated_string = create_string(parameters_name[3], updated_values)
    rewrite_parameter_file(parameters_name[3], updated_string, friction_name)

    updated_values = np.round( prior_distribution[4], decimals=1)
    updated_string = create_string(parameters_name[4], updated_values)
    rewrite_parameter_file(parameters_name[4], updated_string, friction_name)

    updated_values = np.round( prior_distribution[5], decimals=1)
    updated_string = create_string(parameters_name[5], updated_values)
    rewrite_parameter_file(parameters_name[5], updated_string, friction_name)

    updated_values = np.round( prior_distribution[6], decimals=1)
    updated_string = create_string(parameters_name[6], updated_values)
    rewrite_parameter_file(parameters_name[6], updated_string, friction_name)

    updated_values = np.round( prior_distribution[7], decimals=1)
    updated_string = create_string(parameters_name[7], updated_values)
    rewrite_parameter_file(parameters_name[7], updated_string, friction_name)

    # Update result file name telemac
    updated_string = "RESULTS FILE"+" : "+ result_name_telemac + str(n_simulation) + ".slf"
    rewrite_results_file("RESULTS FILE", updated_string, telemac_name)


def create_string(param_name, values):
    """
    Function generate a new standard string complying with telemac subroutine format
    Parameters
    ----------
    param_name : Name of the parameters to be changed
    values :  values to be added with the string

    Returns
    -------
    String with parameter name and value
    """
    updated_string = param_name + " = " + str(values) + "D0"
    return updated_string


def rewrite_results_file(param_name, updated_string, path):
    """
    Function updates the parameter name and value to the telemac .cas file
    Parameters
    ----------
    param_name :  String for the parameter name
    updated_string : Updated string for the telemac .cas file
    path : Path of the telemac .cas file

    Returns
    -------
    Updated telemac .cas file with updated string
    """
    variable_interest = param_name

    # Open the steering file with read permission and save a temporary copy
    cas_file = open(path, "r")
    read_steering = cas_file.readlines()

    # If the updated_string have more 72 characters, then it divides it in two

    # Preprocess the steering file. If in a previous case, a line had more than 72 characters then it was split in 2,
    # so this loop clean all the lines that start with a number
    temp = []
    for i, line in enumerate(read_steering):
        temp.append(line)

    # Loop through all the lines of the temp file, until it finds the line with the parameter we are interested in,
    # and substitute it with the new formatted line
    for i, line in enumerate(temp):
        line_value = line.split(":")[0].rstrip().lstrip()
        if line_value == variable_interest:
            temp[i] =  updated_string + "\n"

    # Rewrite and close the steering file
    friction_file = open(path, "w")
    friction_file.writelines(temp)
    friction_file.close()

def rewrite_parameter_file(param_name, updated_string, path):
    """
    Function updates the parameter name and value to the friction_calc.f subroutine file
    Parameters
    ----------
    param_name :  String
    Name of the parameter name
    updated_string : String
    String to be updated in the friction_calc.f subroutine file
    path : String
    Path of the friction_calc.f subroutine file

    Returns
    -------
    Updated telemac file with updated string

    """

    # Save the variable of interest without unwanted spaces
    variable_interest = param_name

    # Open the steering file with read permission and save a temporary copy
    friction_file = open(path, "r")
    read_steering = friction_file.readlines()

    # If the updated_string have more 72 characters, then it divides it in two

    # Preprocess the steering file. If in a previous case, a line had more than 72 characters then it was split in 2,
    # so this loop clean all the lines that start with a number
    temp = []
    for i, line in enumerate(read_steering):

            temp.append(line)

    # Loop through all the lines of the temp file, until it finds the line with the parameter we are interested in,
    # and substitute it with the new formatted line
    for i, line in enumerate(temp):
        line_value = line.split("=")[0].rstrip().lstrip()
        if line_value == variable_interest:
            temp[i] = " "+" "+" "+" "+" "+" "+ updated_string + "\n"

    # Rewrite and close the steering file
    friction_file = open(path, "w")
    friction_file.writelines(temp)
    friction_file.close()



def run_telemac(telemac_file_name, number_processors):
    """
    Function runs the telemac file provided
    Parameters
    ----------
    telemac_file_name : String
    Name of the telemac file
    number_processors : Intergar
    number of cores to be added for the simulation

    Returns
    -------
    print statements showing telemac simulation update
    """
    start_time = datetime.now()
    # Run telemac
    bash_cmd = "telemac2d.py " + telemac_file_name + " --ncsize="+number_processors
    process = subprocess.Popen(bash_cmd .split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print("Telemac simulation finished")
    print("Simulation time= " + str(datetime.now() - start_time))




def get_variable_value(file_name,x_mesh, y_mesh, save_name_xyz="", save_raster="",
                       save_name = ""):

    """
    Function extracts the velocity and water depth variables from the .slf file
    and arranged in proper format for further processing.
    Parameters
    ----------
    file_name : String
    Result file name to extract the calibrated variable
    x_mesh : Numpy array
    Latitude of the calibration points
    y_mesh : Numpy array
    Longitude of the calibration points
    save_name_xyz: String
    Path of the file to save the .xyz generated file
    save_raster: String
    Path of the file to save the generated raster from the .xyz file data
    save_name: String
    Path of the file .txt file to save the extracted hydraulic variable data
    Returns
    -------
    Numpy array with latitude, longitude, velocity and water depth data.
    """

########################################################################################################################
    slf = ppSELAFIN(file_name)
    slf.readHeader()
    slf.readTimes()
    # Get the printout times
    times = slf.getTimes()
    # Read the variables names
    variables_names = slf.getVarNames()
    # Removed unnecessary spaces from variables_names
    variables_names = [v.strip() for v in variables_names]
    variables = ["WATER DEPTH", "SCALAR VELOCITY"]
    for m in variables:
        # Get the position of the value of interest
        index_variable_interest = variables_names.index(m)
        # Read the variables values in the last time step
        slf.readVariables(len(times) - 1)
        # Get the values (for each node) for the variable of interest in the last time step
        modelled_results = slf.getVarValues()[index_variable_interest, :]
        x = slf.getMeshX()
        y = slf.getMeshY()
        B = np.array([1, 2, 3])
        for i in range(0, len(x)):
            if i >= 1:
                A = np.array([x[i], y[i], modelled_results[i]])
                B = np.vstack((B, A))
            else:
                A = np.array([x[i], y[i], modelled_results[i]])
                B = A
        if m == "SCALAR VELOCITY":

            save_name_xyz = save_name_xyz.replace("Waterdepth.xyz","Velocity.xyz")
        if m == "WATER DEPTH":

            save_name_xyz = save_name_xyz + "Waterdepth.xyz"
        if len(save_name_xyz) != 0:
            np.savetxt(save_name_xyz, B, delimiter=",", fmt=['%1.6f', '%1.6f', '%1.8f'], header="x,y,variable")
        #############################################################################raster generation
        raster_create(interpol_method="cubic", raster_out_save=save_raster + m + ".tif", save_xyz=save_name_xyz)
        d = rio.open(save_raster + m + ".tif")
        row, col = d.index(x_mesh, y_mesh)
        p = d.read(1)[row, col]
        if m == "SCALAR VELOCITY":
            n = p.reshape(-1, 1)
        if m == "WATER DEPTH":
            j = p.reshape(-1, 1)


    model_result = np.hstack((y_mesh, n))
    main_result = np.hstack((x_mesh, model_result))
    main_results = np.hstack((main_result, j))
    if len(save_name) != 0:
        np.savetxt(save_name, main_results, delimiter=" ", fmt=['%1.6f', '%1.6f', '%1.8f','%1.8f'])
    return main_results



def raster_create(interpol_method, raster_out_save="", save_xyz=""):
    """
    Function creates rasters from .xyz file data
    Parameters
    ----------
    interpol_method: String
    name of the interpolation method for rasterisation
    raster_out_save: String
    path of the file to save the generated raster
    save_xyz: String
    Path of the file .xyz file

    Returns
    Rasters and saves them in defined path.
    -------

    """
    map_file = pp.PreProFuzzy(pd.read_csv(save_xyz, skip_blank_lines=True), attribute="variable", crs='EPSG:4326',
                              nodatavalue=-9999, res=1)
    array_ = map_file.norm_array(method=interpol_method)
    map_file.array2raster(array_, raster_out_save, save_ascii=False)




def append_new_line(file_name, text_to_append):
    """
    Function opens a file and adds new string to the file
    Parameters
    ----------
    file_name : String
    Name of the file to be updated
    text_to_append : String
    String to be added to the file


    Returns
    -------
    the modified file with new lines
    """
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)
