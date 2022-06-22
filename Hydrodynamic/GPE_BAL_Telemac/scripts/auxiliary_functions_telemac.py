'''
Auxiliary functions to couple the Surrogate-Assisted Bayesian inversion technique with Telemac. These functions are
specific of the parameters that wanted to be changed at the time, but they can be used as a base on how to modify
Telemac's input and output files

Contact: eduae94@gmail.com
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


def update_steering_file(prior_distribution, parameters_name, friction_name,
                         telemac_name, result_name_telemac, n_simulation):
    """
    Function amends calibration parameters in the friction_calc subroutine and
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

    Returns updated friction_user file with new calibration parameter values
    -------
    updates the friction_calc subroutine
    """

    # Update A
    updated_values = np.round(prior_distribution[0], decimals=1)
    updated_string = create_string(parameters_name[0], updated_values)
    rewrite_parameter_file(parameters_name[0], updated_string, friction_name)

    # Update B
    updated_values = np.round( prior_distribution[1], decimals=1)
    updated_string = create_string(parameters_name[1], updated_values)
    rewrite_parameter_file(parameters_name[1], updated_string, friction_name)

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
    Function updates the parameter name and value to the telemac file
    Parameters
    ----------
    param_name :  String for the parameter name
    updated_string : Updated string for the telemac file
    path : Path of the telemac file

    Returns
    -------
    Updated telemac file with updated string
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
    Function updates the parameter name and value to the friction_user subroutine file
    Parameters
    ----------
    param_name :  String
    Name of the parameter name
    updated_string : String
    String to be updated in the friction_calc subroutine file
    path : String
    Path of the friction_calc subroutine file

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




def get_variable_value(file_name, calibration_variable, specific_nodes=None, save_name=""):
    """
    Function extracts the calibrated variable with respect to nodes and saves the data in a file
    Parameters
    ----------
    file_name : String
    Result file name to extract the calibrated variable
    calibration_variable : String
    Variable of interest to be extracted corresponding to nodes
    specific_nodes : Numpy array
    Specific nodes to be indexed
    save_name : String
    Name of the saved file

    Returns
    -------
    Numpy array of results in the specific nodes.
    """
    # Read the SELEFIN file
    slf = ppSELAFIN(file_name)
    slf.readHeader()
    slf.readTimes()

    # Get the printout times
    times = slf.getTimes()

    # Read the variables names
    variables_names = slf.getVarNames()
    # Removed unnecessary spaces from variables_names
    variables_names = [v.strip() for v in variables_names]
    # Get the position of the value of interest
    index_variable_interest = variables_names.index(calibration_variable)

    # Read the variables values in the last time step
    slf.readVariables(len(times) - 1)

    # Get the values (for each node) for the variable of interest in the last time step
    modelled_results = slf.getVarValues()[index_variable_interest, :]
    format_modelled_results = np.zeros((len(modelled_results), 2))
    format_modelled_results[:, 0] = np.arange(1, len(modelled_results) + 1, 1)
    format_modelled_results[:, 1] = modelled_results

    # Get specific values of the model results associated in certain nodes number, in case the user want to use just
    # some nodes for the comparison. This part only runs if the user specify the parameter specific_nodes. Otherwise
    # this part is ommited and all the nodes of the model mesh are returned
    if specific_nodes is not None:
        format_modelled_results = format_modelled_results[specific_nodes[:, 0].astype(int) - 1, :]

    if len(save_name) != 0:
        np.savetxt(save_name, format_modelled_results, delimiter="	", fmt=['%1.0f', '%1.3f'])

    # Return the value of the variable of interest
    return format_modelled_results




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
