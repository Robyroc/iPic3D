"""
Created on Tue May 7 12:50 2024

@author: Pranab JD
"""

import os, sys
import glob, h5py
import numpy as np

from datetime import datetime

startTime = datetime.now()

###* =================================================================== *###

dir_ref = "./barrier/ref/"
dir_data = "./barrier/data/"

num_hdf_files = len([name for name in os.listdir(dir_ref) if os.path.isfile(os.path.join(dir_data, name))])

Ex_error = np.zeros(num_hdf_files); Ey_error = np.zeros(num_hdf_files); Ez_error = np.zeros(num_hdf_files)
Bx_error = np.zeros(num_hdf_files); By_error = np.zeros(num_hdf_files); Bz_error = np.zeros(num_hdf_files)

###? Iterate through all hdf files (one for each processor)
for ii in range(num_hdf_files):

    ###? Reference dataset
    for file in glob.glob(dir_ref + "/proc" + str(ii) + ".hdf"):

        data = h5py.File(file, "r")
        field = data.get("fields")
        
        Bx = field.get("Bx"); By = field.get("By"); Bz = field.get("Bz")
        Ex = field.get("Ex"); Ey = field.get("Ey"); Ez = field.get("Ez")

        Bx_ref = np.array(Bx.get("cycle_10")); By_ref = np.array(By.get("cycle_10")); Bz_ref = np.array(Bz.get("cycle_10"))
        Ex_ref = np.array(Ex.get("cycle_10")); Ey_ref = np.array(Ey.get("cycle_10")); Ez_ref = np.array(Ez.get("cycle_10"))

    ###? Modified dataset
    for file in glob.glob(dir_data + "/proc" + str(ii) + ".hdf"):

        data = h5py.File(file, "r")
        field = data.get("fields")
        
        Bx = field.get("Bx"); By = field.get("By"); Bz = field.get("Bz")
        Ex = field.get("Ex"); Ey = field.get("Ey"); Ez = field.get("Ez")

        Bx_data = np.array(Bx.get("cycle_10")); By_data = np.array(By.get("cycle_10")); Bz_data = np.array(Bz.get("cycle_10"))
        Ex_data = np.array(Ex.get("cycle_10")); Ey_data = np.array(Ey.get("cycle_10")); Ez_data = np.array(Ez.get("cycle_10"))

    ###? Compute error
    Bx_diff = np.mean(abs(Bx_ref - Bx_data))/np.linalg.norm(Bx_ref)
    By_diff = np.mean(abs(By_ref - By_data))/np.linalg.norm(By_ref)
    Bz_diff = np.mean(abs(Bz_ref - Bz_data))/np.linalg.norm(Bz_ref)

    Ex_diff = np.mean(abs(Ex_ref - Ex_data))/np.linalg.norm(Ex_ref)
    Ey_diff = np.mean(abs(Ey_ref - Ey_data))/np.linalg.norm(Ey_ref)
    Ez_diff = np.mean(abs(Ez_ref - Ez_data))/np.linalg.norm(Ez_ref)

    Bx_error[ii] = Bx_diff; By_error[ii] = By_diff; Bz_error[ii] = Bz_diff
    Ex_error[ii] = Ex_diff; Ey_error[ii] = Ey_diff; Ez_error[ii] = Ez_diff


###* =================================================================== *###

print("Bx error: ", np.mean(Bx_error))
print("By error: ", np.mean(By_error))
print("Bz error: ", np.mean(Bz_error))
print("Ex error: ", np.mean(Ex_error))
print("Ey error: ", np.mean(Ey_error))
print("Ez error: ", np.mean(Ez_error))


print()
print("Complete .....", "Time Elapsed = ", datetime.now() - startTime)