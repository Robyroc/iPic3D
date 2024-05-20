import os
import numpy as np
import scipy
import h5py


def hdf5_to_txt(h5_input_file):

    h5_data = h5py.File(h5_input_file, "r")

    data = h5_data.get("topology").value

    print(data)

    # ###? Read data from hdf5 file
    # with h5py.File(h5_input_file, 'r') as h5_file:
    #     h5_data = h5_file['np_array'][()]

    ###* ------------------------------------------- *###

    ###? Get the name for the .txt file
    # strip_name = os.path.splitext(h5_input_file)[0]
    # txt_name = '{}.txt'.format(strip_name)

    ###? Save .txt file
    # np.savetxt(txt_name, h5_data)

hdf5_to_txt("../build/barrier/ref/proc5.hdf")