import os
import h5py
import numpy as np
import pandas as pd 




def make_HDF5_file(
    file_path, 
    attributes:dict,


    tracks_shape,    # The shape of the tracks dataset, in standard numpy notation.

    tallies_rows:int,
    # and the tallies columns
    tallies_columns
    ):
    '''
    Makes an HDF5 file with the desired header information

    You can pass anything under 64kb into attributes and it will be saved along with the file. 
    This includes things such as a notes section
    To read attributes, use

    f.attrs.get[<attribute_name>].

    It also creates two datasets, one called tallies and one called tracks

    Note, specifying dataset shape will ensure that file creation goes smoothly.
    The tracks dataset has the shape (<number of steps> + 1, <number of tracks to be saved>, 3)
    You should also specify the total number of rows (photons) in the tallies arrays
    '''
    # I could in theory make it so the HDF5 file is configured to be dynamic, but that would take extra work that doesn't seem worth it right now

    save_dir = file_path.rsplit('/', 1)[0]
    if os.path.isdir(save_dir):
        pass
    else:
        os.makedirs(save_dir, exist_ok=True)


    # Convert tallies_columns into list if supplied as dict
    if isinstance(tallies_columns, dict):
        tallies_columns = list(tallies_columns.keys())
    
    # make the tallies column names into a structured dtype with columns as bools

    tallies_columns.append('track_number')
    
    tallies_dtype = np.dtype([(name, 'i') for name in tallies_columns])



    # actually make the file
    with h5py.File(file_path, 'w') as f:

        # make the two datasets
        tallies_ds = f.create_dataset(name='tallies',
            shape=(tallies_rows,), 
            dtype=tallies_dtype
            )
        
        # set an attribute that keeps track of next writable row 

        tallies_ds.attrs['next_writable'] = 0
        
        tracks_ds = f.create_dataset('tracks', tracks_shape, dtype='f')

        tracks_ds.attrs['next_writable'] = 0

        for key, value in attributes.items():
            f.attrs[key] = value
    
    print(f'Results file created at {file_path}')

    return

def tallies_write(
    file_path:str,
    tallies_dict:dict,
):
    '''
    :param file_path: The path to the previously created hdf5 file
    :type file_path: str
    :param tallies_dict: A dictionary of numpy arrays
    :type tallies_dict: dict
    '''
    


    with h5py.File(file_path, 'r+') as f:
        ds = f['tallies']
        next_row = ds.attrs['next_writable']


        # Get the end row by getting the length of the first array in the dict
        end_row = next_row + len(next(iter(tallies_dict.values())))

        
        for key, value in tallies_dict.items():



            ds[key, next_row:end_row]= value

        
        ds.attrs['next_writable'] = end_row
        
    return




def tracks_write(
    file_path,
    tracks_arr
):
    '''
    Writes a tracks array to a preexisting hdf5 file that was created by make_HDF5_file
    '''
    with h5py.File(file_path, 'r+') as f:
        ds = f['tracks']
        next_row = ds.attrs['next_writable']
        end_row = next_row + tracks_arr.shape[1]

        ds[:, next_row:end_row, :] = tracks_arr
        ds.attrs['next_writable'] = end_row
    
    return







