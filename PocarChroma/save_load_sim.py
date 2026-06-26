import os
import h5py
import numpy as np
import pandas as pd 




def make_HDF5_file(
    file_path, 
    attributes:dict,


    tracks_shape,    # The shape of the tracks dataset, in standard numpy notation.

    hist_rows:int,
    # and the tallies columns
    hist_columns
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



    tallies_dtype = np.dtype([(name, 'i') for name in tallies_columns])



    # actually make the file
    with h5py.File(file_path, 'w') as f:

        # make the two datasets
        hist_ds = f.create_dataset(name='particle_history',
            shape=(tallies_rows,), 
            dtype=tallies_dtype
            )
        
        # set an attribute that keeps track of next writable row 

        hist_ds.attrs['next_writable'] = 0
        
        tracks_ds = f.create_dataset('tracks', tracks_shape, dtype='f')

        tracks_ds.attrs['next_writable'] = 0

        for key, value in attributes.items():
            f.attrs[key] = value
    
    print(f'Results file created at {file_path}')

    return

def particle_histories_write(
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


def tracks_read(file_path):
    '''
    Gets tracks from a given hdf5 file and returns them as a numpy array
    '''

    with h5py.File(file_path, 'r') as f:

        arr = f['tracks'][:]

        return arr



def tallies_read_to_df(file_path):
    '''
    Returns tallies as a pandas dataframe
    '''
    with h5py.File(file_path, 'r') as f:

        df = pd.DataFrame(f['tallies'][:])

        return df


def select_tracks(
    file_path,
    selection_criteria,
    invert_selection = False,
    return_type = 'indices'
    ):
    '''
    Replaces analysis_manager.preprocess_tracks
    Returns the indices of tracks that have a nonzero number for an interaction (i.e. have interacted in that way)
    :param selection_criteria: should be a string of the interaction of interest, or a function that takes a tallies object as an input and returns a 1D boolean array
    :type selection_criteria: str or function
    :param invert_selection: If true, takes the indicies which do not meet the selection criteria
    :type invert_selection: bool 
    :param return_type: 'indicies': return indices, 'tracks': return tracks, 'both': return both indicies and tracks (in that order)
    '''

    with h5py.File(file_path, 'r') as f:

        tracks = f['tracks']
        n_tracks = tracks.shape[1]

        # This assumes that the tracks are the first n rows in tallies

        tallies = f['tallies'][:n_tracks]

        # if selection criteria is callable, run it as a function to get a selection mask (1D array where)
        if callable(selection_criteria):
            sel_arr = selection_criteria(tallies)

            #if the returned array is not the same length as the number of tracks, throw an error
            if len(sel_arr) != n_tracks:
                raise ValueError("The array provided by selection_criteria does not correspond to the number of tracks")

        # if selection criteria is a string, select from those items in the appropriate column
        elif isinstance(selection_criteria, str):
            sel_arr = tallies[selection_criteria][:] > 0

        # If neither is true, throw a ValueError
        else:
            raise ValueError('selection_criteria must be a function or a string')

        # if specified, invert the selection
        if invert_selection:
            sel_arr = np.logical_not(sel_arr)

        # get the indices of tracks
        inds = np.where(sel_arr)

        if return_type == 'indices':
            return inds
        elif return_type == 'tracks':
            return np.take(tracks, inds, axis=1)
        elif return_type == 'both':
            return inds, np.take(tracks, inds, axis=1)
        else:
            raise ValueError('return_type must be "indices", "tracks" or "both"')


        

'''
====== BEGIN TESTING CODE ======



test_file_path = '/home/sam/Documents/pocar-chroma/test1.hdf5'

all_tracks = tracks_read(test_file_path)

def test_mask(tallies):
    sel_arr = tallies['RAYLEIGH_SCATTER'][:] > 0

    return sel_arr


inds, selected_tracks = select_tracks(test_file_path, test_mask, invert_selection=False, return_type='both')

print(selected_tracks)
print(inds)


'''
        




