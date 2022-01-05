import numpy as np
import h5py
import glob
import os
import datetime
import time
import pathlib


def generate_seeds(toy_label):
    """Given a toy id, this function generate a seed for reference and toy data

    Args:
        toy_label (int): Toy id

    Returns:
        (int, int): reference and toy seed
    """    
    seed_factor = datetime.datetime.now().microsecond + datetime.datetime.now().second+datetime.datetime.now().minute
    
    ref_seed = int(2**32-1) - ((int(toy_label) + 1) * seed_factor)
    toy_seed = ((int(toy_label) + 1) * seed_factor) % int(2**32-1)
        
    return ref_seed, toy_seed






def normalize_features(reference, data):
    """
    Normalize features (higgs normalization)

    Parameters
    ----------
    reference : np.ndarray
        Numpy array reference sample
    data : np.ndarray
        Numpy array data sample

    Returns
    -------
    ref_norm : np.ndarray
        Normalized Numpy array reference sample
    data_norm : np.ndarray
        Normalized Numpy array data sample

    """
    
    X_norm = normalize(np.vstack((reference, data)))
    
    ref_size = reference.shape[0]
    
    ref_norm = X_norm[:ref_size, :]
    data_norm = X_norm[ref_size:, :]
    
    return ref_norm, data_norm
    


def create_labels(ref_sample_obj, data_sample_obj, ml_model_type):
    """Create labels for reference and data

    Args:
        ref_sample_obj (np.ndarray): reference inputs
        data_sample_obj (np.ndarray): data inputs
        ml_model_type (str): Model used (it can be 'falkon_model' for Falkon or 'logfalkon_model' for LogFalkon)

    Returns:
        np.ndarray: outputs for reference and data
    """    
    
    # Labels
    if ml_model_type == 'falkon_model':
        label_ref = -1 * np.ones(ref_sample_obj.size)
    elif ml_model_type == 'logfalkon_model':
        label_ref = np.zeros(ref_sample_obj.size)
    else:
        raise Exception("Unknown Model!")
        
    label_data = np.ones(data_sample_obj.size)
    
    Y = np.hstack((label_ref, label_data))
        
    return Y


    
def create_output_folders(exp_out_path, n_toy):
    """
    Create all the output folders

    Parameters
    ----------
    exp_out_path : string
        output folder for the experiment

    Returns
    -------
    None.

    """

    if not os.path.isdir(exp_out_path):
        os.mkdir(exp_out_path)
    
    for i in range(n_toy):
        toy_out_path =  '{}/toy_{}'.format(exp_out_path,i)
        toy_out_path = str(pathlib.PurePath(toy_out_path).as_posix())
        if not os.path.isdir(toy_out_path):
            os.mkdir(toy_out_path)    
    
 


def normalize(X):
    """Standardize dataset

    Args:
        X (np.ndarray): Original Dataset

    Returns:
        np.ndarray: Normalized Dataset
    """    

    X_norm = X.copy()
    
    for j in range(X_norm.shape[1]):
        column = X_norm[:, j]

        mean = np.mean(column)
        std = np.std(column)
    
        if np.min(column) < 0:
            column = (column-mean)*1./ std
        elif np.max(column) > 1.0:                                                                                                                                        
            column = column *1./ mean
    
        X_norm[:, j] = column
    
    return X_norm
    


def read_data(n_events, features, input_path, seed_state, cut_mll = None):
    """Load dataset of size n_events from one or multiple h5 files
    Order of files read and order of samples extracted from single file
    are random.

    Args:
        n_events (int): number of events to be considered (size of the dataset)
        features (List): list of features to be extracted from the .h5 file
        input_path (str): repository of the .h5 file (or multiple .h5 files)
        seed_state (np.random.RandomState): pseudorandom generator
        cut_mll (float, optional): cut for mll feature. Defaults to None.

    Returns:
        (np.ndarray, np.ndarray): dataset and cut vector (None if cut_mll is None)
    """    
    
    n_files = len(glob.glob(input_path + '/*.h5'))

    files_order = np.arange(n_files)                                                                                                                                                           
    seed_state.shuffle(files_order)

    toy_label = input_path.split("/")[-1]
        
    n_features = len(features)
    HLF = np.zeros((n_events,n_features), dtype=np.double)
    cut_vector = np.zeros(n_events, dtype=bool)

    start_idx = 0
    
    for num_file in files_order:
        
        # Read file
        f = h5py.File(input_path+ '/'+toy_label+str(num_file+1)+".h5", 'r')
        
        keys=list(f.keys())
        if len(keys)==0:
            continue
        
        if not set(features).issubset(set(keys)):
            raise Exception("The requested features are {}, \
                    while the file contains {}".format(features,keys))
        
        n_samples = f[features[0]].shape[0]
        
        
        current_samples = np.zeros((n_samples, n_features), dtype=np.double)
        for j in range(n_features):
            current_samples[:,j] = np.array(f[features[j]], dtype=np.double)
        
        if cut_mll is not None:
            current_cut = np.array(f['mll']) > cut_mll

        seed_state.shuffle(current_samples)
        
        if start_idx + n_samples >= n_events:
            offset = n_events - start_idx
        else:
            offset = n_samples
        
        HLF[start_idx:start_idx+offset, :] = current_samples[:offset, :]
        if cut_mll is not None:
            cut_vector[start_idx:start_idx+offset] = current_cut[:offset]      
        start_idx += offset
        if start_idx == n_events:
            break

        f.close()

    if start_idx != n_events:
        raise Exception("Only {} events were sampled, instead of {}"
                        .format(start_idx, n_events))
    
    if cut_mll is not None:
        return HLF, cut_vector
    return HLF, None