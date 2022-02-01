import numpy as np
import h5py
import glob
import os
import datetime
import time
import pathlib
import torch
from scipy.stats import chi2, norm

def z_score_chi2(t_obs, dof = None, t_ref = None):
    assert (dof is not None) or (t_ref is not None)
    if dof is not None:
        p = chi2.cdf(float('inf'),dof)-chi2.cdf(t_obs,dof)
    else: 
        p = np.sum(t_ref >= t_obs) / t_ref.shape[0]

    return norm.ppf(1 - p)

def compute_zscores(t_obs, dof=None, t_ref=None):
    assert (dof is not None) or (t_ref is not None)
    z_scores = np.zeros(t_obs.shape[0])
    for idx, t in enumerate(t_obs):
        z_scores[idx] = z_score_chi2(t, dof = dof, t_ref = t_ref)                   
    return z_scores




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
    


def read_data(n_events, features, input_path, seed_state, cut : tuple = None):
    """Load dataset of size n_events from one or multiple h5 files
    Order of files read and order of samples extracted from single file
    are random.

    Args:
        n_events (int): number of events to be considered (size of the dataset)
        features (List): list of features to be extracted from the .h5 file
        input_path (str): repository of the .h5 file (or multiple .h5 files)
        seed_state (np.random.RandomState): pseudorandom generator
        cut (tuple[str, float], optional): remove all rows s.t. has value for feature cut[0] lower than cut[1]. Defaults to None.

    Raises:
        Exception: Feature specified are not in the dataset
        Exception: Not able to sample n_events events

    Returns:
        (np.ndarray, np.ndarray): dataset and cut vector (None if cut is None)

    """    
    
    files = glob.glob(input_path + '/*.h5')
    n_files = len(files)

    files_order = np.arange(n_files)                                                                                                                                                           
    seed_state.shuffle(files_order)

    toy_label = input_path.split("/")[-1]
        
    n_features = len(features)
    HLF = np.zeros((n_events,n_features), dtype=np.float64)
    cut_vector = np.zeros(n_events, dtype=bool)

    start_idx = 0
    
    for num_file in files_order:
    #    print("[--] file: {}".format(files[num_file]))
        # Read file
#        f = h5py.File(input_path+ '/'+toy_label+str(num_file+1)+".h5", 'r')
        f = h5py.File(files[num_file], 'r')
        
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
        
        if cut is not None:
            current_cut = np.array(f[cut[0]]) > cut[1]

        seed_state.shuffle(current_samples)
        
        if start_idx + n_samples >= n_events:
            offset = n_events - start_idx
        else:
            offset = n_samples
        
        HLF[start_idx:start_idx+offset, :] = current_samples[:offset, :]
        if cut is not None:
            cut_vector[start_idx:start_idx+offset] = current_cut[:offset]      
        start_idx += offset
        if start_idx == n_events:
            break

        f.close()

    if start_idx != n_events:
        raise Exception("Only {} events were sampled, instead of {}"
                        .format(start_idx, n_events))
    
    if cut is not None:
        return HLF, cut_vector
    return HLF, None
