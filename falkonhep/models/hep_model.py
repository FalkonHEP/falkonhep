import os

import numpy as np

from falkonhep.utils import read_data, normalize_features

class HEPModel:
    """
    Generic model
    """

    def __init__(self, reference_path, data_path, output_path, options):
        """Create a model for HEP anomaly detection

        Args:
            reference_path (str): path of directory containing data used as reference data (set of .h5 files)
            data_path (str): path of directory containing data used as datasample (set of .h5 files)
            output_path (str): directory in which results will be stored (If the directory doesn't exist it will be created)
            options ([type]): [description]
        """        
        self.reference_path = reference_path
        self.data_path = data_path
        self.output_path = output_path
        self.options = options
        os.makedirs(output_path, exist_ok=True)

    def __generate_nosignal(self, R, B, S, features, cut_mll, normalize, ref_state):
        bkg_size = ref_state.poisson(lam=B)
        bkg, c_vect_bkg = read_data(R + bkg_size, features, self.reference_path, ref_state, cut_mll)
        reference = bkg[:R,:]
        bkg = bkg[R:, :]
        if normalize:
            reference, bkg = normalize_features(reference, bkg)
        if cut_mll is not None:
            c_vect_ref, c_vect_bkg = c_vect_bkg[ : R], c_vect_bkg[R : ]
            return reference[c_vect_ref, :], bkg[c_vect_bkg, :], len(bkg[c_vect_bkg, :]), None
        return reference, bkg, bkg_size, None

    def __generate_resonant(self, R, B, S, features, cut_mll, normalize, ref_state, sig_state):
        bkg_size = ref_state.poisson(lam=B)
        sig_size = sig_state.poisson(lam=S)
        bkg, c_vect_bkg = read_data(R + bkg_size, features, self.reference_path, ref_state, cut_mll)
        reference = bkg[:R, :]
        bkg = bkg[R:, :]
        sig, c_vect_sig = read_data(sig_size, features, self.data_path, sig_state, cut_mll)
        if cut_mll is not None:
            c_vect_ref, c_vect_bkg = c_vect_bkg[:R], c_vect_bkg[R:]
            reference = reference[c_vect_ref, :]
            data = np.vstack((bkg[c_vect_bkg, :], sig[c_vect_sig, :]))
            if normalize:
                reference, data = normalize_features(reference, data)
            return reference, data, len(bkg[c_vect_bkg, :]), len(sig[c_vect_sig, :])
        data = np.vstack((bkg, sig))
        if normalize:
            reference, data = normalize_features(reference, data)
        return reference, data, bkg_size, sig_size

    def __generate_nonresonant(self, R, B, S, features, cut_mll, normalize, ref_state, sig_state):
        reference, c_vect_ref = read_data(R, features, reference_path, ref_state, cut_mll)
        bkg_size = sig_state.poisson(lam=B + S)
        bkg, c_vect_bkg = read_data(bkg_size, features, self.data_path, sig_state, cut_mll)
        if normalize:
            reference, bkg = normalize_features(reference, bkg)
        if cut_mll is not None:
            return reference[c_vect_ref, :], bkg[c_vect_bkg, :], len(bkg[c_vect_bkg, :]), None
        return reference, bkg, bkg_size, None

    def generate_dataset(self, R, B, S, features, cut_mll, normalize, sig_type, ref_state, sig_state):
        """Generate dataset

        Args:
            R (int): Reference size
            B (int): Expected background size
            S (int): Expected signal size
            features (list): List of features
            cut_mll (float): Cut applied to mll
            normalize (bool): If true normalization is applied
            sig_type (int): Type of signal applied (admitted 0: no-signal, 1: resonant, 2: non-resonant)
            ref_state (np.random.RandomSeed): pseudorandom generator used for background size
            sig_state (np.random.RandomSeed): pseudorandom generator used for signal size

        Raises:
            Exception: if the signal type is not in {0, 1, 2}

        Returns:
            (np.ndarray, np.ndarray, int, int (or None)): reference data, data-sample, background size, signal size (if sig_type=1)
        """        
        if sig_type == 0:
            return self.__generate_nosignal(R, B, S, features, cut_mll, normalize, ref_state)
        elif sig_type == 1:
            return self.__generate_resonant(R, B, S, features, cut_mll, normalize, ref_state, sig_state)
        elif sig_type == 2:
            return self.__generate_nonresonant(R, B, S, features, cut_mll, normalize, ref_state, sig_state)
        raise Exception("Unknown signal type")

    def __create_labels(self, ref_size, data_size):
        """Given reference and data size, it returns

        Args:
            ref_size (int): reference sample size
            data_size (int): data sample size
        
        Returns:
            (np.ndarray): returns the label vector
        """        
        pass

    def __build_model(self, model_parameters, weight):
        """Function used to build the model

        Args:
            model_parameters (Map): model parameters
            weight (float): weight
        """        
        pass

    def learn_t(self, R, B, S, features, model_parameters, sig_type, cut_mll = None, normalize = False, seeds = None):
        """Method used to compute the t values 

        Args:
            R (int): Size of the reference \(N_0\)
            B (int): Mean of the Poisson distribution from which the size of the background is sampled
            S (int): Mean of the Poisson distribution from which the size of the signal is sampled
            features (List): List containing the name of the features used
            model_parameters (Map): Dictionary containing the parameters for the model used
            sig_type (int): Type of signal (0: no-signal, 1: resonant, 2: non-resonant).
            cut_mll (int, optional): Cut MLL. Defaults to None.
            normalize (bool, optional): If True data will be normalized before fitting the model. Defaults to False.
            seeds (Tuple, optional): A tuple (reference_seed, data_seed) used to generate reference and data sample, if None two random seeds are generated. Defaults to None.
        """        
        pass

    def save_result(self, fname, i, t, Nw, train_time, ref_seed, sig_seed):
        """Function which save the result of learn_t in a file

        Args:
            fname (str): File name in which the result will be stored (the file will be stored in the output path specified). 
            If the file already exists, the result will be appended. 
            i (int): Toy identifier
            t (float): value of t obtained
            Nw (float): Nw
            train_time (float): Time spent in fitting the model
            ref_seed (int): seed to reproduce the reference sample
            sig_seed (int): seed to reproduce the data sample
        """        
        with open(self.output_path + "/{}".format(fname), "a") as f:
            f.write("{},{},{},{},{},{}\n".format(i, t, Nw, train_time, ref_seed, sig_seed))

