import time
import torch
import numpy as np

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from falkonhep.models import HEPModel
from falkonhep.utils import generate_seeds, fix

class LogFalkonHEPModel(HEPModel):

    def __create_labels(self, ref_size, data_size):
        ref_labels = np.zeros(ref_size, dtype=np.float32)
        data_labels = np.ones(data_size, dtype=np.float32)
        return np.hstack((ref_labels, data_labels))

    def learn_t(self, R, B, S, features, model_parameters, sig_type, cut_mll = None, normalize = False, seeds = None):
        """Method used to compute the t values 

        Args:
            R (int): Size of the reference \(N_0\)
            B (int): Expected background size
            S (int): Expected signal size
            features (List): List containing the name of the features used
            model_parameters (Map): a map containing the parameters for LogistFalkon e.g.
            ```
            model_parameters = {
                'sigma' : kernel lengthscale,
                'penalty_list' : list of regularization parameters,
                'iter_list' : list of number of CG iterations,
                'M' : number of Nystrom centers,
                'cg_tol' : [Optional] tolerance of CG (default 1e-7),
                'keops_active' : [Optional] if pyKeops will be used (default "no") 
            }
            ```
            sig_type (int): Type of signal (0: no-signal, 1: resonant, 2: non-resonant).
            cut_mll (int, optional): Cut MLL. Defaults to None.
            normalize (bool, optional): If True data will be normalized before fitting the model. Defaults to False.
            seeds (Tuple, optional): A tuple (reference_seed, data_seed) used to generate reference and data sample, if None two random seeds are generated. Defaults to None.
        """
        ref_seed, data_seed = seeds if seeds is not None else generate_seeds(np.random.randint(100))
        ref_state, data_state = np.random.RandomState(ref_seed), np.random.RandomState(data_seed)
        
        reference, data_sample, bck_size, sig_size = self.generate_dataset(R, B, S, features, cut_mll, normalize, sig_type, ref_state, data_state)
        
        data = np.vstack((reference, data_sample))
        data_size = bck_size + sig_size if sig_size is not None else bck_size

        # Labels
        labels = self.__create_labels(reference.shape[0], data_size)
      
        # Create and fit model
        weight = B / R 
        model = self.__build_model(model_parameters, weight)

        Xtorch = fix(data.reshape(data.shape[0], data.shape[1]))
        Ytorch = fix(labels.reshape(-1, 1))        

        train_time = time.time()
        model.fit(Xtorch, Ytorch)
        train_time = time.time() - train_time

        ref_pred = model.predict(torch.from_numpy(reference).contiguous())
        data_pred = model.predict(torch.from_numpy(data_sample).contiguous())

        # Compute Nw and t

        Nw = weight*torch.sum(torch.exp(ref_pred))
        diff = weight*torch.sum(1 - torch.exp(ref_pred))
        t = 2 * (diff + torch.sum(data_pred).item())

        del data_pred, reference, data_sample
        return t, Nw, train_time, ref_seed, data_seed, ref_pred
        
    def __build_model(self, model_parameters, weight):

        cg_tol = model_parameters['cg_tol'] if 'cg_tol' in model_parameters else 1e-7
        keops_active = model_parameters['keops_active'] if 'keops_active' in model_parameters else "no"
        kernel = GaussianKernel(model_parameters['sigma'])
        configuration = {
            'kernel' : kernel,
            'penalty_list' : model_parameters['penalty_list'],
            'iter_list' : model_parameters['iter_list'],
            'M' : model_parameters['M'],
            'options' : FalkonOptions(cg_tolerance=cg_tol, keops_active=keops_active),
            'loss' : WeightedCrossEntropyLoss(kernel=kernel, neg_weight=weight)
        }
        if 'seed' in model_parameters:
            configuration['seed'] = model_parameters['seed']
        return LogisticFalkon(**configuration)