import time
import torch
import numpy as np

from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions


from falkonhep.models import HEPModel
from falkonhep.utils import generate_seeds

class FalkonHEPModel(HEPModel):


    def __create_labels(self, ref_size, data_size):
        ref_labels = np.zeros(ref_size, dtype=np.float64) - 1
        data_labels = np.ones(data_size, dtype=np.float64)
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
                'penalty' : regularization parameter \(\lambda \),
                'maxiter' : [Optional] maximum number of CG iterations (default 10000000),
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
        raise NotImplementedError()
      

    def __build_model(self, model_parameters, weight):

        def weight_fun(Y):
            wvec = torch.ones(Y.shape,dtype=Y.dtype)
            wvec[Y==-1] = weight
            return wvec

        cg_tol = model_parameters['cg_tol'] if 'cg_tol' in model_parameters else 1e-7
        keops_active = model_parameters['keops_active'] if 'keops_active' in model_parameters else "no"
        maxiter = model_parameters['maxiter'] if 'maxiter' in model_parameters else 10000000

        kernel = GaussianKernel(torch.Tensor([model_parameters['sigma']]))
        configuration = {
            'kernel' : kernel,
            'penalty' : model_parameters['penalty'],
            'maxiter' : maxiter,
            'M' : model_parameters['M'],
            'options' : FalkonOptions(cg_tolerance=cg_tol, keops_active=keops_active),
            'weight_fn' : weight_fun,
        }
        if 'seed' in model_parameters:
            configuration['seed'] = model_parameters['seed']
        return Falkon(**configuration)