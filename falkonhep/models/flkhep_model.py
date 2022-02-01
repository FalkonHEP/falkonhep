import torch
import numpy as np

from falkon import Falkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions


from falkonhep.models import HEPModel

class FalkonHEPModel(HEPModel):


    def create_labels(self, ref_size, data_size):
        ref_labels = np.zeros(ref_size, dtype=np.float64) - 1
        data_labels = np.ones(data_size, dtype=np.float64)
        return np.hstack((ref_labels, data_labels))

    def __loglikelihood(self, f):
        c = 1e-5
        p = (f + 1)/2
        n = (1 - f)/2


     #   p[p <= 0], n[n <= 0] = c, c #1e-10
     #   p[p >= 1], n[n >= 1] = 1 - c, 1 - c# 1e-10
#        n[n >= 1] = 1 - 1e-10
        
        return torch.log(p / n)

    def predict(self, data):
        preds = self.model.predict(torch.from_numpy(data).contiguous())
        return self.__loglikelihood(preds)

    #def make_predictions(self, model, reference, data_sample):
    #    ref_pred = model.predict(torch.from_numpy(reference).contiguous())
    #    data_pred = model.predict(torch.from_numpy(data_sample).contiguous())
    #    
    #    return self.__loglikelihood(ref_pred), self.__loglikelihood(data_pred)


    def build_model(self, model_parameters, weight):

        def weight_fun(Y):
            wvec = torch.ones(Y.shape,dtype=Y.dtype)
            wvec[Y==-1] = weight
            return wvec

        cg_tol = model_parameters['cg_tol'] if 'cg_tol' in model_parameters else 1e-7
        keops_active = model_parameters['keops_active'] if 'keops_active' in model_parameters else "no"
        maxiter = model_parameters['maxiter'] if 'maxiter' in model_parameters else 10000000
        use_cpu = model_parameters['use_cpu'] if 'use_cpu' in model_parameters else False

        kernel = GaussianKernel(torch.Tensor([model_parameters['sigma']]))
        configuration = {
            'kernel' : kernel,
            'penalty' : model_parameters['penalty'],
            'maxiter' : maxiter,
            'M' : model_parameters['M'],
            'options' : FalkonOptions(cg_tolerance=cg_tol, keops_active=keops_active, use_cpu=use_cpu),
            'weight_fn' : weight_fun,
        }
        if 'seed' in model_parameters:
            configuration['seed'] = model_parameters['seed']
        self.model= Falkon(**configuration)