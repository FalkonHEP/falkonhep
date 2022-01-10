import torch
import numpy as np

from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from falkonhep.models import HEPModel

class LogFalkonHEPModel(HEPModel):

    def create_labels(self, ref_size, data_size):
        ref_labels = np.zeros(ref_size, dtype=np.float64)
        data_labels = np.ones(data_size, dtype=np.float64)
        return np.hstack((ref_labels, data_labels))

    def make_predictions(self, model, reference, data_sample):
        ref_pred = model.predict(torch.from_numpy(reference).contiguous())
        data_pred = model.predict(torch.from_numpy(data_sample).contiguous())
        return ref_pred, data_pred
        
    def build_model(self, model_parameters, weight):

        cg_tol = model_parameters['cg_tol'] if 'cg_tol' in model_parameters else 1e-7
        keops_active = model_parameters['keops_active'] if 'keops_active' in model_parameters else "no"
        kernel = GaussianKernel(torch.Tensor([model_parameters['sigma']]))
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