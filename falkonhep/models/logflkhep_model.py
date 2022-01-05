from falkon import LogisticFalkon
from falkon.kernels import GaussianKernel
from falkon.options import FalkonOptions
from falkon.gsc_losses import WeightedCrossEntropyLoss

from falkonhep.models import HEPModel

class LogFalkonHEPModel(HEPModel):

    def learn_t(self, R, B, S, features, model_parameters, cut_mll = None, normalize = False):
        """Method used to compute the t values 

        Args:
            R (int): Size of the reference \(N_0\)
            B (int): Mean of the Poisson distribution from which the size of the background is sampled
            S (int): Mean of the Poisson distribution from which the size of the signal is sampled
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
            cut_mll (int, optional): Cut MLL. Defaults to None.
            normalize (bool, optional): If True data will be normalized before fitting the model. Defaults to False.
        """
        pass

    def __build_model(self, model_parameters, weight):

        cg_tol = model_parameters['cg_tol'] if 'cg_tol' in model_parameters else 1e-7
        keops_active = model_parameters['keops_active'] if keops_active in model_parameters else "no"
        kernel = GaussianKernel(model_parameters['sigma'])
        configuration = {
            'kernel' : kernel,
            'penalty_list' : model_parameters['penalty_list'],
            'iter_list' : model_parameters['iter_list'],
            'M' : model_parameters['M'],
            'options' : FalkonOptions(cg_tolerance=cg_tol, keops_active=keops_active),
            'loss' : WeightedCrossEntropyLoss(kernel=kernel, neg_weight=weight)
        }
        return LogisticFalkon(**configuration)