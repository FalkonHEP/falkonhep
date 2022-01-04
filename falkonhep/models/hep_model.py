import os

class HEPModel:
    """
    Generic model
    """

    def __init__(self, reference_path, data_path, output_path, options):
        
        self.reference_path = reference_path
        self.data_path = data_path
        self.output_path = output_path
        self.options = options
        os.makedirs(output_path, exist_ok=True)

    def learn_t(self, R, B, S, features, model_parameters, cut_mll = None, normalize = False):
        """
        Method used to compute the t values 

        Args:
            R (int): Size of the reference N_0
            B (int): Mean of the Poisson distribution from which the size of the background is sampled
            S (int): Mean of the Poisson distribution from which the size of the signal is sampled
            model_parameters (Map): Dictionary containing the parameters for the model used
            features (List): List containing the name of the features used
            cut_mll (int, optional): Cut MLL. Defaults to None.
            normalize (bool, optional): If True data will be normalized before fitting the model. Defaults to False.
        """        
        pass