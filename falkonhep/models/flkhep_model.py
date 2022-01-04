from falkonhep.models import HEPModel

class FalkonHEPModel(HEPModel):

    def __init__(self, reference_path, data_path, output_path, options):
        super().__init__(reference_path, data_path, output_path, options)