import numpy as np

import argparse as ap

from falkonhep import LogFalkonHEPModel
from falkonhep.utils import plot_reference, plot_sig

R = 500000
B = 100000

features = ['lep1_eta', 'lep1_pT', 'lep1_phi', 'lep2_eta', 'lep2_pT', 'lep2_phi', 'missen_mag', 'missen_phi']


def execute_experiment(reference_path, data_path, output_path, sigtype, S, ntoys_ref, ntoys_data):

    rnd_state = np.random.RandomState(12)
    model_parameters = {
        'sigma' : 4.5,
        'penalty_list' : [1e-6],
        'iter_list' : [1000000],
        'M' : 10000,
        'keops_active': "no",
        "use_cpu" : False,
        'seed' : 12,
        'cg_tol' : np.sqrt(1e-7)
    }
    
    # eventually, normalization function can be specified as last argument
    model = LogFalkonHEPModel(reference_path, data_path, output_path)

#(self, R:int, B:int, S:int, features:list, model_parameters:dict, sig_type:int, 
#                cut:tuple = None, normalize:bool = False, seeds:tuple = None)
#
    for i in range(ntoys_ref):
        model_parameters['seed'] = rnd_state.randint(low = 0, high = 2**32)
        t, Nw, train_time, ref_seed, sig_seed = model.learn_t(R=R, B=B, S=0, features = features, model_parameters=model_parameters, sig_type=0, cut=None, seeds=None, normalize = False)
        print("[REF] i: {}/{}\tt: {}\ttraining time: {}".format(i+1, ntoys_ref, round(t, 2), round(train_time, 2)))
        model.save_result("reference.log", i, t, Nw, train_time, ref_seed, sig_seed)
    for i in range(ntoys_data):
        model_parameters['seed'] = rnd_state.randint(low = 0, high = 2**32)
        t, Nw, train_time, ref_seed, sig_seed = model.learn_t(R, B, S, features, model_parameters, sig_type=sigtype, normalize = False)
        print("[SIG] i: {}/{}\tt: {}\ttraining time: {}".format(i+1, ntoys_data, round(t, 2), round(train_time, 2)))
        model.save_result("signal_{}.log".format(S), i, t, Nw, train_time, ref_seed, sig_seed)
    
    plot_reference(model.output_path + "/reference.log", "SUSY (low level) reference", model.output_path + "/reference", bins=6, verbose = False)
    plot_sig(model.output_path + "/reference.log", model.output_path + "/signal_{}.log".format(S), "SUSY (low level)", model.output_path + "/signal_{}".format(S), bins=6, verbose = True)

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('reference_path', type=str, help="Path for reference")
    parser.add_argument('data_path', type=str, help="Path for data")
    parser.add_argument('output_path', type=str, help="Path for output")

    parser.add_argument('sigtype', type=int, choices=[0, 1, 2], help="Signal type:\n0: no signal\n1: resonant\n2: non-resonant")
    parser.add_argument('NS', type=int, help="Expected signal size")

    parser.add_argument('--ntoys-ref', type=int, default=1, help="Number of toys for reference")
    parser.add_argument('--ntoys-data', type=int, default=1, help="Number of toys for data")

    args = parser.parse_args()

    reference_path = args.reference_path 
    data_path = args.data_path 
    output_path = args.output_path 
    sigtype = args.sigtype
    NS = args.NS
    ntoys_ref, ntoys_data = args.ntoys_ref, args.ntoys_data

    execute_experiment(reference_path, data_path, output_path, sigtype, NS, ntoys_ref, ntoys_data)
