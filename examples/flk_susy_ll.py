import numpy as np

import argparse as ap

from falkonhep import FalkonHEPModel
from falkonhep.utils import plot_reference

R = 500000
B = 100000

features = ['lep1_eta', 'lep1_pT', 'lep1_phi', 'lep2_eta', 'lep2_pT', 'lep2_phi', 'missen_mag', 'missen_phi']


def execute_experiment(reference_path, data_path, output_path, sigtype, S, ntoys):

    print("[--] S = {}".format(S))
    model_parameters = {
        'sigma' : 4.5,
        'penalty' : 1e-5, # 1e-5
        'maxiter' : 1000000,
        'M' : 10000,
        'keops_active': "auto",
        'seed' : 12,
        'cg_tol' : np.sqrt(1e-7)
    }
    model = FalkonHEPModel(reference_path, data_path, output_path, None)

    for i in range(ntoys):
        t, Nw, train_time, ref_seed, sig_seed, ref_pred = model.learn_t(R, B, 0, features, model_parameters, sig_type=0, normalize = True)
        print("[REF] i: {}\tt: {}\tNw: {}\t training time: {}".format(i, t, Nw, train_time))
        model.save_result("reference.log", i, t, Nw, train_time, ref_seed, sig_seed)
    #for i in range(100):
        t, Nw, train_time, ref_seed, sig_seed, ref_pred = model.learn_t(R, B, S, features, model_parameters, sig_type=sigtype, normalize = True)
        print("[SIG] i: {}\tt: {}\tNw: {}\t training time: {}".format(i, t, Nw, train_time))
        model.save_result("signal.log", i, t, Nw, train_time, ref_seed, sig_seed)
    plot_reference(model.output_path + "/reference.log", "SUSY (low level) reference", model.output_path + "/reference", bins=6)
        

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('reference_path', type=str, help="Path for reference")
    parser.add_argument('data_path', type=str, help="Path for data")
    parser.add_argument('output_path', type=str, help="Path for output")

    parser.add_argument('sigtype', type=int, choices=[0, 1, 2], help="Signal type:\n0: no signal\n1: resonant\n2: non-resonant")
    parser.add_argument('NS', type=int, help="Expected signal size")

    parser.add_argument('--ntoys', type=int, default=10, help="Number of toys")

    args = parser.parse_args()

    reference_path = args.reference_path 
    data_path = args.data_path 
    output_path = args.output_path 
    sigtype = args.sigtype
    NS = args.NS
    ntoys = args.ntoys

    execute_experiment(reference_path, data_path, output_path, sigtype, NS, ntoys)
