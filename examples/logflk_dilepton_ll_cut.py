import numpy as np

import argparse as ap

from falkonhep import LogFalkonHEPModel
from falkonhep.utils import plot_reference

R = 100000
B = 20000

features = ['delta_phi', 'eta1', 'eta2', 'pt1', 'pt2']


def execute_experiment(reference_path, data_path, output_path, sigtype, S, ntoys):

    model_parameters = {
        'sigma' : 3.0,
        'penalty_list' : [1e-6],
        'iter_list' : [1000000],
        'M' : 10000,
        'keops_active': "auto",
        'seed' : 12,
        'cg_tol' : np.sqrt(1e-7)
    }
    cut = ('mll', 95)
    model = LogFalkonHEPModel(reference_path, data_path, output_path, None)

    for i in range(ntoys):
        t, Nw, train_time, ref_seed, sig_seed, _ = model.learn_t(R, B, 0, features, model_parameters, sig_type=0, normalize = True, cut=cut)
        print("[REF] i: {}\tt: {}\tNw: {}\t training time: {}".format(i, t, Nw, train_time))
        model.save_result("reference.log", i, t, Nw, train_time, ref_seed, sig_seed)
#    for i in range(ntoys):
    #    t, Nw, train_time, ref_seed, sig_seed, _ = model.learn_t(R, B, S, features, model_parameters, sig_type=sigtype, normalize = True, cut=cut)
    #    print("[SIG] i: {}\tt: {}\tNw: {}\t training time: {}".format(i, t, Nw, train_time))
    #    model.save_result("signal.log", i, t, Nw, train_time, ref_seed, sig_seed)
    plot_reference(model.output_path + "/reference.log", "DiMuon (low level, cut mll >= 95) reference", model.output_path + "/reference", bins=6)
        

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('reference_path', type=str, help="Path for reference")
    parser.add_argument('data_path', type=str, help="Path for data")

    #./dimuon/logfalkon/cut/zprime300/30/
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
