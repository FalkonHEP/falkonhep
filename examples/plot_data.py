#!/usr/bin/env python3

import argparse as ap
from falkonhep.utils import plot_sig


#data_path = "./susy/falkon/500"
#out_file = "./susy_falkon_500"
#title = "Falkon: Susy NS = 500 (100 vs 100)"
#
#def plot_results(ref_file, data_file, title, out_file, bins=6):
#    ref_file = data_path + "/reference.log"
#    data_file = data_path + "/signal.log"
#    #def plot_sig(ref_file, data_file, title, out_file, bins=6):
#    plot_sig(ref_file, data_file, title, out_file, bins=bins)
    

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("ref_file", type=str, help="File containing results on reference samples")
    parser.add_argument("data_file", type=str, help="File containing results on data sample")
    parser.add_argument("title", type=str, help="Plot title")
    parser.add_argument("out", type=str, help="Name of the output file")
    parser.add_argument("--bins", type=int, default=6, help="Number of bins")
    
    args = parser.parse_args()

    plot_sig(args.ref_file, args.data_file, args.title, args.out, bins=args.bins)
