import argparse as ap
from falkonhep.utils import plot_sig


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("ref_file", type=str, help="File containing results on reference samples")
    parser.add_argument("data_file", type=str, help="File containing results on data sample")
    parser.add_argument("title", type=str, help="Plot title")
    parser.add_argument("out", type=str, help="Name of the output file")
    parser.add_argument("--bins", type=int, default=6, help="Number of bins")
    
    args = parser.parse_args()

    plot_sig(args.ref_file, args.data_file, args.title, args.out, bins=args.bins)
