import numpy as np
import matplotlib.pyplot as plt


import scipy.stats as stats
from scipy.stats import chi2, chisquare, norm
from .data_utils import compute_zscores


def return_best_chi2dof(tobs):
    """
    Returns the most fitting value for dof assuming tobs follows a chi2_dof distribution,
    computed with a kolmogorov-smirnov test, removing NANs and negative values.
    Parameters
    ----------
    tobs : np.ndarray
        observations
    Returns
    -------
        best : tuple
            tuple with best dof and corresponding chi2 test result
        nans : int
            number of nans in tobs
        negs : int
            number of negative tobs
    """
    
    nans = len([t for t in tobs if str(t) == 'nan']) # count nans
    negs = len([t for t in tobs if t < 0]) # count negative t values
    
    tobs = [t for t in tobs if str(t) != 'nan'] # remove nans
    tobs = [t for t in tobs if t >= 0] # remove negative t values
    
    
    dof_range = np.arange(np.nanmedian(tobs) - 10, np.nanmedian(tobs) + 10, 0.1)
    
    ks_tests = []
    
    for dof in dof_range:
        
        test = stats.kstest(tobs, lambda x:stats.chi2.cdf(x, df=dof))[0]
        
        ks_tests.append((dof, test))
        
    ks_tests = [test for test in ks_tests if test[1] != 'nan'] # remove nans
    
    ks_tests = [test for test in ks_tests if test[0] >= 0] # retain only positive dof
        
    best = min(ks_tests, key = lambda t: t[1]) # select best dof according to KS test result
        
    return best, nans, negs


def return_best_chi2dof_old(tobs, n_bins):
    
    nans = len([t for t in tobs if str(t) == 'nan']) # count nans
    negs = len([t for t in tobs if t < 0]) # count negative t values
    
    tobs = [t for t in tobs if str(t) != 'nan'] # remove nans
    tobs = [t for t in tobs if t >= 0] # remove negative t values
    
    hist_xrange = (np.min(tobs), np.max(tobs))
    
    obs_hist, bins = np.histogram(tobs, bins=n_bins, range=(hist_xrange[0], hist_xrange[1]))
    
    obs_count = sum(obs_hist)
    
    median = np.median(tobs)
    
    dof_range = np.arange(median - 10, median + 10, 0.1)
    
    chi2_tests = []
    
    for dof in dof_range:
        exp_hist = obs_count * np.array(list(map(lambda x, y : chi2.cdf(x, dof) \
                                             - chi2.cdf(y, dof), 
                                             bins[1:], bins[:-1])))
        
        test = chisquare(f_obs=obs_hist, f_exp=exp_hist)[0]
        
        chi2_tests.append((dof, test))
        
    chi2_tests = [test for test in chi2_tests if test[1] != 'nan'] # remove nans
    
    chi2_tests = [test for test in chi2_tests if test[0] >= 0] # retain only positive dof
        
    best = min(chi2_tests, key = lambda t: t[1]) # select best dof according to chisquare test result
        
    return best, nans, negs


def err_bar(hist, n_samples):
    
    bins_counts = hist[0]
    bins_limits = hist[1]
    
    x   = 0.5*(bins_limits[1:] + bins_limits[:-1])
    
    bins_width = 0.5*(bins_limits[1:] - bins_limits[:-1])
    err = np.sqrt(np.array(bins_counts)/(n_samples*np.array(bins_width)))
    
    return x, err

def read_results(fname):
    t_values, Nw, train_time = [], [], []
    num_nan, num_neg = 0, 0

    with open(fname, "r") as f:
        ris = f.readlines()
    for line in ris:
        splitted = line.split(",")
        if float(splitted[1]) != float(splitted[1]):
            num_nan+=1
            continue
        elif float(splitted[1]) < 0:
            num_neg += 1
            continue
        t_values.append(float(splitted[1]))
        Nw.append(float(splitted[2]))
        train_time.append(float(splitted[3]))
    return np.array(t_values), np.array(Nw), np.array(train_time), num_nan, num_neg

def plot_reference(results_file, title, out_file, bins=6, verbose=False):
    t_values, _, _, num_nan, num_neg = read_results(results_file)
    if verbose:
        print("[++] Results:\n\tTotal = {}\tValid = {}\tInvalid = {}\tNegative = {}".format(len(t_values) + num_nan + num_neg, len(t_values), num_nan, num_neg))
    fig, ax = plt.subplots()
    ax.set_title(title)

    hist = ax.hist(t_values, bins = bins, color='lightblue', edgecolor='dodgerblue',
                    density=True, label='Reference')

    x_err, err = err_bar(hist, t_values.shape[0])
    ax.errorbar(x_err, hist[0], yerr = err, color='dodgerblue', marker='o', ms=6, ls='', lw=1)


    best, nan, neg = return_best_chi2dof(t_values, bins)
    dof = int(best[0])
    chi2_range = chi2.interval(alpha=0.999, df=dof)
    r_len = chi2_range[1] - chi2_range[0]
    x = np.arange(chi2_range[0] - r_len/2, chi2_range[1] + r_len/2, .05)
        
    chisq = stats.chi2.pdf(x, df=dof)       
    ax.plot(x, chisq, color='r', lw=2, label='$\chi^2(${}$)$'.format(dof))

    ax.set_ylim(bottom=0)

    ax.set_ylabel('$\mathbf{P(t)}$')
    ax.set_xlabel('$\mathbf{t}$')
    ax.legend(loc="upper right")
    plt.savefig("{}.pdf".format(out_file))


def plot_sig(ref_file, data_file, title, out_file, bins=6, verbose=False):
    tref_values, _, _, nan_ref, neg_ref = read_results(ref_file)
    rdata_values, _, _, nan_data, neg_data = read_results(data_file)

    if verbose:
        print("[++] Results:\n\t[+] Reference results:\n\t\tTotal = {}\tValid = {}\tInvalid = {}\tNegative = {}".format(len(tref_values) + nan_ref + neg_ref, len(tref_values), nan_ref, neg_ref))
        print("\t[+] Data results:\n\t\tTotal = {}\tValid = {}\tInvalid = {}\tNegative = {}".format(len(rdata_values) + nan_data + neg_data, len(rdata_values), nan_data, neg_data))

    fig, ax = plt.subplots()
    ax.set_title(title)

    hist_ref = ax.hist(tref_values, bins = bins, color='lightblue', edgecolor='dodgerblue',
                    density=True, label='Reference')
    hist_data = ax.hist(rdata_values, bins=bins, color='orange', edgecolor='darkorange',
                    density=True, label='New Physics')

    x_err, err = err_bar(hist_ref, tref_values.shape[0])
    ax.errorbar(x_err, hist_ref[0], yerr = err, color='dodgerblue', marker='o', ms=6, ls='', lw=1)

    x_err, err = err_bar(hist_data, rdata_values.shape[0])
    ax.errorbar(x_err, hist_data[0], yerr = err, color='darkorange', marker='o', ms=6, ls='', lw=1)

#def compute_zscores(t_obs, dof=None, t_ref=None):

    z_sc = compute_zscores(rdata_values, dof=None, t_ref=tref_values)
        
    res = '$z :$ {}'.format(round(np.median(z_sc),2))

    best, nan, neg = return_best_chi2dof(tref_values, bins)
    dof = int(best[0])
    z_sc_chi2 = compute_zscores(rdata_values, dof=dof, t_ref=None)

    res += '\n$t_{{\mathrm{{obs}}}} :$ {}\n$z_{{\chi^2}} :$ {}'.format(round(np.median(rdata_values), 2),
                                                                        round(np.median(z_sc_chi2),2)) 
        
    if verbose:
        print("\t[++] Z and t medians:\n\t\tZ [chi square] = {}\tZ [histogram] = {}\tt = {}".format(
            round(np.median(z_sc_chi2), 2), 
            round(np.median(z_sc), 2),
            round(np.median(rdata_values), 2)))
    xy_text=(0.85, 0.51)
    ax.annotate(res,  xy=xy_text, ha = 'center', va = 'center', size=14,
                    xycoords='axes fraction')



    chi2_range = chi2.interval(alpha=0.999, df=dof)
    r_len = chi2_range[1] - chi2_range[0]
    x = np.arange(chi2_range[0] - r_len/2, chi2_range[1] + r_len/2, .05)
        
    chisq = stats.chi2.pdf(x, df=dof)       
    ax.plot(x, chisq, color='r', lw=2, label='$\chi^2(${}$)$'.format(dof))

    ax.set_ylim(bottom=0)

    ax.set_ylabel('$\mathbf{P(t)}$')
    ax.set_xlabel('$\mathbf{t}$')
    ax.legend(loc="upper right")
    plt.savefig("{}.pdf".format(out_file))
   
