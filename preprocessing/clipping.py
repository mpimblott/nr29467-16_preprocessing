
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def histogram(
    data: np.ndarray,
    sdev_factor = 1.5
):
    '''
    Taken from https://stackoverflow.com/a/67153389 'is-there-a-way-i-can-find-the-range-of-local-maxima-of-histogram' answer by Max Pierini.
    Clip the background information by finding the peaks in the distribution of the data and removing data outside of the upper peak
    '''
    sd = np.std(data)
    '''
    Create a histogram of the initial data and create a line plot of the bin averages
    '''
    hist = plt.hist(data, 600, density=True, alpha=.25)
    bin_means = hist[1][:-1] + np.diff(hist[1])[0] / 2
    density = hist[0]
    plt.plot(bin_means, density, linewidth=.5, color='r')
    plt.savefig(f'initial_density_hist.png')
    plt.close()
    '''
    Plot the moving average of the density to smooth out the peaks
    '''
    plt.figure()
    norm_density = density / density.sum()
    moving_avg: np.ndarray = pd.Series(norm_density).rolling(7, center=True).mean().values
    plt.plot(bin_means, norm_density, linewidth=.5, color='b', alpha=.5)
    plt.plot(bin_means, moving_avg, linewidth=.5, color='r')
    plt.legend(['Density', 'Moving Average'])

    '''
    Find the peaks of the moving average, select the 2 most prominent peaks (these should roughly be the peaks of the centres of the 2 distributions)
    '''
    peaks, peak_properties = find_peaks(moving_avg, prominence=0.0001)
    peaks = peaks[np.argsort(peak_properties['prominences'])[-2:]]
    upper_peak = peaks[np.argsort(bin_means[peaks])[-1]]
    upper_cutoff = bin_means[upper_peak] + sdev_factor * sd
    lower_cutoff = bin_means[upper_peak] - sdev_factor * sd
    print(f'lower cutoff: {lower_cutoff}, upper cutoff: {upper_cutoff}')
    # for i, peak in enumerate(peaks):
    #     print(f'prominence at peak {i} ({bin_means[peak]})', peak_prominences(moving_avg, [peak]))
    #     plt.axvline(bin_means[peak], color='g', linewidth=1)
    plt.axvline(bin_means[upper_peak], color='y', linewidth=1)
    plt.axvline(upper_cutoff, color='r', linewidth=1)
    plt.axvline(lower_cutoff, color='r', linewidth=1)
    plt.savefig(f'moving_averages.png')
    plt.close()
    data[data < lower_cutoff] = np.NaN
    data[data > upper_cutoff] = np.NaN
    # plot a histogram of the data after the cutoffs
    valid_data = data != np.NaN
    plt.hist(data[valid_data], bins=256, density=True, alpha=.25)
    plt.axvline(bin_means[upper_peak], color='y', linewidth=1)
    plt.savefig(f'preserved_information.png')
    plt.close()
    return (lower_cutoff, upper_cutoff)