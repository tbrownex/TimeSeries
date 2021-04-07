import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def getFreq(df):
    fft = tf.signal.rfft(df['temp'])
    f_per_dataset = np.arange(0, len(fft))
    
    n_samples_h = len(df['temp'])
    hours_per_year = 24*365.2524
    years_per_dataset = n_samples_h/(hours_per_year)

    f_per_year = f_per_dataset/years_per_dataset
    plt.step(f_per_year, np.abs(fft))
    plt.xscale('log')
    plt.ylim(0, 400000)
    plt.xlim([0.1, max(plt.xlim())])
    plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    _ = plt.xlabel('Frequency (log scale)')
    return