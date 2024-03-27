import os
from pathlib import Path
import matplotlib.pyplot as plt
#import numpy as np
#from scipy.fftpack import fft
#import time
#USE_GUI = False

#import matplotlib as mpl
from dynamight.core.time import TimeSeries


DIRNAME = Path(os.path.dirname(__file__))
csv_filename = DIRNAME / 'random_vibration.csv'
xlim = [20., 2000.]

csv_filename = DIRNAME / 'car_engine.csv'
xlim = [20., 200.]

csv_filename = DIRNAME / 'aircraft_takeoff.csv'
#xlim = None
xlim = [20., 2000.]

plot_waterfall = True


basename = os.path.basename(csv_filename).rsplit('.')[0]

time = TimeSeries.load_from_csv_filename(csv_filename)
time.resample_time_by_length().remove_mean()

if plot_waterfall:
    fig, (ax1, ax2) = plt.subplots(nrows=2)
    ax1.plot(time.time, time.response[:, 0], color='grey')

#time.set_start_time(20)
#time.set_end_time(300)
time.set_time_range(25.0, 150.0)
#time.detrend()

#t_total=306.849595; dt=1.e-04
tmax = time.tmax
t_total = time.t_total
dt = time.dt
print(f'time: t_total={t_total:g}; dt={dt:g}')

#Fs=10000.3; df=0.00325892
Fs = time.sampling_frequency
df = time.df
nyquist_frequency = time.nyquist_frequency
print(f'time: Fs={Fs:g}; fnyq={nyquist_frequency:g}; df={df:g}')


dft = time.to_fft(sided=1, fft_type='real_imag')  # mag_phase
df = dft.df
fsampling = dft.fsampling
print(f'time: fsampling={fsampling:g}; df={df:g}')


if 0:
    psd = time.to_psd_welch(
        sided=1, window='hann',
        window_size_sec=2.0, overlap=0.5)
    psd.downsample_by_n(10)


if plot_waterfall:
    windows, win_time, win_time_response = time.to_time_windowed_data(
        iresponse=0, window_size_sec=10.0,
        overlap=0.5)
    time_windowed = time.to_time_windowed(
        iresponse=0, window_size_sec=10.0, overlap=0.5)
    #time2.remove_mean()
    #psd_windowed = time_windowed.to_psd_windowed(sided=1)
    #psd_windowed.downsample_by_n(10)
    psd_windowed = time_windowed.to_psd_welch(
        sided=1, window='hann',
        window_size_sec=1.0, overlap=0.5)
    psd_windowed.set_colormap('viridis')

    nwindows = len(windows)

    colors = psd_windowed.get_colors()
    for iresponse, color in zip(range(nwindows), colors):
        timei = win_time[:, iresponse]
        #print(f'  {iresponse}: [{timei.min():g}, {timei.max():g}]')
        time_responsei = win_time_response[:, iresponse]
        ax1.plot(timei, time_responsei, color=color)
    ax1.set_ylabel('Response (g)')
    ax1.set_xlabel('Time (sec)')
    ax1.grid(True, which='both')

    log_mean = psd_windowed.get_log_mean()
    fig, ax = psd_windowed.plot(
        ifig=1, ax=ax2,
        xscale='log', yscale='log',
        xlim=xlim, ylim=None, linestyle='-',
        title='', y_units='g',
        plot_maximax=True,
        show=False)
    ax2.plot(psd_windowed.frequency, log_mean,
             color='k', linestyle='--', label='log mean')
    plt.show()
    png_filename = DIRNAME / f'{basename}_waterfall.png'
    fig.savefig(png_filename)
    plt.close()

ifig = 1
if 0:
    fig, ax = dft.plot_mag_phase(
        ifig=ifig, ax=None,
        xscale='log', yscale_mag='log', linestyle='-',
        xlim=xlim, show=False)
    ifig += 1

if 0:
    fig, ax = psd.plot(
        ifig=ifig, ax=None,
        xscale='log', yscale='log',
        xlim=xlim, ylim=None, linestyle='-',
        title='', y_units='g', show=True)
    ifig += 1

if 1:
    psd.set_colormap('viridis')
    dft_psd = dft.to_psd(sided=1)
    dft_psd.label = ['RV Response (Per DFT)']
    psd.label = ['RV Response (Per Welch)']
    #ax = dft.plot_mag(ifig=ifig, ax=None,
                      #xscale='log', yscale='log', linestyle='-',
                      #xlim=xlim, show=False)

    fig, ax = dft_psd.plot(
        ifig=ifig, ax=None,
        xscale='log', yscale='log',
        xlim=xlim, ylim=None, linestyle='-',
        title='', y_units='g', show=False)
    fig, ax = psd.plot(
        ifig=ifig, ax=ax,
        xscale='log', yscale='log',
        xlim=xlim, ylim=None, linestyle='-',
        title='', y_units='g', show=True)
    png_filename = DIRNAME / f'{basename}.png'
    fig.savefig(png_filename)



print('done')
