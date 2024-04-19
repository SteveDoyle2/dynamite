import os
from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
#import numpy as np
#from scipy.fftpack import fft
#USE_GUI = False

from dynamight.core.time import TimeSeries
#from dynamight.core.psd import PowerSpectralDensity

def plot_waterfall_time(time: TimeSeries,
                        time_ax: plt.Axes,
                        iresponse: int=0,
                        window_size_sec: float=10.0,
                        overlap: float=0.5,
                        colormap_name: str='viridis',
                        show_time_waterfall: bool=True) -> TimeSeries:
    windows, win_time, win_time_response = time.to_time_windowed_data(
        iresponse=0, window_size_sec=10.0,
        overlap=0.5)
    time_windowed = time.to_time_windowed(
        iresponse=0, window_size_sec=10.0, overlap=0.5)
    #time2.remove_mean()

    nwindows = len(windows)
    time_windowed.set_colormap(colormap_name)

    colors = time_windowed.get_colors()
    if show_time_waterfall:
        for iresponse, color in zip(range(nwindows), colors):
            timei = win_time[:, iresponse]
            #print(f'  {iresponse}: [{timei.min():g}, {timei.max():g}]')
            time_responsei = win_time_response[:, iresponse]
            time_ax.plot(timei, time_responsei, color=color)
        time_ax.set_ylabel('Response (g)')
        time_ax.set_xlabel('Time (sec)')
        time_ax.grid(True, which='both')
    return time_windowed

def main():
    DIRNAME = Path(os.path.dirname(__file__))
    csv_filename = DIRNAME / 'random_vibration.csv'
    xlim = [20., 2000.]

    csv_filename = DIRNAME / 'car_engine.csv'
    xlim = [20., 200.]

    csv_filename = DIRNAME / 'aircraft_takeoff.csv'
    #xlim = None
    xlim = [20., 2000.]

    colormap_name = 'viridis'
    plot_waterfall = True
    show_time_waterfall = True
    show_psd_waterfall = True
    show_log_mean = True

    plot_waterfall2 = True

    basename = os.path.basename(csv_filename).rsplit('.')[0]

    time = TimeSeries.load_from_csv_filename(csv_filename)
    time.resample_time_by_length().remove_mean()

    if plot_waterfall:
        fig1, (ax1, ax2) = plt.subplots(nrows=2)
        ax1.plot(time.time, time.response[:, 0], color='grey')

    if plot_waterfall2:
        fig2, (ax3, ax4) = plt.subplots(nrows=2)
        ax3.plot(time.time, time.response[:, 0], color='grey')
        time_og = deepcopy(time)
        time_og.set_start_time(110.0)
        time_windowed_og = plot_waterfall_time(
            time_og, ax3, iresponse=0,
            window_size_sec=10.0, overlap=0.5,
            colormap_name=colormap_name,
            show_time_waterfall=show_time_waterfall)
        time_windowed_og.set_colormap(colormap_name=colormap_name)

        psd_windowed_og = time_windowed_og.to_psd_welch(
            sided=1, window='hann',
            window_size_sec=1.0, overlap=0.5)
        if show_psd_waterfall:
            psd_windowed_og.set_colormap(colormap_name=colormap_name)
            fig, unused_ax = psd_windowed_og.plot(
                ax=ax4,
                xscale='log', yscale='log',
                xlim=xlim, ylim=None, linestyle='-',
                title='', y_units='g',
                plot_maximax=True,
                show=False)

        log_meani = psd_windowed_og.plot_log_mean(
            ax4, xlim=xlim, filter_nan=False, linestyle='-',
            linewidth=1, color='k', label='log mean 2')
        psd_windowed_og2 = psd_windowed_og.filter_below_curve(log_meani)
        psd_windowed_og2.plot_log_mean(
            ax4, xlim=xlim, filter_nan=True,
            linestyle='-', linewidth=2, color='r',
            label='log mean filtered 2')
        #ax2.set_yscale('log')

    #time.set_start_time(20)
    #time.set_end_time(300)
    time.set_time_range(25.0, 110.0)
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
        time_windowed = plot_waterfall_time(
            time, ax1, iresponse=0,
            window_size_sec=10.0, overlap=0.5,
            colormap_name=colormap_name,
            show_time_waterfall=show_time_waterfall)
        psd_windowed = time_windowed.to_psd_welch(
            sided=1, window='hann',
            window_size_sec=1.0, overlap=0.5)

        psd_windowed.set_colormap(colormap_name)
        if show_psd_waterfall:
            fig, unused_ax = psd_windowed.plot(
                ax=ax2,
                xscale='log', yscale='log',
                xlim=xlim, ylim=None, linestyle='-',
                title='', y_units='g',
                plot_maximax=True,
                show=False)
        log_mean = psd_windowed.plot_log_mean(
            ax2, xlim=None, # filter_nan=False,
            linestyle='-', linewidth=1, label='log mean')

        #ax2.plot(psd_windowed.frequency, log_mean,
                 #linestyle='--', label='Log Mean')
        if 0:
            plt.show()
            png_filename = DIRNAME / f'{basename}_waterfall.png'
            fig.savefig(png_filename)
            plt.close()


    if show_log_mean:
        psd_windowed2 = psd_windowed.filter_below_curve(log_mean)
        #time_windowed = plot_waterfall_time(
            #time, ax1, iresponse=0,
            #window_size_sec=10.0, overlap=0.5, colormap_name=colormap_name)
        psd_windowed = time_windowed.to_psd_welch(
            sided=1, window='hann',
            window_size_sec=1.0, overlap=0.5)
        psd_windowed2.plot_log_mean(
            ax2, xlim=xlim, filter_nan=True,
            linestyle='-', linewidth=2, color='r',
            label='log mean filtered')

        psd_windowed_og2.plot_log_mean(
            ax2, xlim=xlim, filter_nan=True,
            linestyle='--', linewidth=2, color='k',
            label='log mean filtered 2')
        ax2.set_yscale('log')

        ax2.legend()
        plt.show()


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


if __name__ == '__main__':
    main()
