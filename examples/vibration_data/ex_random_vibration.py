import os
from copy import deepcopy
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
#from scipy.fftpack import fft
#USE_GUI = False

from pyyeti.psd import area, rescale, psd2time

from dynamight.core.psd import PowerSpectralDensity, get_grms
from dynamight.core.time import TimeSeries
from dynamight.core.stats import psd_db_scale
#from dynamight.core.psd import PowerSpectralDensity

def mag_transmissibility_random(freq: np.ndarray,
                                Q: float,
                                fnatural: float) -> np.ndarray:
    # transmissibility of signal
    #ft = psd.to_sdof_transmissibility(Q, fnatural)
    #comp_response = ft.response.real
    rho2 = freq / fnatural
    zeta = 1 / (2 * Q)
    tzr2 = 4 * zeta**2 * rho2
    num = 1 + tzr2
    denom = (1 - rho2) ** 2 + tzr2
    comp_response = np.sqrt(num/denom)
    return comp_response

def power_transmissibility_random(freq: np.ndarray,
                                  Q: float,
                                  fnatural: float) -> np.ndarray:
    # transmissibility of signal
    #ft = psd.to_sdof_transmissibility(Q, fnatural)
    #comp_response = ft.response.real
    rho2 = freq / fnatural
    zeta = 1 / (2 * Q)
    tzr2 = 4 * zeta**2 * rho2
    num = 1 + tzr2
    denom = (1 - rho2) ** 2 + tzr2
    comp_response = num/denom
    asdf
    return comp_response

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

def set_background_color(bgcolor: str, *axes: tuple[plt.Axes, ...]) -> None:
    """you have to call this after applying limits or the formatter doesn't work"""
    assert len(axes) > 0, axes
    for ax in axes:
        ax.set_facecolor(bgcolor)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        #ax.yaxis.set_major_formatter(ScalarFormatter())


def get_model() -> tuple[Path, Path, str, list[float]]:
    DIRNAME = Path(os.path.dirname(__file__))
    csv_filename = DIRNAME / 'random_vibration.csv'
    xlim = [20., 2000.]

    csv_filename = DIRNAME / 'car_engine.csv'
    xlim = [20., 200.]

    csv_filename = DIRNAME / 'aircraft_takeoff.csv'
    #xlim = None
    xlim = [20., 2000.]

    basename = os.path.basename(csv_filename).rsplit('.')[0]

    return DIRNAME, csv_filename, basename, xlim

def run_baseline(time: TimeSeries,
                 xlim: list[float],
                 bgcolor: str,
                 colormap_name: str,
                 png_filename: str='',
                 figsize: tuple[float, float]=(8., 8.),
                 dpi: int=200,
                 ylim_psd: Optional[tuple[float, float]]=None,
                 time_og: Optional[TimeSeries]=None) -> tuple[np.ndarray,
                                                              plt.Figure,
                                                              tuple[plt.Axes, plt.Axes]]:
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize, dpi=dpi)
    set_background_color(bgcolor, ax1, ax2)

    if time_og:
        ax1.plot(time_og.time, time_og.response[:, 0], color='grey')

    time_windowed = plot_waterfall_time(
        time, ax1, iresponse=0,
        window_size_sec=10.0, overlap=0.5,
        colormap_name=colormap_name,
        show_time_waterfall=True)

    psd_windowed = time_windowed.to_psd_welch(
        sided=1, window='hann',
        window_size_sec=1.0, overlap=0.5)
    psd_windowed.set_colormap(colormap_name)

    fig, unused_ax = psd_windowed.plot(
        ax=ax2,
        xscale='log', yscale='log',
        xlim=xlim, ylim=None, linestyle='-',
        title='', y_units='g',
        plot_maximax=True,
        show=False)

    log_mean = psd_windowed.plot_log_mean(
        ax2, xlim=None, # filter_nan=False,
        linestyle='-', linewidth=1, label='log mean',
        color='red')
    freq_log_mean = np.column_stack([psd_windowed.frequency, log_mean])

    #fig.subplots_adjust(bottom=0.15)
    if ylim_psd:
        ax2.set_ylim(ylim_psd)
    if png_filename:
        fig.savefig(png_filename)
    return freq_log_mean, fig, (ax1, ax2)

def run_filter(time: TimeSeries,
               xlim: list[float],
               bgcolor: str,
               colormap_name: str,
               png_filename: str='',
               figsize: tuple[float, float]=(8., 8.),
               dpi: int=200,
               ylim_psd: Optional[tuple[float, float]]=None,
               time_og: Optional[TimeSeries]=None) -> tuple[np.ndarray,
                                                            plt.Figure,
                                                            tuple[plt.Axes, plt.Axes]]:
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize, dpi=dpi)
    set_background_color(bgcolor, ax1, ax2)

    ax1.plot(time_og.time, time_og.response[:, 0], color='grey')
    time_windowed = plot_waterfall_time(
        time, ax1, iresponse=0,
        window_size_sec=10.0, overlap=0.5,
        colormap_name=colormap_name,
        show_time_waterfall=True)
    psd_windowed = time_windowed.to_psd_welch(
        sided=1, window='hann',
        window_size_sec=1.0, overlap=0.5)
    log_mean = psd_windowed.get_log_mean()
    freq_log_mean = np.column_stack([psd_windowed.frequency, log_mean])
    psd_windowed2 = psd_windowed.filter_below_curve(log_mean)

    psd_windowed2.set_colormap(colormap_name=colormap_name)
    fig, unused_ax = psd_windowed2.plot(
        ax=ax2,
        xscale='log', yscale='log',
        xlim=xlim, ylim=None, linestyle='-',
        title='', y_units='g',
        plot_maximax=True,
        show=False)
    psd_windowed2.plot_log_mean(
        ax2, xlim=xlim, filter_nan=True,
        linestyle='-', linewidth=2, color='r',
        label='log mean filtered')
    if ylim_psd:
        ax2.set_ylim(ylim_psd)

    if png_filename:
        fig.savefig(png_filename)
    return freq_log_mean, fig, (ax1, ax2)



def main():
    dirname, csv_filename, basename, xlim = get_model()
    colormap_name = 'viridis'
    figsize = (8., 8.)
    dpi = 200
    runall = True
    #runall = False
    #ylim_psd = [1e-7, 1e-2]
    ylim_psd = [1e-5, 1]

    time = TimeSeries.load_from_csv_filename(csv_filename)
    ktime = 3.0
    time.response *= 10  # magnitude is too low -> 100x increase in PSD
    time.time /= ktime  # increase freq by 3x

    time = time.resample_time_by_length()
    time_og = deepcopy(time)
    bgcolor = 'lightgrey'
    if runall:
        png_filename = dirname / f'{basename}_1baseline.png'
        run_baseline(
            time_og, xlim,
            bgcolor, colormap_name, png_filename, ylim_psd=ylim_psd)

        png_filename = dirname / f'{basename}_2nostartuptransient.png'
        time.set_start_time(25.0/ktime)
        run_baseline(
            time, xlim,
            bgcolor, colormap_name, png_filename, time_og=time_og, ylim_psd=ylim_psd)

        #--------------------------------------------------------
        png_filename = dirname / f'{basename}_3filterlogmean.png'
        time.set_start_time(110.0/ktime)
        run_baseline(
            time, xlim,
            bgcolor, colormap_name, png_filename, time_og=time_og, ylim_psd=ylim_psd)

    #--------------------------------------------------------
    png_filename = dirname / f'{basename}_4filternostartuptransient_remove_nan.png'
    time.set_start_time(110.0/ktime)
    run_filter(time, xlim, bgcolor, colormap_name, png_filename=png_filename,
               figsize=figsize, dpi=dpi, time_og=time_og, ylim_psd=ylim_psd)

    png_filename = dirname / f'{basename}_5_logmeannominal.png'
    time_og.set_start_time(25.0/ktime)
    log_mean, fig, (ax1, ax2) = run_filter(
        time_og, xlim, bgcolor, colormap_name, png_filename=png_filename,
        figsize=figsize, dpi=dpi, time_og=time_og, ylim_psd=ylim_psd)

    dat_filename = dirname / f'{basename}_log_mean.dat'
    np.savetxt(dat_filename, log_mean)
    #---------------------------------------------------------------
    part2(dirname, dat_filename, basename, xlim)

    #set_background_color(bgcolor, ax1, ax2)

    plt.show()

def backup(time: TimeSeries,
           time_windowed: TimeSeries,
           dirname: Path,
           basename: str,
           xlim,
           #plot_waterfall: bool,
           #plot_waterfall2: bool,
           #show_time_waterfall: bool,
           #show_psd_waterfall: bool,
           #show_log_mean: bool,
           colormap_name: str,
           dpi: int=200,
           ):
    plot_waterfall = True
    show_time_waterfall = True
    show_psd_waterfall = True
    show_log_mean = True
    plot_waterfall2 = True
    #--------------------------------------------------------

    time.resample_time_by_length().remove_mean()

    bgcolor = 'lightgrey'
    if plot_waterfall:
        fig1, (ax1, ax2) = plt.subplots(nrows=2)
        set_background_color(bgcolor, ax1, ax2)
        ax1.plot(time.time, time.response[:, 0], color='grey')

    if plot_waterfall2:
        fig2, (ax3, ax4) = plt.subplots(nrows=2)
        set_background_color(bgcolor, ax3, ax4)
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
            png_filename = dirname / f'{basename}_waterfall.png'
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
        psd.set_colormap(colormap_name)
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
        png_filename = dirname / f'{basename}.png'
        fig.savefig(png_filename)

    #--------------------------------------------------------
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
    fig, unused_ax = psd_windowed.plot(
        ax=ax2,
        xscale='log', yscale='log',
        xlim=xlim, ylim=None, linestyle='-',
        title='', y_units='g',
        plot_maximax=True,
        show=False)

    print('done')

def grms_from_psd(freq_response: np.ndarray) -> float:
    areai = area(freq_response)
    assert len(areai) == 1, areai
    grms = areai[0] ** 0.5
    return grms

def part2(dirname: Path, dat_filename: Path, basename: str, xlim) -> None:
    assert str(dat_filename).endswith('.dat'), dat_filename
    assert dat_filename.exists(), dat_filename

    dpi = 200
    ndB = 6.0 + 4.9
    fontsize = None # 20
    bgcolor = 'lightgrey'
    log_mean1 = np.loadtxt(dat_filename)
    freq = log_mean1[:, 0]
    P = log_mean1[:, 1]
    p2, freq_center, msv, ms = rescale(P, freq, n_oct=6, freq=None, extendends=True, frange=None)
    log_mean2 = np.column_stack([freq_center, p2])
    grms1 = area(log_mean1)[0]
    #grms2 = area(log_mean2)[0]
    smc_min_work = np.array([
        [20, 0.0053],
        [150, 0.04],
        [800, 0.04],
        [2000, 0.0064],
    ])
    smc_min_work_bound = np.array([
        [20, 0.0053],
        [150, 0.04],
        [290, 0.04],
        [320, 0.1],
        [350, 0.2],
        [396, 0.25],
        [420, 0.25],
        [520, 0.14],
        [680, 0.35],
        [800, 0.35],
        [983.4, 0.02647],
        [2000, 0.0064],
    ])
    #asdf = get_grms(smc_min_work_bound[:, 0], smc_min_work_bound)
    grmsi = grms_from_psd(smc_min_work_bound)

    png_filename2 = dirname / f'{basename}_6_components.png'
    png_filename1 = dirname / f'{basename}_7_smc_min.png'
    #fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=figsize, dpi=dpi)
    fig1 = plt.figure(dpi=dpi)
    fig2 = plt.figure(dpi=dpi)
    ax1 = fig1.gca()
    ax2 = fig2.gca()

    psdi = np.ones(freq.shape)
    psd = PowerSpectralDensity(
        freq, psdi, label='base', sided=1,
        is_onesided_center=False, octave_spacing=0)
    #Q = 1 / (2 * zeta)

    # --------------------------------------------------
    Q1 = 5
    fnatural1 = 400.

    #Q2 = 5
    #fnatural2 = 1200.

    Q3 = 35
    fnatural3 = 1800.
    comp_response1 = mag_transmissibility_random(freq, Q1, fnatural1)
    #comp_response2 = mag_transmissibility_random(freq, Q2, fnatural2)
    comp_response3 = mag_transmissibility_random(freq, Q3, fnatural3)
    mag_component_response_raw = comp_response1 * comp_response3 # comp_response2 *
    component_response_raw = mag_component_response_raw ** 2

    ax2.loglog(freq, comp_response1, color='C0', label='Sub-component 1: $'+f'Q={Q1:.0f}; f_n={fnatural1:.0f}'+'$ Hz')
    #ax2.loglog(freq, comp_response2, color='C1', label='Sub-component 2: $'+f'Q={Q2:.0f}; f_n={fnatural2:.0f}'+'$ Hz')
    ax2.loglog(freq, comp_response3, color='C2', label='Sub-component 2: $'+f'Q={Q3:.0f}; f_n={fnatural3:.0f}'+'$ Hz')
    ax2.loglog(freq, mag_component_response_raw, color='C3', label='Predicted Component Response')

    #plt.close(fig1)
    #plt.close(fig2)
    ax2.legend(fontsize=fontsize)
    ax2.set_title('Component Level Transmissibilities', fontsize=fontsize)
    ax2.set_xlabel('Frequency (Hz)', fontsize=fontsize)
    ax2.set_ylabel('Transmissibility $(G_{out}/G_{in})$', fontsize=fontsize)
    ax2.grid(which='both')
    ax2.set_xlim(xlim)
    #plt.show()

    # --------------------------------
    component_response_rescale, freq_center, msv, ms = rescale(component_response_raw, freq, n_oct=6, freq=None, extendends=True, frange=None)

    engine_comp_raw = log_mean1[:, 1] * component_response_raw

    # 6 dB to maximax
    #
    #The resulting P95/50 spectrum is 4.9 dB above the log-mean
    #maxi-max spectrum from a series of flights or tests (B.1.1).
    mpe_psdi = psd_db_scale(engine_comp_raw, dB=6)
    p95_50_psdi = psd_db_scale(engine_comp_raw, dB=ndB)
    #engine_comp_rescale = log_mean2[:, 1] * component_response_rescale
    p95_50_psd_octi, _freq_center, msv, ms = rescale(p95_50_psdi, freq, n_oct=6)

    response = np.column_stack([freq, engine_comp_raw])
    #mpe_octave = np.column_stack([freq_center, engine_comp_rescale])
    #p95_50 = np.column_stack([freq_center, engine_comp_rescale2])
    mpe = np.column_stack([freq, mpe_psdi])
    p95_50 = np.column_stack([freq, p95_50_psdi])
    p95_50_oct = np.column_stack([freq_center, p95_50_psd_octi])

    grms_response = grms_from_psd(response)
    grms_mpe = grms_from_psd(mpe)
    #grms_mpe_octave = area(mpe_octave)[0]
    grms_p95_50 = grms_from_psd(p95_50)
    grms_p95_50_oct = grms_from_psd(p95_50_oct)
    grms_atp = grms_from_psd(smc_min_work_bound)

    #psd2 = ft.to_psd(sided=1)

    # pre-10x scale = 0.016 grms
    # peak at .000243 @ 254 Hz
    # -> 10x time scale -> 100x PSD scale
    ax1.loglog(freq,             log_mean1[:, 1],                  color='grey', label='Base Input '                 +f' (GRMS={grms1:.2f})')
    ax1.loglog(response[:, 0],   response[:, 1],                   color='C0',   label='Response = Base$*$Component '+f' (GRMS={grms_response:.2f})')
    ax1.loglog(mpe[:, 0],        mpe[:, 1],                        color='C1',   label='MPE (+6 dB; '                +f' GRMS={grms_mpe:.2f})')
    ax1.loglog(p95_50[:, 0],     p95_50[:, 1],                     color='C2',   label='P95/50  (+10.9 dB; '         +f' GRMS={grms_p95_50:.2f})')
    ax1.loglog(p95_50_oct[:, 0], p95_50_oct[:, 1],                 color='C4',   label='P95/50 $1/6^{th} Octave$'    +f' (GRMS={grms_p95_50_oct:.2f})')
    ax1.loglog(smc_min_work_bound[:, 0], smc_min_work_bound[:, 1], color='C3',   label='ATP Test Levels '            +f' (GRMS={grms_atp:.2f})')

    ax1.loglog(smc_min_work[:, 0], smc_min_work[:, 1],
               color='k', linestyle='--',
               label='SMC-S-016 Min Work')
    ax1.set_xlabel('Frequency (Hz)', fontsize=fontsize)
    ax1.set_ylabel('PSD ($g^2$/Hz)', fontsize=fontsize)
    ax1.grid(which='both')
    ax1.set_xlim(xlim)
    ax1.legend(fontsize=fontsize)
    ax1.set_ylim([1e-5, None])
    set_background_color(bgcolor, ax1, ax2)

    fig1.savefig(png_filename1)
    fig2.savefig(png_filename2)
    plt.show()

def main2():
    dirname, csv_filename, basename, xlim = get_model()
    dat_filename = dirname / f'{basename}_log_mean.dat'
    part2(dirname, dat_filename, basename, xlim)

if __name__ == '__main__':
    freq_response = np.array([
        [20, .0053],
        [75, .02],
        [800, .02],
        [2000, .0032],
    ])
    out = grms_from_psd(freq_response)
    print(out)
    main2()

    #main()

    asdf
