from __future__ import annotations
from typing import Optional # , Union
from pathlib import PurePath, Path
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from dynamight.typing import Limit
from dynamight.core.load_utils import _update_label, _response_squeeze
import dynamight.core.fourier_transform as ft
#import dynamight.core.time as time
import dynamight.core.psd as dypsd # PowerSpectralDensity
from dynamight.core.freq_utils import fft_to_psd_df
from dynamight.core.srs import ShockResponseSpectra, time_to_srs


class TimeSeries:
    def __init__(self, time: np.ndarray,
                 time_response: np.ndarray,
                 label: list[str]=None):
        if time_response.ndim == 1:
            time_response = time_response.reshape(len(time_response), 1)
        if 'complex' in time_response.dtype.name:
            raise TypeError(time_response)

        assert isinstance(time, np.ndarray), time
        self.time = time

        assert isinstance(time_response, np.ndarray), time_response
        self.response = time_response

        self.label = _update_label(label)

    @property
    def ntimes(self) -> int:
        return self.time

    @property
    def nresponses(self) -> int:
        return self.response.shape[1]

    @property
    def dt(self) -> float:
        #dt = self.time[1] - self.time[0]  # good if precision is high

        # good for lower precision
        N = len(self.time)
        dt = (self.time[-1] - self.time[0]) / (N - 1)
        return dt

    @property
    def tmin(self) -> float:
        return self.time[0]
    @property
    def tmax(self) -> float:
        return self.time[-1]

    @property
    def t_total(self) -> float:
        return self.time[-1] - self.time[0]

    @property
    def df(self) -> float:
        t_total = self.t_total
        return 1 / t_total
    @property
    def sampling_frequency(self) -> float:
        dt = self.dt
        Fs = 1 / dt
        return Fs
    @property
    def nyquist_frequency(self) -> float:
        fnyquist = self.sampling_frequency / 2
        return fnyquist

    def set_time_range(self, tstart: float, tfinal: float) -> TimeSeries:
        it0, itf = np.searchsorted(self.time, [tstart, tfinal])
        self.time = self.time[it0:itf]
        self.response = self.response[it0:itf, :]
        return self

    def set_start_time(self, tstart: float) -> TimeSeries:
        it0 = np.searchsorted(self.time, tstart)
        self.time = self.time[it0:]
        self.response = self.response[it0:, :]
        return self

    def set_end_time(self, tfinal: float) -> TimeSeries:
        itf = np.searchsorted(self.time, tfinal)
        self.time = self.time[:itf]
        self.response = self.response[:itf, :]
        return self


    def remove_mean(self) -> TimeSeries:
        mean = self.response.mean(axis=0)
        assert len(mean) == self.nresponses
        self.response -= mean[np.newaxis, :]
        return self

    def detrend(self, type='constant') -> TimeSeries:
        """type: linear, constant"""
        #nresponses = self.nresponses
        out = sp.signal.detrend(self.response, type=type)
        assert out.shape == self.response.shape
        self.response = out
        return self

    def downsample_by_n(self, n: int) -> TimeSeries:
        self.time = self.time[::n]
        self.response = self.response[::n, :]
        return self

    def resample_time_by_length(self) -> TimeSeries:
        """
        This is useful when for a constant sample rate, there just isn't enough precision

        Time  Response
        ----  --------
        0.1   0.006840
        0.1   0.017972
        0.2   0.036524
        0.2   0.036524
        0.3   0.027866
        0.3   0.013025

        """
        time = self.time
        # time (sec)
        t0 = time[0]
        tmax = time[-1]
        dt = tmax - t0

        #Fs = 1 / dt   # sample rate (Hz)
        #T = 1 / Fs  #  period (sec)
        num = len(time)
        time2 = np.linspace(t0, tmax, num=num, endpoint=True, dtype=time.dtype)
        self.time = time2
        return self

    @classmethod
    def load_from_csv_filename(cls, csv_filename: Path | str, delimiter: str=','):
        if isinstance(csv_filename, PurePath):
            csv_filename = str(csv_filename)

        delimiter = None
        if csv_filename.lower().endswith('.csv'):
            delimiter = ','
        time_data = np.loadtxt(csv_filename, delimiter=delimiter)
        time = time_data[:, 0]
        data = pd.read_csv(csv_filename, delimiter=delimiter)
        cols = data.columns
        time = data[cols[0]].to_numpy()
        labels = cols[1:]
        response = data[labels].to_numpy()

        labels = [label.strip() for label in labels.tolist()]
        return TimeSeries(time, response, label=labels)

    def to_fft(self, sided: int=1, fft_type: str='real_imag') -> ft.FourierTransform:
        assert fft_type in {'mag_phase', 'real_imag'}, f'fft_type={fft_type}'

        t_total = self.t_total
        dts = np.diff(self.time)
        dt = dts.mean()
        ntimes = len(self.time)

        is_onesided_center = (ntimes % 2 == 1)
        frequency = _onesided_frequency(dt, t_total, ntimes)

        df = 1 / t_total
        # ntimes = nfreq
        # len(frequency, frequency2) = ntimes
        # -> two sided
        #fsampling = 1 / dt
        #frequency2 = np.fft.fftfreq(ntimes, d=1/fsampling) *fsampling
        #df = 1 / tmax
        #fsampling = 1 / dt
        #fnyquist = fsampling / 2
        #frequency = np.arange(0, ntimes) * df

        # TODO: support processing multiple results at once
        response = _response_squeeze(self.response)
        fft_response = sp.fft.fft(response, n=None, axis=-1, norm=None,
                         overwrite_x=False, workers=None, plan=None)
        fft_response /= ntimes
        fft_response = fft_response.reshape(self.response.shape)


        if sided == 1:
            assert len(frequency) == ntimes
            N = ntimes
            #is_odd = N % 2
            #if is_odd:
                #frequency = frequency[:N//2]
                ## TODO: handle the half point
                #fft_response2 = fft_response[(N+1)//2]
                #fft_response2[-1] /= 2
            #else:
                #fft_response2 = fft_response[N//2]
            frequency = frequency[:N//2]
            fft_response = fft_response[:N//2, :]
        else:
            assert len(frequency) == ntimes
            assert fft_response.shape[0] == ntimes, (fft_response.shape, ntimes)

        if fft_type == 'mag_phase':
            mag = np.abs(fft_response)
            phase = np.arctan2(fft_response.imag, fft_response.real)
            fft_response = mag + 1j * phase

        fft = ft.FourierTransform(
            frequency, fft_response, label=self.label, fft_type=fft_type,
            sided=sided, is_onesided_center=is_onesided_center)
        return fft
    def to_time_windowed_data(self, iresponse: int=0,
                              window_size_sec: float= 1.0,
                              overlap: float=0.5) -> np.ndarray:
        ntimes = len(self.time)
        # dt = np.diff(self.time).mean()
        tmax = self.t_total
        windows = get_windows(ntimes, tmax, window_size_sec, overlap)
        nwindows = len(windows)

        window0 = windows[0]
        ntimes_window = slice_len_for(window0, ntimes)

        times = np.zeros((ntimes_window, nwindows), dtype=self.response.dtype)
        time_response = np.zeros((ntimes_window, nwindows), dtype=self.response.dtype)
        for iwindow, window in enumerate(windows):
            ntimes_windowi = slice_len_for(window, ntimes)
            times[:, iwindow] = self.time[window]
            time_response[:, iwindow] = self.response[window, iresponse]
        return windows, times, time_response

    def to_time_windowed(self, iresponse: int=0,
                         window_size_sec: float= 1.0,
                         overlap: float=0.5) -> TimeSeries:
        windows, unused_times, time_response = self.to_time_windowed_data(
            iresponse=iresponse, window_size_sec=window_size_sec,
            overlap=overlap)
        window0 = windows[0]
        time0 = self.time[window0]
        time_ = TimeSeries(time0, time_response, label='')
        return time_

    def to_psd_windowed(self, sided: int=1) -> dypsd.PowerSpectralDensity:
        return self.to_fft(sided=1, fft_type='real_imag').to_psd()
        #return time_.to_psd_welch(sided=1, window='hann', window_size_sec=1.0, overlap=0.5)

    def to_psd_welch(self, sided: int=1, window: str='hann',
                     window_size_sec: float= 1.0,
                     overlap: float=0.5) -> dypsd.PowerSpectralDensity:
        """
        Parameters
        ----------
        overlap : float; default=0.5
            percentage (0.0 to 1.0)

        """
        window = window.lower()
        assert sided in {1, 2}, sided
        assert overlap >= 0.0 and overlap <= 1.0, overlap
        return_onesided = (sided == 1)
        #fsampling = 1 / self.dt

        #ntimes = len(self.time)
        #df = 1 / self.tmax
        fsampling = 1 / self.dt
        #fnyquist = fsampling / 2
        #frequency = np.arange(0, ntimes) * df

        window_size_int = int(fsampling * window_size_sec)
        overlap_int = int(fsampling * overlap)

        #nfft - for 0 padded signals
        #ntimes = len(self.time)
        #is_onesided_center = (ntimes % 2 == 1)
        psd_responses_list = []
        nresponses = self.nresponses
        for iresponse in range(nresponses):
            response = self.response[:, iresponse]
            #response = _response_squeeze(response)
            frequency, psd_responsei = sp.signal.welch(
                response, fs=fsampling, window=window,
                nperseg=window_size_int, noverlap=overlap_int, nfft=None,
                detrend='constant', return_onesided=return_onesided,
                scaling='density', axis=-1, average='mean')
            psd_responses_list.append(psd_responsei)
        psd_response = np.column_stack(psd_responses_list)
        nfreq = len(frequency)
        assert psd_response.shape == (nfreq, nresponses)

        fmax = frequency[-1]
        if sided == 1:
            is_onesided_center = not np.allclose(fmax, fsampling)
        else:
            # doesn't matter
            is_onesided_center = True
        psd = dypsd.PowerSpectralDensity(
            frequency, psd_response, label=self.label,
            sided=sided, is_onesided_center=is_onesided_center,
            octave_spacing=0)
        return psd

    def to_psd(self, sided: int=1) -> dypsd.PowerSpectralDensity:
        assert sided in {1, 2}, sided
        tmax = self.time[-1]

        dt = self.dt
        #dts = np.diff(self.time)
        #dt = dts.mean()
        ntimes = len(self.time)

        frequency = _onesided_frequency(dt, tmax, ntimes)
        response = _response_squeeze(self.response)
        fft_response = sp.fft.fft(response, n=None, axis=-1, norm=None,
                                  overwrite_x=False, workers=None, plan=None)
        fft_response /= ntimes

        #psd_response = fft_response * np.conj(fft_response) / df
        psd_response = fft_to_psd_df(frequency, fft_response)
        assert fft_response.shape[0] == ntimes, (fft_response.shape, ntimes)
        is_onesided_center = (ntimes % 2 == 1)
        #assert sided == 1
        psd = dypsd.PowerSpectralDensity(
            frequency, psd_response, label=self.label,
            sided=sided, is_onesided_center=is_onesided_center)
        return psd

    def to_srs(self, fmin: float=1.0, fmax: float=1_000.,
               Q: float=10.,
               noctave: int=6,
               calc_accel_srs: bool=True,
               calc_rel_disp_srs: bool=True) -> ShockResponseSpectra:
        freq, accel_neg, accel_pos, rel_disp_neg, rel_disp_pos = time_to_srs(
            self.time, self.response, Q,
            fmin=fmin, fmax=fmax, noctave=noctave)
        shock = ShockResponseSpectra(
            freq, Q,
            label=self.label,
            srs_min=accel_neg,
            srs_max=accel_pos,
            rel_disp_min=rel_disp_neg,
            rel_disp_max=rel_disp_pos,
            #calc_accel_srs=calc_accel_srs,
            #calc_rel_disp_srs=calc_rel_disp_srs,
        )
        return shock

    def plot(self, y_units: str='g', ifig: int=1,
             ax: Optional[plt.Axes]=None,
             xlim: Optional[Limit]=None,
             linestyle='-o',
             title: str='Time Series',
             show: bool=True) -> tuple[plt.Figure, plt.Axes]:
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        if title:
            ax.set_title(title)
        ax.set_xlabel('Time (sec)')
        ax.set_ylabel('Response (g)')
        ax.plot(self.time, self.response[:, 0], linestyle, label=self.label[0])
        ax.legend()
        ax.grid(True)
        if show:
            plt.show()
        return fig, ax

def _onesided_frequency(dt: float, tmax: float, ntimes: int) -> np.ndarray:
    df = 1 / tmax
    #fsampling = 1 / dt
    #fnyquist = fsampling / 2
    frequency = np.arange(0, ntimes) * df
    return frequency


def get_windows(ntimes, tmax, window_size_sec, overlap) -> list[Any]:
    ntimes_per_sec = ntimes / tmax
    ntimes_per_window = int(window_size_sec * ntimes_per_sec)
    ntimes_per_overlap = int(overlap * ntimes_per_sec)

    i = 0
    windows = []

    ## TODO: drops the last window because it's probably a different size
    while i < ntimes - ntimes_per_window:
        window = slice(i, i+ntimes_per_window)
        i += ntimes_per_overlap
        windows.append(window)
    return windows

def slice_len_for(slc, ntimes: int):
    start, stop, step = slc.indices(ntimes)
    return max(0, (stop - start + (step - (1 if step > 0 else -1))) // step)
