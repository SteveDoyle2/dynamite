from typing import Optional
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

#import dynamight.core.time as dynatime
from dynamight.core.load_utils import _update_label, _response_squeeze

import dynamight.core.time as dytime
import dynamight.core.fourier_transform as ft
import dynamight.core.vrs as dynvrs # VibrationResponseSpectra
from dynamight.core.freq_utils import _to_twosided_fsampling, psd_to_onesided, psd_to_twosided
#from dynamight.plotting.utils import _set_grid
from dynamight.core.plot_utils import _set_grid


class PowerSpectralDensity:
    def __init__(self, frequency: np.ndarray, psd_response: np.ndarray, label: list[str],
                 sided: int=1, is_onesided_center: bool=None, octave_spacing: int=0):
        if psd_response.ndim == 1:
            psd_response = psd_response.reshape(len(psd_response), 1)
        if 'complex' in psd_response.dtype.name:
            raise TypeError(psd_response)
        assert psd_response.shape[1] == 1, psd_response.shape
        assert psd_response.ndim == 2, psd_response.shape
        self.frequency = frequency
        self.response = psd_response
        self.label = _update_label(label)
        self.octave_spacing = octave_spacing
        self.sided = sided
        self.is_onesided_center = is_onesided_center
        #assert sided == 2
        assert is_onesided_center is not None
        assert sided in {1, 2}, sided
        assert octave_spacing >= 0, octave_spacing
        assert isinstance(frequency, np.ndarray), type(frequency)
        assert isinstance(psd_response, np.ndarray), type(psd_response)
        #print('psd-init', self.fsampling, self.df, self.is_onesided_center)

    @property
    def df(self):
        if self.octave_spacing == 0:
            return self.frequency[1] - self.frequency[0]
        raise RuntimeError(self.octave_spacing)

    @property
    def fsampling(self) -> float:
        assert self.sided in {1, 2}
        if self.octave_spacing == 0:
            fmax = self.frequency[-1]
            df = self.df
            fsampling = _to_twosided_fsampling(
                fmax, df, sided=self.sided,
                is_onesided_center=self.is_onesided_center)
            return fsampling
        raise RuntimeError(self.octave_spacing)

    def to_time_series(self):
        if self.sided == 1:
            self = self.to_twosided(inplace=False)

        assert self.sided == 2, self.sided
        assert self.response.shape[1] == 1, self.response.shape
        if self.octave_spacing == 0:
            magnitude = np.sqrt(self.response * self.df)
            npoints = len(self.frequency)
            phase = np.random.uniform(-1., 1.) * np.pi
        else:
            raise NotImplementedError(self.octave_spacing)
        real_imag = magnitude * np.cos(phase) + 1j * np.sin(phase)
        real_imag *= npoints
        real_imag = _response_squeeze(real_imag)
        ifft = sp.fft.ifft(real_imag, n=None, axis=-1, norm=None,
                           overwrite_x=False, workers=None, plan=None)

        dt = 1 / self.fsampling
        time = np.arange(0, npoints) * dt
        time_series = dytime.TimeSeries(time, ifft.real, self.label)
        return time_series

    def to_miles_equation(self, Q: float) -> np.ndarray:
        zeta = 1 / (2 * Q)
        vrs = np.sqrt(np.pi/(4*zeta) * np.frequency)
        return vrs

    def to_sdof_transmissibility(self, Q: float, fn: float):
        """https://www.dataphysics.com/blog/shock-analysis/understanding-shock-response-spectra/"""
        rho = self.frequency / fn
        rho2 = rho ** 2
        zeta = 1 / (2 * Q)
        num0 = 2j * zeta * rho
        num = 1 + num0
        denom = (1 - rho2) + num0
        transmissibility = num / denom
        trans = ft.FourierTransform(
            self.frequency.copy(), transmissibility, self.label,
            sided=1, is_onesided_center=False,
            #octave_spacing=0,
        )
        return trans

    def to_sdof_vrs_response(self, Q: float, fn: float) -> dynvrs.VibrationResponseSpectra:
        """http://www.vibrationdata.com/tutorials_alt/frf.pdf"""
        zeta = 1 / (2 * Q)
        freq2 = self.frequency ** 2
        num = -fn ** 2 * freq2
        denom = fn ** 2 - freq2 + 1j * (2 * zeta * self.frequency * fn)

        psd_response = np.abs(num / denom)
        psd_response *= self.response[0, 0] / psd_response[0]
        vrsi = dynvrs.VibrationResponseSpectra(
            self.frequency, psd_response.real, self.label,
            sided=1, is_onesided_center=False, octave_spacing=0)
        return vrsi

    def to_vrs(self, Q: float) -> dynvrs.VibrationResponseSpectra:
        """vibration response spectra"""
        zeta = 1 / (2 * Q)
        df = np.diff(self.frequency).mean()
        if 0:
            rho = self.frequency[:, np.newaxis] / self.frequency[np.newaxis, :]
            rho2 = rho ** 2
            num0 = 4 * zeta ** 2 * rho2
            denom = ((1 - rho2) ** 2) + num0

            grms0 = ((1 + num0)/ denom)
            grms1 = grms0 * self.response[:, 0] * df
            grms = np.sqrt(grms1.sum(axis=0))
            assert len(grms) == len(self.frequency)
        elif 0:
            fn = 100.
            rho = self.frequency / fn
            rho2 = rho ** 2
            num0 = 4 * zeta ** 2 * rho2
            denom = ((1 - rho2) ** 2) + num0

            grms0 = (1 + num0)/ denom
            grms1 = grms0 * self.response[:, 0] * df
            grms = np.sqrt(grms1.sum())
            grmss = np.array([grms])
            freqs = np.array([fn])
        elif 1:
            freqs = self.frequency
            grmss = self.frequency.copy()
            for i, fn in enumerate(self.frequency):
                rho = self.frequency / fn
                rho2 = rho ** 2
                num0 = 4 * zeta ** 2 * rho2
                denom = ((1 - rho2) ** 2) + num0

                grms0 = (1 + num0)/ denom
                grms1 = grms0 * self.response[:, 0] * df
                grms = np.sqrt(grms1.sum())
                grmss[i] = grms

        vrsi = dynvrs.VibrationResponseSpectra(
            freqs, grmss, label=self.label,
            sided=self.sided,
            is_onesided_center=self.is_onesided_center,
            octave_spacing=self.octave_spacing)
        return vrsi

    def to_onesided(self, inplace: bool=True):
        if self.sided == 1:
            return self
        assert self.sided == 2, self.sided
        frequency, response, is_onesided_center = psd_to_onesided(
            self.frequency, self.response)
        if inplace:
            self.frequency = frequency
            self.response = response
            self.sided = 1
            self.is_onesided_center = is_onesided_center
            out = self
        else:
            out = PowerSpectralDensity(
                frequency.copy(), response.copy(), label=self.label, sided=1,
                is_onesided_center=is_onesided_center,
                octave_spacing=self.octave_spacing)
        return out

    def to_twosided(self, inplace: bool=True):
        if self.sided == 2:
            self.fsampling
            self.df
            return self
        assert self.sided == 1, self.sided

        #print('psd', self.fsampling, self.df, self.is_onesided_center)
        frequency, response = psd_to_twosided(self.frequency, self.response,
                                              self.is_onesided_center, self.df)
        if inplace:
            self.frequency = frequency
            self.response = response
            self.sided = 2
            out = self
        else:
            out = PowerSpectralDensity(
                frequency.copy(), response, self.label, sided=2,
                is_onesided_center=False,
                octave_spacing=self.octave_spacing)
        out.fsampling
        out.df
        return out

    def grms(self) -> np.ndarray:
        grms = get_grms(self.frequency, self.response)
        return grms

    def resample(self, frequency: np.ndarray, inplace: bool=True):
        """uses a log-log interp"""
        # TODO: get rid of for loop
        psd_response = np.zeros(self.response.shape, dtype=self.response.dtype)
        for iresp in range(self.response.shape[1]):
            responsei = self.response[:, iresp]
            psd_response = 2 ** np.interp(
                np.log2(frequency), np.log2(self.frequency), np.log2(responsei))

        if inplace:
            self.frequency = frequency
            self.response = psd_response
            out = self
        else:
            out = PowerSpectralDensity(
                frequency.copy(), psd_response, label=self.label,
                sided=self.sided, is_onesided_center=self.is_onesided_center,
                octave_spacing=self.octave_spacing)
        return out

    def plot(self, ifig: int=1,
             ax: Optional[plt.Axes]=None,
             y_units: str='g', xscale: str='log', yscale: str='log',
             xlim: Optional[tuple[float, float]]=None,
             ylim: Optional[tuple[float, float]]=None,
             linestyle='-o',
             show: bool=True):
        self.fsampling
        self.df
        if ax is None:
            fig = plt.figure(ifig)
            ax = fig.gca()
        ax.set_title('PSD')
        ax.set_xlabel('Frequency (Hz)')
        assert self.octave_spacing == 0, self.octave_spacing
        ax.set_ylabel(f'PSD (${y_units}^2$/Hz)')
        ax.plot(self.frequency, self.response[:, 0], linestyle, label=self.label[0])
        ax.legend()
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        _set_grid(ax, xscale, yscale)
        if show:
            plt.show()


def get_grms(frequency: np.ndarray,
             response: np.ndarray) -> np.ndarray:
    """
    Calculates GRMS from a PSD

    Parameters
    ----------
    frequency (nfreq,) float array
        frequencies
    response: (nfreq, nresponse) float array
        the PSD acceleration response

    https://femci.gsfc.nasa.gov/random/randomgrms.html
    """
    nresponse = response.shape[1]
    log2 = np.log10(2)
    ten_log2 = 10 * log2
    fhigh = frequency[1:]
    flow = frequency[:-1]
    flow_fhigh = flow / fhigh
    fhigh_flow = fhigh / flow
    noctaves = np.log10(fhigh_flow) / log2

    psd_high = response[1:, :]
    psd_low = response[:-1, :]
    db = 10 * np.log10(psd_high / psd_low)
    m = db / noctaves[:, np.newaxis]
    exp = m / ten_log2
    is_close = np.isclose(m, -ten_log2)
    not_close = ~is_close

    A = np.zeros(psd_high.shape, dtype=psd_high.dtype)
    A[not_close] = ten_log2 * psd_high[not_close] / (ten_log2 + m[not_close]) * \
        (fhigh[:, np.newaxis][not_close] - flow[:, np.newaxis][not_close] * (flow_fhigh[:, np.newaxis][not_close]) ** exp[not_close])

    if is_close.sum():
        A[is_close] = psd_low[is_close] * flow[:, np.newaxis][is_close] * np.log(fhigh_flow[:, np.newaxis][is_close])

    Asum = A.sum(axis=0)
    grms = np.sqrt(Asum)
    assert len(grms) == nresponse
    return grms

