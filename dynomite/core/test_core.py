import unittest

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from dynomite.core.freq_utils import _to_twosided_fsampling
from dynomite.core.time import TimeSeries
from dynomite.core.psd import PowerSpectralDensity

class TestCore(unittest.TestCase):
    def test_1(self):
        time = np.linspace(0., 1., num=101)
        time_response = 2 * np.sin(time)
        resp = TimeSeries(time, time_response, label='cat')
        resp.plot(y_units='g', ifig=1, ax=None, linestyle='-o', show=True)

    def test_main(self):
        #  [0, 1, 2, *3, 4, 5, 6]
        fsamp2 = _to_twosided_fsampling(fmax=3, df=1, sided=1, is_onesided_center=1)
        assert np.allclose(fsamp2, 6), fsamp2

        #  [0, 1, *1.5, 2, 3]
        fsamp1 = _to_twosided_fsampling(fmax=1, df=1, sided=1, is_onesided_center=0)
        assert np.allclose(fsamp1, 3), fsamp1

        A = 2.
        freq = 10  # Hz
        #nperiods = 5.
        #npoints_per_period = 1001
        df = 1
        fmax = 2000.
        dt = 1 / fmax
        #tmax = 1 / df
        nfreqs = int(fmax / df)
        ntimes = nfreqs
        #tmax = nperiods * npoints_per_period * dt
        #ntimes = int(nperiods * npoints_per_period * dt)
        #ntimes = tmax / dt

        t = np.linspace(0., ntimes, num=ntimes) * dt
        y = A * np.sin(2*np.pi*freq*t)

        time_y = TimeSeries(t, y, label=['numpy'])
        time_y2 = time_y.to_fft(sided=1, fft_type='real_imag').to_psd(sided=1).to_onesided().to_time_series()
        if 0:
            time_y.plot(ifig=1, show=False)

        #fft_y = time_y.to_fft(fft_type='mag_phase', sided=2)
        fft_y = time_y.to_fft(fft_type='real_imag', sided=2)
        if 0:
            fft_y.plot_real_imag(ifig=2, xscale='log', show=False)
            fft_y.plot_mag_phase(ifig=3, xscale='log', show=False)
        psd_fft_y0 = fft_y.to_psd(sided=2)
        psd_y1 = time_y.to_psd(sided=2)
        #psd_y.label = ['numpy']
        if 0:
            psd_y1.plot(ifig=4, yscale='linear', show=False)

        psd_y2 = time_y.to_psd_welch(sided=1, window='boxcar')
        psd_y2.label = ['welch']
        if 0:
            psd_y2.plot(ifig=4, xscale='log', yscale='linear', show=False)

        fig1 = plt.figure(5)
        ax1, ax2, ax3 = fig1.subplots(nrows=3)

        time_y_from_fft = fft_y.to_time()
        fft_y.to_onesided()
        psd_fft_y0.to_onesided()
        psd_y1.to_onesided()
        psd_y2.to_onesided()

        time_y.plot(ax=ax1, y_units='g', linestyle='-', show=False)
        time_y_from_fft.plot(ax=ax1, y_units='g', linestyle='--', show=False)
        #fft_y.plot_mag(ax=ax2, y_units='g', show=False)

        #vrs.plot(ax=ax2, y_units='g', yscale='linear', linestyle='--o', show=False)

        psd_fft_y0.plot(ax=ax3, y_units='g', yscale='linear', show=False)
        psd_y1.plot(ax=ax3, y_units='g', yscale='linear', linestyle='--o', show=False)
        psd_y2.plot(ax=ax3, y_units='g', yscale='linear', linestyle='--', show=False)

        #fft_y.to_twosided()

        psd_fft_y0.to_twosided()
        psd_y1.to_twosided()
        psd_y2.to_twosided()
        plt.close()

        frequency = np.array([20., 150., 600., 2000])
        psd_response = np.array([0.0053, 0.04, 0.04, 0.0036])
        num = 1 + 2000 // 2
        frequency2 = np.linspace(20., 2000., num=num,)
        #frequency2[0] = 1e-6
        #psd_response2 = np.interp(frequency2, frequency, psd_response)

        psd1 = PowerSpectralDensity(
            frequency, psd_response, label=['base'], sided=1,
            is_onesided_center=True, octave_spacing=0)

        psd2 = psd1.resample(frequency2, inplace=False)
        psd2.label = ['interp']
        vrs = psd2.to_vrs(Q=10)
        vrs.label = ['vrs']

        vrs2 = psd2.to_sdof_vrs_response(Q=10, fn=100)
        vrs2.label = ['vrs2']

        fig = plt.figure(10)
        #ax1 = fig.gca()
        ax1, ax2 = fig.subplots(nrows=2)
        psd1.plot(ax=ax1, y_units='g', xscale='log', yscale='log', linestyle='-', show=False)
        #psd2.plot(ax=ax1, y_units='g', xscale='log', yscale='log', linestyle='--', xlim=xlim, show=False)
        xlim = None
        vrs.plot(ax=ax2, y_units='g', xscale='log', yscale='log', linestyle='-', xlim=xlim, show=False)

        xlim = (10., 2000.)
        #vrs2.plot(ax=ax1, y_units='g', xscale='log', yscale='log', linestyle='--', xlim=xlim, show=False)

        #ax1.set_ylim([0.001, 0.1])
        ax2.set_ylim([1, 20.])
        #ax1.set_xlim([10, 2000])
        #ax2.set_xlim([10, 2000])

    def test_plot_tf(self):
        fig11 = plt.figure(11)
        ax11 = fig11.gca()
        frequency = np.array([20., 2000])
        psd_response = np.array([1, 1])
        psd2 = PowerSpectralDensity(
            frequency, psd_response, label=['base'], sided=1,
            is_onesided_center=True, octave_spacing=0)
        frequency2 = np.linspace(20., 2000., num=2001-20)
        psd2.resample(frequency2, inplace=True)

        for Q in [5, 10, 25, 50]:
            tf = psd2.to_sdof_transmissibility(Q=Q, fn=100)
            tf.label = [f'Q={Q}']
            tf.frequency /= 100.
            tf.plot_mag(ax=ax11, y_units='g', xscale='log', yscale='log', linestyle='-', show=False)
        ax11.set_xlim((0.1, 10.))
        ax11.set_ylim((0.01, 100.))
        #ax11.xaxis.set_major_formatter('{:.1f} km')
        ax11.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        ax11.yaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
        plt.show()

if __name__ == '__main__':
    unittest.main()
