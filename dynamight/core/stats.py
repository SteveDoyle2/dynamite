import numpy as np
from scipy.stats import norm

def half_power_bandwidth(fmid: float,
                         flower: float,
                         fupper: float) -> float:
    """https://endaq.com/pages/power-spectral-density"""
    df = fupper - flower
    Q = fmid / df
    return Q

#def stats():
    #return kurtosis, skew, crest_factor, skewness, rms, mini, maxi


def signal_stats(a, b, num):
    """
    Parameters
    ----------
    a : float array
        time column
    b : float array
        amplitude column
    num : int
        number of coordinates

    Returns
    -------
    sample_rate : flot
        sample rate
    dt : float
        time step
    mean : float
        average
    std : float
        standard deviation
    rms : float
        root mean square
    skewness : float
    kurtosis : float
        peakedness
    duration : float

    https://www.vibrationdata.com/Python.htm
    https://www.vibrationdata.com/python/tompy.py
    """
    bmax = max(b)
    bmin = min(b)

    ave = np.mean(b)

    duration = a[num-1] - a[0];

    dt = duration / float(num-1)
    sample_rate = 1/dt


    rms = np.sqrt(np.var(b))
    std = np.std(b)

    skewness = stats.skew(b)
    kurtosis = stats.kurtosis(b, fisher=False)

    print ("\n max = %8.4g  min=%8.4g \n" % (bmax,bmin))

    print ("     mean = %8.4g " % ave)
    print ("  std dev = %8.4g " % sd)
    print ("      rms = %8.4g " % rms)
    print (" skewness = %8.4g " % skewness)
    print (" kurtosis = %8.4g " % kurtosis)

    print ("\n  start = %8.4g sec  end = %8.4g sec" % (a[0],a[num-1]))
    print ("    dur = %8.4g sec \n" % dur)
    return sr, dt, ave, sd, rms, skewness, kurtosis, dur


def psd_db_scale(response: np.ndarray,
                 dB: float) -> np.ndarray:
    """6 dB = 4x

    3dB (3.01dB to be exact) is a factor of 2 for ASD
    curves (g2/Hz),
    while 6dB is a factor of 2 for grms values.

    For example, reducing a peak ASD value of 12g2/Hz -3dB
    would give you 6g2/Hz; reducing a value of 12grms -3dB
    results in a value of 8.48grms and reducing it -6dB
    results in a value of 6.0grms. This tends to be confusing
    for people new to random vibration.
    """
    scale = 10 ** (dB / 10)
    return response * scale

def grms_srs_db_scale(response: np.ndarray,
                 dB: float) -> np.ndarray:
    """
    3 dB = 2x
    Used for gRMS and Shock ???
    """
    scale = 20 ** (dB / 10)
    return response * scale

def bessel_correction(frequency: np.ndarray,
                      responses: np.ndarray,
                      nsamples: int=0,
                      kscale: float=0.0,
                      probability: float=0.95,
                      confidence: float=0.50):
    """
    Z = (x - mu) / sigma
    Z: z-score
    x: observed value
    mu: mean of the sample
    sigma: standard deviation of the sample
    chi2: chi^2 of the sample???

    https://ntrs.nasa.gov/api/citations/20120001364/downloads/20120001364.pdf
    https://www.vibrationdata.com/tutorials_alt/P9550.pdf

    The problem is to find a tolerance factor k that will yield a limit
    which covers at least 100*beta of the population with a confidence of
    100*gamma percent as expressed by:
    P(F(xbar+ks) >= beta) = gamma, where F is the CDF
    P(95, 50) = P(beta, gamma)

    z=1.645 for 95%

    NOTE: PPF (percent point function) is the inverse of the CDF
    """
    if nsamples == 0 and kscale == 0.0:
        raise RuntimeError('nsamples=0 and kscale=0.0')

    z = norm.ppf(probability)
    gamma = 1 - confidence
    if kscale:
        assert nsamples == 0, 'nsamples=0 and kscale>0.0'
        assert nsamples > 0, f'kscale={kscale}'
    else:
        assert kscale == 0.0, 'nsamples>0 and kscale!=0.0'
        assert nsamples > 0, f'nsamples={nsamples}'
        #Z is the normal distribution value for the
        # confidence percentage Pc
        #kscale = (Zp + Zc / nsamples ** 0.5)

        # If is not known, the sample standard deviation (s)
        # is calculated from the sample data.
        # This approximation is a random variable with a
        # Chi-squared distribution [2] for n-1:

        chi2_nm1 = chi2.ppf(gamma, df=nsamples-1)
        kscale = z * np.sqrt((nsamples-1)/chi2_nm1)

        # for n=3 and confidence (gamma)=50%, xgamma^2 = 1.39
        # for nsamples>>1, k=1.64

        # https://www.youtube.com/watch?v=1IiZ8iFAFbs&ab_channel=LearnChemE
        # 5% distribution above it
        #x_0.05, 9 (n - 1) = 16.92 from table
        #= Chisq.inv.rt(alpha, n-1) = 16.92   (rt = right tail)
        #= Chisq.inv (alpha, n-1) = 3.33 (left tail of PDF)

    #chi2.ppf(0.95, df=5)   # 11.07
    #chi2.cdf(11.07, df=5)  # 0.95

    #chi2.ppf(0.05, df=9)   # 3.33***
    log_responses = np.log10(responses)
    log_xmean = np.mean(log_responses, axis=0)
    log_sample_std = np.std(log_responses, axis=0)
    assert len(log_responses) == len(frequency)

    log_std = kscale * log_sample_std
    log_xl = log_xmean + kscale * log_std
    xl = 10 ** log_xl
    return xl
