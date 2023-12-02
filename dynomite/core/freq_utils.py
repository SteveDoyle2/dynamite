import numpy as np


def fft_to_psd_df(frequency: np.ndarray, fft_response: np.ndarray) -> np.ndarray:
    df = frequency[1] - frequency[0]
    psd_response = fft_response * np.conj(fft_response) / df
    return psd_response.real

def dft_to_onesided(frequency: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    nfreq = len(frequency)
    is_onesided_center = (nfreq % 2 == 1)
    if is_onesided_center:
        # odd
        #[0, 1, 2, 3, 4]
        # center freq is 2
        ifreq = nfreq // 2 + 1
    else:
        # even
        #[0, 1, 2, 3]
        # center freq is 1.5
        ifreq = nfreq // 2
    frequency2 = frequency[:ifreq]
    response2 = response[:ifreq, :]
    assert len(frequency2) == ifreq, frequency2.shape
    return frequency, response2, is_onesided_center

def psd_to_onesided(frequency: np.ndarray, response: np.ndarray,
                    ) -> tuple[np.ndarray, np.ndarray, bool]:
    nfreq = len(frequency)
    is_onesided_center = (nfreq % 2 == 1)
    k = 2
    if is_onesided_center:
        # odd
        #[0, 1, 2, 3, 4]
        # center freq is 2
        ifreq = nfreq // 2 + 1
        response = k * response[:ifreq, :]
        response[-1, :] /= k  #  don't adjust the Nyquist frequency
    else:
        # even
        #[0, 1, 2, 3]
        # center freq is 1.5
        ifreq = nfreq // 2
        response = k * response[:ifreq, :]

    frequency2 = frequency[:ifreq]
    assert len(frequency2) == ifreq, frequency2.shape
    response[0, :] /= k  #  don't adjust the 0 frequency
    return frequency2, response, is_onesided_center

def psd_to_twosided(frequency: np.ndarray, response: np.ndarray,
                    is_onesided_center: bool, df: float) -> tuple[np.ndarray, np.ndarray]:
    assert response.shape[1] == 1, response.shape
    nfreq = len(frequency)
    #is_even = (nfreq % 2 == 0)
    k = 2
    response2 = 1 / k * np.vstack([response,
                                  np.flipud(response)])
    #  don't adjust the 0/-1 frequency
    response2[0, :] *= k
    #response2[-1, :] *= k
    if is_onesided_center:
        # odd
        #[0, 1, 2, 3, 4]
        # center freq is 2
        nfreq = len(frequency) * 2 - 1
        ifreq = nfreq // 2 + 1
        response2[-1, :] *= k  #  don't adjust the Nyquist frequency
    else:
        # even:
        #[0, 1, 2, 3]
        # center freq is 1.5
        nfreq = len(frequency) * 2
        ifreq = nfreq // 2

    frequency2 = np.arange(0, nfreq) * df
    assert len(frequency2) == nfreq, frequency2.shape
    assert response2.shape[1] == 1, response2.shape
    return frequency2, response2

def _to_twosided_fsampling(fmax: float, df: float,
                           sided: int, is_onesided_center: bool) -> float:
    if sided == 2:
        fsampling = fmax
    else:
        assert sided == 1, sided
        if is_onesided_center:
            fsampling = fmax * 2
        else:
            fsampling = fmax * 2 + df
    return fsampling
