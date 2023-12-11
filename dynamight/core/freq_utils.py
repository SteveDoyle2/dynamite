import os
from itertools import count

from numpy.linalg import pinv
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig


def fft_to_psd_df(frequency: np.ndarray, fft_response: np.ndarray) -> np.ndarray:
    df = frequency[1] - frequency[0]
    psd_response = fft_response * np.conj(fft_response) / df
    return psd_response.real

def dft_to_onesided(frequency: np.ndarray,
                    response: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
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
                    is_onesided_center: bool,
                    df: float) -> tuple[np.ndarray, np.ndarray]:
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


def pseudo_response_spectra(acceleration: np.ndarray,
                            dt: float, periods=None,
                            Q: float=10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    What does ins_resp stand for?  Instaneous Response?  response_spectra was the OG name

    The function generates:
     - pseudo-spectral acceleration (PSA)
     - pseudo-spectral velocity (PSV)
     - spectral displacement (SD) spectra
    for given damping ratio (xi).
    Spectral ordinates are for linear-elastic single-degree-of-freedom
    system with unit mass.


    Reference:
    Wang, L.J. (1996). Processing of near-field earthquake accelerograms:
    Pasadena, California Institute of Technology.

    This code is converted from Matlab code of Dr. Erol Kalkan, P.E.
    Link:
    https://www.mathworks.com/matlabcentral/fileexchange/57906-pseudo-spectral-acceleration--velocity-and-displacement-spectra?s_tid=prof_contriblnk

    Pulled from: https://github.com/dertuncay/Response-Spectra/tree/master
    modified to be a bit cleaner/vectorized

    Parameters
    ----------
    data : np.ndarray
        acceleration (cm/s^2)
    dt : float
        sampling interval
    periods: np.ndarray
        spectral periods (Default: 0.01 to 10 seconds with 100 samples)
    Q : float; default=10.0
        amplifcation factor; xi = 1/(2Q)
        where xi is the fraction of critical damping

    Returns
    -------
    PSA : np.ndarray
        Pseudo-spectral acceleration ordinates
    PSV : np.ndarray
        Pseudo-spectral velocity ordinates
    SD : np.ndarray
        spectral displacement ordinates

    """
    if periods is None:
        periods = np.array([
            0.01,0.02,0.022,0.025,0.029,0.03,0.032,0.035,0.036,
            0.04,0.042,0.044,0.045,0.046,0.048,0.05,0.055,0.06,0.065,0.067,0.07,
            0.075,0.08,0.085,0.09,0.095,0.1,0.11,0.12,0.125,0.13,0.133,0.14,0.15,
            0.16,0.17,0.18,0.19,0.2,0.22,0.24,0.25,0.26,0.28,0.29,0.3,0.32,0.34,
            0.35,0.36,0.38,0.4,0.42,0.44,0.45,0.46,0.48,0.5,0.55,0.6,0.65,0.667,
            0.7,0.75,0.8,0.85,0.9,0.95,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,
            2,2.2,2.4,2.5,2.6,2.8,3,3.2,3.4,3.5,3.6,3.8,4,4.2,4.4,4.6,4.8,5,7.5,10])

    n = len(periods)
    displ_max = np.empty(n, dtype='float64')
    veloc_max = np.empty(n, dtype='float64')
    absacc_max = np.empty(n, dtype='float64')
    foverm_max = np.empty(n, dtype='float64')

    #vel = data[0].integrate(method='cumtrapz')
    #dist = data[0].integrate(method='cumtrapz')

    """ Spectral solution """

    xi = 1 / (2 * Q)
    omegans = 2 * np.pi / periods  # Angular frequency
    C = 2 * xi * omegans # Two time of critical damping and angular freq.
    K = omegans ** 2
    A = np.zeros((n, 2, 2), dtype='float64')
    A[:, 0, 1] = 1.
    A[:, 1, 0] = -K
    A[:, 1, 1] = -C
    Ae = expm(A*dt)
    #AeT = np.transpose(expm(A*dt), axes=[0, 2, 1])
    eye2 = np.eye(2, dtype=int)
    Ae_eye2 = Ae - eye2
    pinvA = pinv(A)
    temp2 = np.einsum('nij,njl->nil', Ae_eye2, pinvA)

    zero_one = np.array([[0.0], [1.0]])
    AeB = np.dot(temp2, zero_one)

    nacc = len(acceleration)
    ks = np.arange(1, nacc)
    AeB_accels2 = np.einsum('nij,k->nkij', AeB, acceleration[ks])
    #AeB_accels3 = np.einsum('nij,k->nkji', AeB, acceleration[ks])
    for i, omegan, Aei, AeB_accels in zip(count(), omegans, Ae, AeB_accels2):
        #assert np.allclose(Aei.T, AeTi)
        y = np.zeros((2, nacc))
        yi = y[:, 0].reshape(2, 1)
        for k, AeB_accel in zip(ks, AeB_accels):
            # ix1 = ixj @ jx1
            # 1xi = (ixj @ jx1)^T
            aai = Aei @ yi
            #AAi = yi.T @ Aei.T
            #AAi = Aei.T @ yi.T
            AA = np.reshape(aai, (2,1))
            yi = np.add(AA, AeB_accel)
            y[:,k] = yi.ravel()
        displ = y[0,:].ravel()                # Relative displacement vector (cm)
        veloc = y[1,:].ravel()                # Relative velocity (cm/s)
        foverm = (omegan**2) * displ          # Lateral resisting force over mass (cm/s^2)
        absacc = -2*xi*omegan*veloc - foverm  # Absolute acceleration from equilibrium (cm/s^2)

        # Extract peak values
        displ_max[i] = max(abs(displ))    # Spectral relative displacement (cm)
        veloc_max[i] = max(abs(veloc))    # Spectral relative velocity (cm/s)
        absacc_max[i] = max(abs(absacc))  # Spectral absolute acceleration (cm/s^2)
        foverm_max[i] = max(abs(foverm))  # Spectral value of lateral resisting force over mass (cm/s^2)

    pseudo_acc_max = displ_max*K           # Pseudo spectral acceleration (cm/s^2)
    pseudo_veloc_max = displ_max*omegans   # Pseudo spectral velocity (cm/s)

    PSA = pseudo_acc_max    # PSA (cm/s2)
    PSV = pseudo_veloc_max  # PSV (cm/s)
    SD = displ_max          # SD  (cm)
    return PSA, PSV, SD

def plotting(PSA: np.ndarray,
             PSV: np.ndarray,
             SD: np.ndarray,
             periods: np.ndarray,
             length_unit: str,
             saving: str='show',
             logplot: bool=True,
             title: str='') -> plt.Figure:
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)

    if logplot:
        ax1.loglog(periods, PSA)
        ax2.loglog(periods, PSV)
        ax3.loglog(periods, SD)
    else:
        ax1.plot(periods, PSA)
        ax2.plot(periods, PSV)
        ax3.plot(periods, SD)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    accel_unit = f'{length_unit}/s^{2}'
    velocity_unit = f'{length_unit}/s'

    title2 = title + '\n' if title else ''
    title2 += 'Pseudo spectral acceleration ($\mathregular{' + accel_unit + '}$)'

    ax1.set_title(title2)
    ax2.set_title(f'Pseudo spectral velocity ({velocity_unit})')
    ax3.set_title(f'Spectral displacement ({length_unit})')

    ax1.set_ylabel('Amplitude', ha='center', va='center', rotation='vertical')
    ax2.set_ylabel('Amplitude', ha='center', va='center', rotation='vertical')
    ax3.set_ylabel('Amplitude', ha='center', va='center', rotation='vertical')

    # Set common labels
    fig.text(0.5, 0.01, 'Periods (s)', ha='center', va='center')
    #fig.text(0.015, 0.5, 'Amplitude ($\mathregular{cm/s^{2}}$)', ha='center', va='center', rotation='vertical')

    plt.tight_layout()

    cwd = os.getcwd()
    if title:
        png_filename = os.path.join(cwd,  f'{title}_spectra.png')
    else:
        png_filename = os.path.join(cwd, 'spectra.png')

    if saving == 'show':
        plt.show()
    elif saving == 'save':
        plt.savefig(png_filename, dpi=300)
    elif saving == 'show_save':
        plt.savefig(png_filename, dpi=300)
        plt.show()

    return fig

# https://www.vibrationdata.com/software.htm
# https://community.sw.siemens.com/s/article/dynamic-stiffness-compliance-mobility-and-more
# https://www.vibrationdata.com/tutorials_alt/frf.pdf
def compliance(force, displacement):
    """1/k at low end; 1/(omega^2*m) at high end"""
    return displacement / force
def dynamic_stiffness(force, displacement):
    """inverse of compliance terms"""
    return force / displacement

def mobility(force, velocity):
    """omega/k at low end; 1/(omega*m) at high end"""
    return velocity / force
def mechanical_impedance(force, velocity):
    return force / velocity

def accelerance(force, acceleration):
    """omega^2/k at low end; 1/m at high end"""
    return acceleration / force
def dynamic_mass(force, acceleration):
    """inverse of accelerance terms"""
    return force / acceleration

inertance = accelerance
apparent_mass = dynamic_mass
receptance = compliance
admittance = compliance
