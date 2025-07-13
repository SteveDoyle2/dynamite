import numpy as np
import matplotlib.pyplot as plt


def get_fstar(nmode: int,
              mach: float,
              LD: float,
              gamma: float=1.4) -> float:
    inner = 1 + (gamma - 1) / 2 * mach**2
    denom = mach / (inner ** 0.5 + 1.75)
    fstar = (nmode - 0.25) / denom
    return fstar

def sonic_fatigue(LD: float):
    gamma = 1.4
    dmach = 0.05
    mach = np.linspace(0.6, 1.3+dmach, num=51)
    fstar1 = get_fstar(1, mach, LD, gamma=gamma)
    fstar2 = get_fstar(2, mach, LD, gamma=gamma)
    fstar3 = get_fstar(3, mach, LD, gamma=gamma)

    # 4.3.2-2
    log_p2_max_q = 9.0 - 3.3 * LD + 20. * np.log10(-mach**2 + 2*mach - 0.7)

    # 4.3.2-3
    log_p1_max_q = log_p2_max_q - 2 * LD**2 + 26*LD - 86

    # 4.3.2-4
    if LD < 4.5:
        log_p3_max_q = log_p2_max_q - 11.0
    else:
        log_p3_max_q = log_p2_max_q

    xoL = np.linspace(0.1, 1., num=101)
    #--------------------
    fig1 = plt.figure(1)
    # fig2 = plt.figure(1)
    # fig3 = plt.figure(1)
    ax1 = fig1.gca()
    # ax2 = fig2.gca()
    # ax3 = fig3.gca()
    ax1.plot(xoL, log_p1_max_q, label='Mode 1')
    ax1.plot(xoL, log_p2_max_q, label='Mode 2')
    ax1.plot(xoL, log_p3_max_q, label='Mode 3')
    ax1.set_xlabel('x/L')
    ax1.set_ylabel('log(pN,max/q)')
    ax1.set_xlim([0.0, 1.0])
    plt.show()
    #--------------------


    alpha1 = 3.5
    alpha2 = 6.3
    alpha3 = 10.0
    cos1_axL = np.abs(np.cos(alpha1 * xoL))
    cos2_axL = np.abs(np.cos(alpha2 * xoL))
    cos3_axL = np.abs(np.cos(alpha3 * xoL))
    log_p1_q_xoL1 = log_pn_max_q - 10 * (1.0 + (0.33 * LD - 0.60) * (1 - xoL) - cos1_axL)
    U = 1.
    L = 1.
    freq = L / U * fstar


def main():
    L = 20.
    D = 5.
    LD = L / D
    sonic_fatigue(LD=4.0, )

if __name__ == '__main__':
    main()
