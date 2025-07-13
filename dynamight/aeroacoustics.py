import numpy as np
import matplotlib.pyplot as plt


def get_fstar(nmode: int,
              mach: float,
              LD: float,
              gamma: float=1.4) -> float:
    inner = 1 + (gamma - 1) / 2 * mach**2
    denom = mach / (inner ** 0.5) + 1.75
    fstar = (nmode - 0.25) / denom
    return fstar


def get_log_pn_max(mach: float, LD: float):
    # 4.3.2-2
    # it's gotta be 9 because ax2 is right
    # log_p2_max_q = 8.0 - 3.3 * LD + 20. * np.log10(-mach**2 + 2*mach - 0.7)
    log_p2_max_q = 9.0 - 3.3 * LD + 20. * np.log10(-mach**2 + 2*mach - 0.7)

    # 4.3.2-3
    log_p1_max_q = log_p2_max_q - 2 * LD**2 + 26*LD - 86

    # 4.3.2-4
    if LD < 4.5:
        log_p3_max_q = log_p2_max_q - 11.0
    else:
        log_p3_max_q = log_p2_max_q
    return log_p1_max_q, log_p2_max_q, log_p3_max_q


def sonic_fatigue(LD: float):
    gamma = 1.4
    dmach = 0.05

    if 1:
        machs = np.linspace(0.6, 1.3+dmach, num=51)
        fstar1s = get_fstar(1, machs, LD, gamma=gamma)
        fstar2s = get_fstar(2, machs, LD, gamma=gamma)
        fstar3s = get_fstar(3, machs, LD, gamma=gamma)
        fig1 = plt.figure(1)
        fig2 = plt.figure(2)
        # fig3 = plt.figure(3)
        ax1 = fig1.gca()
        ax2 = fig2.gca()
        # ax3 = fig3.gca()
        ax1.plot(machs, fstar1s, label='Mode 1')
        ax1.plot(machs, fstar2s, label='Mode 2')
        ax1.plot(machs, fstar3s, label='Mode 3')
        ax1.legend()
        ax1.set_xlabel('Mach')
        ax1.set_ylabel('freq*')
        ax1.set_xlim([0.6, 1.3])
        ax1.set_ylim([0.2, 1.2])
        ax1.grid(True)
        fig1.suptitle('Seems good')
        log_p1_max_qs, log_p2_max_qs, log_p3_max_qs = get_log_pn_max(
            machs, LD)
        max1 = log_p1_max_qs.max()
        max2 = log_p2_max_qs.max()
        max3 = log_p3_max_qs.max()
        imax1 = np.where(log_p1_max_qs == max1)[0]
        imax2 = np.where(log_p2_max_qs == max2)[0]
        imax3 = np.where(log_p3_max_qs == max3)[0]
        ax2.plot(machs, log_p1_max_qs, label=f'Mode 1: {max1:g}', color='C0')
        ax2.plot(machs, log_p2_max_qs, label=f'Mode 2: {max2:g}', color='C1')
        ax2.plot(machs, log_p3_max_qs, label=f'Mode 3: {max3:g}', color='C2')
        ax2.plot(machs[imax1], log_p3_max_qs[imax1], marker='o', color='C0')
        ax2.plot(machs[imax2], log_p1_max_qs[imax2], marker='o', color='C1')
        ax2.plot(machs[imax3], log_p2_max_qs[imax3], marker='o', color='C2')
        ax2.legend()
        ax2.set_xlabel('Mach')
        ax2.set_ylabel('20 log(pN,max/q)')
        ax2.set_xlim([0.6, 1.3])
        ax2.set_ylim([-36., -12.])
        ax2.set_yticks(np.arange(-36, -12. + 1, 2.0))
        ax2.grid(True)
        fig2.suptitle('Seems good')


    #================
    # mach = np.linspace(0.6, 1.3+dmach, num=51)
    mach = 0.6
    fstar1 = get_fstar(1, mach, LD, gamma=gamma)
    fstar2 = get_fstar(2, mach, LD, gamma=gamma)
    fstar3 = get_fstar(3, mach, LD, gamma=gamma)

    log_p1_max_q, log_p2_max_q, log_p3_max_q = get_log_pn_max(
        mach, LD)

    xoL = np.linspace(0., 1., num=501)

    #--------------------
    fig11 = plt.figure(11)
    fig12 = plt.figure(12)
    fig13 = plt.figure(13)
    ax11 = fig11.gca()
    ax12 = fig12.gca()
    ax13 = fig13.gca()
    ax11.grid(True)
    ax11.set_xlabel('x/L')
    ax11.set_ylabel('$log(pN/q)$')
    ax11.set_xlim([0.0, 1.0])
    ax11.set_ylim([-50.0, -20.0])
    ax2.set_yticks(np.arange(-50, -20. + 1, 5.0))

    ax12.grid(True)
    ax12.set_xlabel('x/L')
    ax12.set_ylabel('$log(pN/q) - log(pN,max/q)$')
    ax12.set_xlim([0.0, 1.0])

    ax13.grid(True)
    ax13.set_xlabel('x/L')
    ax13.set_ylabel('$log(pb/q) - log(p2,max/q)$')
    ax13.set_xlim([0.0, 1.0])
    ax13.set_ylim([-25., -15.])
    #--------------------

    # 4.3.2-5
    alpha1 = 3.5
    alpha2 = 6.3
    alpha3 = 10.0
    cos1_axL = np.abs(np.cos(alpha1 * xoL))
    cos2_axL = np.abs(np.cos(alpha2 * xoL))
    cos3_axL = np.abs(np.cos(alpha3 * xoL))
    LD33 = (0.33 * LD - 0.60)
    omxl = 1 - xoL
    assert len(omxl) == cos1_axL.size
    assert isinstance(log_p1_max_q, float)
    log_p1_q_xoL = log_p1_max_q - 10 * (1.0 + LD33 * omxl - cos1_axL)
    log_p2_q_xoL = log_p2_max_q - 10 * (1.0 + LD33 * omxl - cos2_axL)
    log_p3_q_xoL = log_p3_max_q - 10 * (1.0 + LD33 * (1 - xoL) - cos3_axL)
    d05 = (log_p1_q_xoL - 0.5)
    d10 = (log_p1_q_xoL - 1.0)
    val05 = np.where(d05 == d05.min())[0]
    val10 = np.where(d10 == d10.min())[0]

    d05 = (xoL - 0.5)
    d10 = (xoL - 1.0)
    i05 = np.where(d05 == d05.min())[0][0]
    i10 = np.where(d10 == d10.min())[0][0]
    ax11.plot(xoL, log_p1_q_xoL, label=f'Mode 1 (x={log_p1_q_xoL[i05]:g}; x=1.0: {log_p1_q_xoL[i10]:g})')
    ax11.plot(xoL, log_p2_q_xoL, label=f'Mode 2 (x={log_p2_q_xoL[i05]:g}; x=1.0: {log_p2_q_xoL[i10]:g})')
    ax11.plot(xoL, log_p3_q_xoL, label=f'Mode 3 (x=0.5: x={log_p3_q_xoL[i05]:g}; x=1.0: {log_p3_q_xoL[i10]:g})')
    ax11.legend()
    ax12.plot(xoL, log_p1_q_xoL - log_p1_max_q, label='Mode 1')
    ax12.plot(xoL, log_p2_q_xoL - log_p2_max_q, label='Mode 2')
    ax12.plot(xoL, log_p3_q_xoL - log_p3_max_q, label='Mode 3')
    ax12.set_ylim([None, 0.])
    ax12.legend()

    # 4.3.2-6
    log_pb_q = log_p2_max_q + 3.3*LD - 28 + 3*(1-LD)*(1-xoL)

    ax13.plot(xoL, log_pb_q - log_p2_max_q, label='broadband')
    ax13.legend()
    fig13.suptitle(f'Wrong; p2_max={log_p2_max_q:g}')
    plt.show()
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
