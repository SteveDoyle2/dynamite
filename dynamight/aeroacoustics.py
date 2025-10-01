"""
sonic fatigue: https://apps.dtic.mil/sti/pdfs/ADB004600.pdf

per Shaw (7,9) and Smith (8)
7: Shaw https://apps.dtic.mil/sti/pdfs/ADA377359.pdf

Plenovich: https://ntrs.nasa.gov/api/citations/19980008833/downloads/19980008833.pdf
FPL = SPL + 20*log10(2.9e-9 psi/qinf)
"""
import numpy as np
import matplotlib.pyplot as plt
from pyNastran.utils.atmosphere import (
    atm_dynamic_pressure,
    atm_density,
    atm_speed_of_sound,
    convert_velocity,
)

keas_to_fts = 1.68781
psf_to_pa = 47.880208


def strarray(myarray_str: str) -> np.ndarray:
    #print(myarray_str)
    lines = myarray_str.split('\n')
    slines = [np.array(line.strip(',').split(','), dtype='float64') for line in lines]
    array = np.vstack(slines, dtype='float64')
    xs = array[:, 0]
    ix = np.argsort(xs)

    assert array.shape[1] == 2, array.shape
    return array[ix, :]

# plot1_data_x1 = strarray("""
# 0.65, 0.315327420165442
# 0.7500000000000001, 0.7016400211896062
# 0.7500000000000001, 0.3002762798657992
# 0.8000000000000002, 0.3002762798657992
# 0.8500000000000002, 0.684916531967781
# 0.9000000000000004, 0.6715377405903208
# 0.9000000000000004, 0.29024218633270404
# 0.9500000000000003, 1.0494885970035723
# 0.9500000000000003, 0.6681930427459558
# 0.9500000000000003, 0.2818804417217914
# 1.0000000000000002, 1.0277480610151994
# 1.0500000000000003, 1.0160416185599217
# 1.0500000000000003, 0.6481248556797653
# 1.0500000000000003, 0.27184634818869624
# 1.1000000000000003, 1.000990478260279
# 1.1000000000000003, 0.6414354599910351
# 1.1000000000000003, 0.27184634818869624
# 1.1500000000000004, 0.9876116868828188
# 1.1500000000000004, 0.6330737153801227
# 1.1500000000000004, 0.26850165034433116
# 1.2000000000000004, 0.6213672729248448
# 1.2500000000000004, 0.9675434998166283
# 1.2500000000000004, 0.6113331793917497
# 1.2500000000000004, 0.258467556811236
# 0.700570884871551, 1.1391206313416007
# 0.6013320647002854, 1.1824126268320179
# 0.649952426260704, 1.1607666290868093
# 0.7498572787821123, 1.1188275084554677
# 0.8011417697431018, 1.0985343855693346
# 0.8504281636536632, 1.0822998872604281
# 0.9023786869647955, 1.0647125140924463
# 1.2007611798287345, 0.9794813979706876
# 1.3006660323501427, 0.9564825253664033
# 1.3006660323501427, 0.6020293122886132
# 1.0002854424357754, 0.6547914317925589
# 0.7998097050428163, 0.6980834272829761
# 0.7498572787821123, 0.710259301014656
# 0.7012369172216937, 0.7251409244644869
# 0.6506184586108468, 0.7373167981961667
# 0.6013320647002854, 0.7549041713641486
# 0.6006660323501427, 0.32333709131905286
# 0.6999048525214082, 0.3098083427282974
# 0.8510941960038059, 0.2908680947012401
# 1.0009514747859183, 0.27869222096956026
# 1.2014272121788774, 0.26381059751972935
# 1.3006660323501427, 0.25704622322435156""")


plot2_data_mode1 = strarray("""0.6218191049933749, -34.26940848645762
0.6611497448966431, -32.77663220949843
0.7984549200065183, -29.688493260945798
0.8371401082319196, -29.263517177222383
0.9205039645486296, -28.609088767275683
0.9594070974964275, -28.682570874066116
1.0761164963398215, -28.620841155922296
1.1594803526565314, -29.182758784650744
0.9972972972972973, -28.538555691554468
0.7462577962577963, -30.56548347613219
0.6975051975051976, -31.62301101591187
0.5978170478170478, -35.23623011015911
1.1966735966735969, -29.71358629130967
1.3000000000000003, -31.56425948592411
1.2454261954261954, -30.44798041615667
1.038773388773389, -28.509179926560588
1.120997920997921, -28.861689106487148""")

plot2_data_mode3 = strarray("""0.6975051975051976, -28.62668298653611
0.7404365904365905, -27.65728274173807
0.7586278586278586, -27.392900856793148
0.7746361746361747, -27.099143206854347
0.797920997920998, -26.746634026927786
0.8408523908523908, -26.217870257037944
0.8677754677754679, -25.982864137086906
0.8983367983367985, -25.806609547123625
0.9296257796257796, -25.689106487148106
0.9594594594594595, -25.542227662178703
0.9972972972972973, -25.454100367197064
1.0424116424116425, -25.571603427172583
1.0795218295218296, -25.689106487148106
1.0984407484407486, -25.806609547123625
1.13991683991684, -26.070991432068546
1.1821205821205822, -26.51162790697675
1.1981288981288982, -26.687882496940027
1.2425155925155926, -27.392900856793148
1.2759875259875262, -28.00979192166463
1.2941787941787943, -28.42105263157895
1.3000000000000003, -28.538555691554468
0.8182952182952183, -26.51162790697675""")
plot2_data_mode2 = strarray("""0.5985446985446986, -21.194614443084458
0.6385654885654886, -19.402692778457777
0.6814968814968815, -18.02203182374541
0.6975051975051976, -17.610771113831092
0.7331600831600832, -16.788249694002452
0.7753638253638254, -16.02447980416157
0.7964656964656965, -15.671970624235009
0.8546777546777546, -15.084455324357409
0.8968814968814969, -14.731946144430848
0.9558212058212059, -14.555691554467568
0.995841995841996, -14.496940024479807
1.051871101871102, -14.555691554467568
1.096985446985447, -14.731946144430848
1.1508316008316009, -15.143206854345168
1.1966735966735969, -15.671970624235009
1.2163201663201664, -16.02447980416157
1.2621621621621624, -16.758873929008573
1.2978170478170479, -17.552019583843332""")

plot3_mode2 = strarray("""0.006355932203389841, -7.001394700139468
0.060734463276836154, -7.336122733612271
0.09322033898305085, -8.033472803347278
0.10240112994350284, -8.368200836820083
0.16242937853107345, -10.571827057182704
0.2012711864406779, -12.635983263598325
0.22669491525423727, -14.281729428172941
0.2450564971751412, -15.676429567642955
0.2655367231638417, -14.19804741980474
0.29519774011299427, -12.10599721059972
0.3213276836158192, -10.488145048814502
0.35310734463276827, -8.619246861924683
0.4018361581920903, -6.108786610878659
0.4399717514124293, -4.658298465829845
0.4858757062146891, -3.6820083682008358
0.5134180790960451, -3.542538354253833
0.5663841807909603, -3.960948396094837
0.5967514124293785, -4.686192468619245
0.6341807909604518, -5.941422594142257
0.6772598870056495, -7.894002789400277
0.7055084745762711, -9.539748953974895
0.74364406779661, -12.13389121338912
0.7704802259887004, -10.264993026499301
0.8036723163841807, -8.11715481171548
0.8418079096045197, -5.774058577405856
0.8785310734463274, -3.737796373779636
0.918785310734463, -1.8967921896792168
0.955508474576271, -0.8368200836820063
0.9851694915254237, -0.306834030683401
1.001412429378531, -0.13947001394699943
0.13276836158192087, -9.31659693165969
0.03954802259887005, -7.085076708507669
0.18008474576271186, -11.408647140864712
0.28177966101694907, -13.026499302649928
0.33757062146892647, -9.428172942817293
0.41878531073446323, -5.411436541143653
0.7259887005649716, -10.822873082287307
0.8241525423728812, -6.806136680613665""")


def fstar_to_freq(fstar: np.ndarray,
                  velocity: float, length: float) -> np.ndarray:
    # fstar = num/denom
    freq = fstar * velocity / length
    return freq


def get_fstar(nmode: int,
              mach: float,
              gamma: float=1.4) -> float:
    inner = 1 + (gamma - 1) / 2 * mach**2
    num = (nmode - 0.25)
    denom = mach / (inner ** 0.5) + 1.75
    fstar = num / denom
    return fstar


def get_log_pn_max(mach: float, length_depth: float):
    # 4.3.2-2
    # it's gotta be 9 because ax2 is right
    # log_p2_max_q = 8.0 - 3.3 * LD + 20. * np.log10(-mach**2 + 2*mach - 0.7)
    log_p2_max_q = 9.0 - 3.3 * length_depth + 20. * np.log10(-mach**2 + 2*mach - 0.7)

    # 4.3.2-3
    log_p1_max_q = log_p2_max_q - 2 * length_depth**2 + 26*length_depth - 86

    # 4.3.2-4
    if length_depth < 4.5:
        log_p3_max_q = log_p2_max_q - 11.0
    else:
        log_p3_max_q = log_p2_max_q
    return log_p1_max_q, log_p2_max_q, log_p3_max_q


def make_strouhal_plot(fig_id: int, length_depth: float=4.0):
    # stx = f*L/U
    stx = np.logspace(0.01, 100., num=101)
    LD = 4.
    L = 20.
    Ukeas = 200.  # KEAS
    Ufts = convert_velocity(Ukeas, 'knots', 'ft/s')
    # D = L / LD

    freq = stx/L*Ufts

    asos_fts = atm_speed_of_sound(0., velocity_units='ft/s')
    mach = Ufts / asos_fts
    log_p1_max_qs, log_p2_max_qs, log_p3_max_qs = get_log_pn_max(mach, LD)
    xol = np.linspace(0.01, 1., num=101)
    p_broadband = get_broadband_pressure(xol, log_p2_max_qs, length_depth=length_depth)

    fig, ax = get_fig_ax(fig_id)
    ax.set_ylim([-40, 10])
    ax.grid(True)
    ax.set_ylabel('20 log$(p/q)$')
    ax.set_xlabel('Strouhal Number, St=$f L/D$')
    fig.suptitle('4.3.2-5: Broadband Cavity Noise Spectrum Shape')
    ax.plot(stx, p_broadband)
    #----------------------------------------------------

    # fig5 = plt.figure(5)
    # ax5 = fig5.gca()
    # mach = np.linspace(0.452, 1., num=101)
    # resp = -mach ** 2 + 2 * mach - 0.7
    # ax5.plot(mach, resp)
    # ax5.grid(True)
    # plt.show()


def get_fig_ax(fig: plt.Figure | int | plt.Axes) -> [plt.Figure, plt.Axes]:
    if isinstance(fig, int):
        figure = plt.figure(fig)
        ax = figure.gca()
    elif isinstance(fig, plt.Figure):
        figure = fig
        ax = figure.gca()
    elif isinstance(fig, plt.Axes):
        ax = fig
        figure = ax.gcf()
    else:
        raise TypeError(fig)
    return figure, ax


def vel_plot(fig_id: plt.Figure | int,
             alts: np.ndarray, rhos: np.ndarray,
             vels: np.ndarray,
             use_keas: bool=True):
    fig, ax = get_fig_ax(fig_id)
    alts2 = alts / 1000
    ax_spl = ax
    plot_q = False
    if plot_q:
        ax_q = ax.twinx()
    vel_unit = 'KEAS' if use_keas else 'ft/s'

    print(f'vel_plot keas={vels}')
    assert len(vels) > 0, vels
    for vel in vels:
        if use_keas:
            qs_psf = 0.5 * rhos * (vel * keas_to_fts) ** 2
        else:
            qs_psf = 0.5 * rhos * vel ** 2
        qs_Pa = qs_psf * psf_to_pa
        #qs_psi = qs_psf / 144.
        #print(f'spl.max = {spl.max()}')
        # spl = 20 * np.log10(qs_psi / 20 * 1e-6) # per sonic_fatigue
        spl = 20 * np.log10(qs_Pa / (20 * 10**(-6)))  # per sonic_fatigue
        label = rf'{vel:g} {vel_unit}; $\Delta$SPL=[{spl.min():.0f}, {spl.max():.0f}]'
        if plot_q:
            ax_q.semilogy(alts2, qs_psf, label=label)
            # ax_spl.plot(alts, spl)
            ax_spl.plot(alts2, spl)
        else:
            ax_spl.plot(alts2, spl, label=label)
    ax_spl.set_ylabel(r'$\Delta$SPL; 20 log$(q / 20 μPa)$ (dB)')
    ax.set_xlim([0., 65.])  # alts2
    ax.xaxis.set_inverted(True)
    if plot_q:
        ax_q.legend()
    else:
        ax.legend()
    ax.set_xlabel('Altitude (kft)')  # alts2
    # ax.set_xlabel('Altitude (ft)')  # alts
    # ax.set_xlim([-0., 65000.])  # alts
    spl_ticks = np.arange(130, 170+1, 5.)
    alt_ticks = np.arange(0., 65+1., 5.)

    ax_spl.set_xticks(alt_ticks)
    # ax_q.set_xticks(alt_ticks)
    # ax_spl.set_yticks(spl_ticks)
    # ax_spl.set_ylim((spl_ticks.min(), spl_ticks.max()))

    ax.grid(True)
    if plot_q:
        ax_q.set_ylabel('q (psf)')
        # ax_q.set_ylim([130, 210])

    # fig = ax.get_figure()
    fig.suptitle(r'Altitude vs. $\Delta$SPL, q; Design Chart for' +
                 '\nDetermining Reference "q" Decibel Level')


def plenovich_spl_plot(fig_id: int,
                       alts: np.ndarray, rhos: np.ndarray,
                       vels: np.ndarray, fpl: float=0.0):
    """Plenovich"""
    fig, ax = get_fig_ax(fig_id)
    plot_q = False
    ax_spl = ax
    if plot_q:
        ax_q = ax.twinx()
    for vel in vels:
        qs_psf = 0.5 * rhos * (vel * keas_to_fts) ** 2
        qs_psi = qs_psf / 144.
        dspl = 20 * np.log10(2.9e-9 / qs_psi)  # per Plenovich
        spl = fpl - dspl
        # print(f'spl.max = {spl.max()}')
        label = f'{vel:g} KEAS'
        if plot_q:
            ax_q.semilogy(alts, qs_psf, label=label)
            ax_spl.plot(alts, spl)
        else:
            ax_spl.plot(alts, spl, label=label)
    if plot_q:
        ax_q.legend()
    else:
        ax_spl.legend()
    ax_spl.set_ylabel('SPL (dB)')
    # ax.set_xlim([-65., 0.])  # alts2
    # ax.set_xlabel('-Altitude (kft)')  # alts2
    ax.set_xlabel('Altitude (ft)')  # alts
    ax.set_xlim([-0., 65000.])  # alts
    ax.grid(True)
    if plot_q:
        ax_q.set_ylabel('q (psf)')
    fig = ax.get_figure()
    fig.suptitle(f'SPL vs. Altitude (FPL={fpl} dB)')


def _cavity_pressure_cos_alpha(xol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha1 = 3.5
    alpha2 = 6.3
    alpha3 = 10.0
    cos1_axl = np.abs(np.cos(alpha1 * xol))
    cos2_axl = np.abs(np.cos(alpha2 * xol))
    cos3_axl = np.abs(np.cos(alpha3 * xol))
    return cos1_axl, cos2_axl, cos3_axl


def get_cavity_pressure(log_p1_max_q: float, log_p2_max_q: float, log_p3_max_q: float,
                        xol: np.ndarray, length_depth: float=4.0):
    # 4.3.2-5
    cos1_axl, cos2_axl, cos3_axl = _cavity_pressure_cos_alpha(xol)

    LD33 = (0.33 * length_depth - 0.60)
    omxl = 1 - xol
    assert len(omxl) == cos1_axl.size
    assert isinstance(log_p1_max_q, float)
    # log_p1_q_xol = log_p1_max_q - 10 * (1.0 + LD33 * (1 - xoL) - cos1_axL)
    # log_p2_q_xol = log_p2_max_q - 10 * (1.0 + LD33 * (1 - xoL) - cos2_axL)
    # log_p3_q_xol = log_p3_max_q - 10 * (1.0 + LD33 * (1 - xoL) - cos3_axL)
    log_p1_q_xol = log_p1_max_q - 10 * (1.72 - 0.7 * xol - cos1_axl)
    log_p2_q_xol = log_p2_max_q - 10 * (1.72 - 0.7 * xol - cos2_axl)
    log_p3_q_xol = log_p3_max_q - 10 * (1.72 - 0.7 * xol - cos3_axl)

    log_p2_q_xol_min = log_p2_max_q - 10 * (1.72 - 0.7 * xol)
    log_p2_q_xol_max = log_p2_max_q - 10 * (1.72 - 0.7 * xol - 1)

    return log_p1_q_xol, log_p2_q_xol, log_p3_q_xol, log_p2_q_xol_min, log_p2_q_xol_max


def plot_cavity_pressure(fig_id: int,
                         xol: np.ndarray,
                         log_p1_q_xol: np.ndarray,
                         log_p2_q_xol: np.ndarray,
                         log_p3_q_xol: np.ndarray,
                         log_p2_q_xol_min: np.ndarray,
                         log_p2_q_xol_max: np.ndarray,
                         ) -> None:
    fig, ax = get_fig_ax(fig_id)

    ax.grid(True)
    ax.set_xlabel('x/L')
    ax.set_ylabel('20 log$(p_N/q)$')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-50.0, -20.0])
    ax.set_yticks(np.arange(-50, -20. + 1, 5.0))

    # d05 = (log_p1_q_xoL - 0.5)
    # d10 = (log_p1_q_xoL - 1.0)
    # val05 = np.where(d05 == d05.min())[0]
    # val10 = np.where(d10 == d10.min())[0]

    d05 = (xol - 0.5)
    d10 = (xol - 1.0)
    i05 = np.where(d05 == d05.min())[0][0]
    i10 = np.where(d10 == d10.min())[0][0]

    ax.plot(xol, log_p1_q_xol, label=f'Mode 1 (x=0.5: {log_p1_q_xol[i05]:3f}; x=1.0: {log_p1_q_xol[i10]:3f})', color='C0')
    ax.plot(xol, log_p2_q_xol, label=f'Mode 2 (x=0.5: {log_p2_q_xol[i05]:3f}; x=1.0: {log_p2_q_xol[i10]:3f})', color='C1')
    ax.plot(xol, log_p3_q_xol, label=f'Mode 3 (x=0.5: {log_p3_q_xol[i05]:3f}; x=1.0: {log_p3_q_xol[i10]:3f})', color='C2')
    ax.plot(xol, log_p2_q_xol_min, linestyle='--', color='C1')
    ax.plot(xol, log_p2_q_xol_max, linestyle='--', color='C1')
    ax.legend()


def get_broadband_pressure(xol: np.ndarray,
                           log_p2_max_q: float, length_depth: float=4.0):
    """4.3.2-6"""
    # log_p2_max_q is the right term (mabye wrong value) can't be x/L
    log_pb_q = log_p2_max_q + 3.3*length_depth - 28 + 3*(1-length_depth)*(1-xol)
    return log_pb_q


def sonic_fatigue(vels: np.ndarray,
                  length: float, length_depth: float, fpl: float=0.0):
    """
    sonic fatigue: https://apps.dtic.mil/sti/pdfs/ADB004600.pdf

    per Shaw (7,9) and Smith (8)
    7: Shaw https://apps.dtic.mil/sti/pdfs/ADA377359.pdf

    Plenovich: https://ntrs.nasa.gov/api/citations/19980008833/downloads/19980008833.pdf
    FPL = SPL + 20*log10(2.9e-9 psi/qinf)
    """
    gamma = 1.4
    dmach = 0.05
    machi = 0.9

    if 1:
        machs = np.linspace(0.6, 1.3+dmach, num=101)
        fstar1s = get_fstar(1, machs, gamma=gamma)
        fstar2s = get_fstar(2, machs, gamma=gamma)
        fstar3s = get_fstar(3, machs, gamma=gamma)
        fstars_dict = {
            1: fstar1s,
            2: fstar2s,
            3: fstar3s,
        }

        alts = np.linspace(0., 65000+1, num=660)
        alts2 = alts / -1000
        rhos = np.array([
            atm_density(alt, density_units='slug/ft^3')
            for alt in alts])
        # ft/s
        # vels = [225.]
        plenovich_spl_plot(1, alts, rhos, vels, fpl=fpl)

        log_p1_max_qs, log_p2_max_qs, log_p3_max_qs = get_log_pn_max(
            machs, length_depth)

        asos_fts = atm_speed_of_sound(0., velocity_units='ft/s')
        # U = convert_velocity(230., 'knots', 'ft/s')
        U = machs * asos_fts
        freq_dict = {
            1: fstar_to_freq(fstar1s, U, length),
            2: fstar_to_freq(fstar2s, U, length),
            3: fstar_to_freq(fstar3s, U, length),
        }

        plot_normalized_frequency(
            2, fstars_dict, machs)
        plot_frequency(
            3, freq_dict, machs, uvelocity=U)

        make_strouhal_plot(3, length_depth=length_depth)
        # return

        plot_peak_variation_with_mach(
            5,
            machs,
            log_p1_max_qs, log_p2_max_qs, log_p3_max_qs,
            length_depth=length_depth,
        )
        # plt.show()

    #================
    # mach = np.linspace(0.6, 1.3+dmach, num=51)
    # fstar1 = get_fstar(1, machi, gamma=gamma)
    # fstar2 = get_fstar(2, machi, gamma=gamma)
    # fstar3 = get_fstar(3, machi, gamma=gamma)
    log_p1_max_q, log_p2_max_q, log_p3_max_q = get_log_pn_max(
        machi, length_depth)

    xol = np.linspace(0., 1., num=10001)

    #--------------------
    # if 1:
    #     fig15 = plt.figure(15)
    #     ax15 = fig15.gca()
    #     ax15.grid(True)
    #     ax15.set_xlabel('x/L')
    #     ax15.set_ylabel('SPL')
    #     ax15.set_xlim([0.0, 1.0])

    #--------------------
    cos1_axl, cos2_axl, cos3_axl = _cavity_pressure_cos_alpha(xol)
    plot_cos_alpha(10, xol, cos1_axl, cos2_axl, cos3_axl)

    log_p1_q_xoL, log_p2_q_xoL, log_p3_q_xoL, log_p2_q_xoL_min, log_p2_q_xoL_max = get_cavity_pressure(
        log_p1_max_q, log_p2_max_q, log_p3_max_q,
        xol, length_depth=length_depth)
    plot_cavity_pressure(
        11,
        xol,
        log_p1_q_xoL, log_p2_q_xoL, log_p3_q_xoL,
        log_p2_q_xoL_min, log_p2_q_xoL_max)

    plot_normalized_modal_pressure(
        12, xol,
        log_p1_q_xoL, log_p1_max_q,
        log_p2_q_xoL, log_p2_max_q,
        log_p3_q_xoL, log_p3_max_q,
        log_p2_q_xoL_min, log_p2_q_xoL_max,
        length_depth=length_depth,
    )

    log_pb_q = get_broadband_pressure(
        xol, log_p2_max_q, length_depth=length_depth)

    plot_broadband_pressure(
        13, xol, log_pb_q, log_p2_max_q,
        length_depth=length_depth)

    vel_plot(16, alts, rhos, vels, use_keas=True)

    plt.show()
    # U = 1.
    # L = 1.
    # freq = L / U * fstar1


def plot_peak_variation_with_mach(fig_id: int,
                                  machs,
                                  log_p1_max_qs, log_p2_max_qs, log_p3_max_qs,
                                  length_depth: float=4.0):
    fig, ax = get_fig_ax(fig_id)

    max1 = log_p1_max_qs.max()
    max2 = log_p2_max_qs.max()
    max3 = log_p3_max_qs.max()
    imax1 = np.where(log_p1_max_qs == max1)[0]
    imax2 = np.where(log_p2_max_qs == max2)[0]
    imax3 = np.where(log_p3_max_qs == max3)[0]
    ax.plot(machs, log_p1_max_qs, label=f'Mode 1: {max1:g}', color='C0')
    ax.plot(machs, log_p2_max_qs, label=f'Mode 2: {max2:g}', color='C1')
    ax.plot(machs, log_p3_max_qs, label=f'Mode 3: {max3:g}', color='C2')
    ax.plot(machs[imax1], log_p1_max_qs[imax1], marker='o', color='C0')
    ax.plot(machs[imax2], log_p2_max_qs[imax2], marker='o', color='C1')
    ax.plot(machs[imax3], log_p3_max_qs[imax3], marker='o', color='C2')
    ax.plot(plot2_data_mode1[:, 0], plot2_data_mode1[:, 1], linestyle='--', marker='o', linewidth=2, color='k')
    ax.plot(plot2_data_mode2[:, 0], plot2_data_mode2[:, 1], linestyle='--', marker='o', linewidth=2, color='k')
    ax.plot(plot2_data_mode3[:, 0], plot2_data_mode3[:, 1], linestyle='--', marker='o', linewidth=2, color='k')
    ax.legend()
    ax.set_xlabel('Mach')
    ax.set_ylabel('$20$ log$(p_{N,max}/q)$')
    ax.set_xlim([0.6, 1.3])
    yticks = np.arange(-36, -12. + 1, 4.0)
    ax.set_yticks(yticks)
    ax.set_ylim([-36., -12.])

    ax.grid(True)
    fig.suptitle(f'Seems good. 4.3.2-2 Variation of Modal Peak Amplification\n'
                 f' with Mach Number for First 3 Modes (L/D={length_depth}')


def plot_cavity_distribution(fig_id: int, length_depth: float=4.0):
    fig = plt.figure(fig_id)
    ax = fig.gca()
    return fig, ax


def plot_normalized_modal_pressure(fig_id: int,
                                   xol: np.ndarray,
                                   log_p1_q_xol: np.ndarray, log_p1_max_q: float,
                                   log_p2_q_xol: np.ndarray, log_p2_max_q: float,
                                   log_p3_q_xol: np.ndarray, log_p3_max_q: float,
                                   log_p2_q_xol_min: np.ndarray,
                                   log_p2_q_xol_max: np.ndarray,
                                   length_depth: float=4.0) -> tuple[plt.Figure, plt.Axes]:
    """cavity_distribution"""
    fig, ax = get_fig_ax(fig_id)
    ax.grid(True)
    ax.set_xlabel('x/L')
    ax.set_ylabel('20 log$(p_N/q) - $20 log$(p_{N,max}/q)$')
    ax.set_xlim([0.0, 1.0])
    xticks = np.arange(0.0, 1.1, 0.1)
    yticks = np.arange(-20, 2, 2)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    fig.suptitle('4.3.2-3 Modal Pressre Distribution Along\n'
                 f'Cavity Length; L/D={length_depth}')

    ax.plot(xol, log_p1_q_xol - log_p1_max_q, label='Mode 1', marker='o')
    ax.plot(xol, log_p2_q_xol - log_p2_max_q, label='Mode 2')
    ax.plot(xol, log_p3_q_xol - log_p3_max_q, label='Mode 3')
    ax.plot(xol, log_p2_q_xol_min - log_p2_max_q, linestyle='--', color='C1')
    ax.plot(xol, log_p2_q_xol_max - log_p2_max_q, linestyle='--', color='C1')

    # tag specific points on the curves
    xpoints = [0.5, 1.0]
    for xoli in xpoints:
        xols = np.array([xoli, xoli])
        log_p1_q_xoL_, log_p2_q_xoL_, log_p3_q_xoL_, log_p2_q_xoL_min_, log_p2_q_xoL_max_ = get_cavity_pressure(
            log_p1_max_q, log_p2_max_q, log_p3_max_q,
            xols, length_depth=length_depth)
        delta = log_p2_q_xoL_min_ - log_p2_max_q
        ax.scatter(xols, delta, marker='o', color='C3', label=f'x/L={xoli:.1f}, y={delta.max():.2f}')

    ax.plot(plot3_mode2[:, 0], plot3_mode2[:, 1], linestyle='--', color='k')  # marker='o',
    ax.set_ylim([-18, 0.])
    ax.legend()
    return fig, ax


def plot_broadband_pressure(fig_id: int,
                            xol: np.ndarray,
                            log_pb_q: np.ndarray,
                            log_p2_max_q: np.ndarray,
                            length_depth: float=4.0) -> None:
    fig, ax = get_fig_ax(fig_id)
    ax.plot(xol, log_pb_q - log_p2_max_q, label='broadband')
    ax.legend()
    fig.suptitle(f'Wrong; p2_max={log_p2_max_q:g}')

    ax.grid(True)
    ax.set_xlabel('x/L')
    ax.set_ylabel('20 log$(p_b/q) - 20 $log$(p_{2,max}/q)$')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-25., -15.])
    fig.suptitle('4.3.2-4 Variation of Broadband Specturm Level\n'
                 f'with Cavity Position; L/D={length_depth}')


def plot_normalized_frequency(fig_id: plt.Figure | int,
                              fstars_dict: dict[int, np.ndarray],
                              machs: np.ndarray):
    fig, ax = get_fig_ax(fig_id)
    for imode, fstars in fstars_dict.items():
        ax.plot(machs, fstars, label=f'Mode {imode} min={fstars.min():.3f} max={fstars.max():.3f}')
    ax.legend()
    ax.set_xlabel('Mach')
    ax.set_ylabel('freq*')
    ax.set_xlim([0.6, 1.3])
    ax.set_ylim([0.2, 1.2])
    ax.grid(True)
    fig.suptitle('Seems good. Variation of Cavity Resonant Strouhal Frequencies\n'
                 'with Mach Number for First 3 Length Modes')


def plot_frequency(fig_id: plt.Figure | int,
                   freq_dict: dict[int, np.ndarray],
                   machs: np.ndarray, uvelocity: float=0.0,
                   ) -> None:
    assert len(freq_dict) > 0, freq_dict
    fig, ax = get_fig_ax(fig_id)
    # f1* = U/L k
    # f1* L / U =
    ax.set_xlabel('Mach')
    ax.set_ylabel('Frequency (Hz)')
    for imode, freqs in freq_dict.items():
        ax.plot(machs, freqs, label=f'Mode {imode} min={freqs.min():.1f} max={freqs.max():.1f}')

    ustr = ''
    if isinstance(uvelocity, float) and uvelocity > 0.0:
        ustr = f'U={uvelocity:.0f} ft/s, '
    fig.suptitle(f'{ustr}Sea Level')
    ax.legend()
    ax.grid(True)


def plot_cos_alpha(fig_id: int,
                   xol: np.ndarray,
                   cos1_axl: np.ndarray,
                   cos2_axl: np.ndarray,
                   cos3_axl: np.ndarray,
                   ) -> None:
    fig, ax = get_fig_ax(fig_id)
    ax.grid(True)
    ax.set_ylabel(r'|cos$(\alpha x/L)$|')
    ax.set_xlabel('x/L')
    ax.set_xlim([0., 1.])
    # cos1_axL = np.abs(np.cos(alpha1 * xoL))

    ax.plot(xol, cos1_axl, label='Mode 1', color='C0')
    ax.plot(xol, cos2_axl, label='Mode 2', color='C1')
    ax.plot(xol, cos3_axl, label='Mode 3', color='C2')


def main():
    length = 20.
    depth = 5.
    length_depth = length / depth
    fpl = -20

    vels = np.array([
        50., 60., 70., 80, 90, 100.,
        110, 120, 130., 140, 150, 175,
        200,
        225, 250,
        275,
        300, 350, 400, 500,
        600, 700, 800, 900, 1000, 1100, 1200,
        1300, 1400, 1500, 1750, 2000, 2500, 3000.,
    ])
    sonic_fatigue(vels=vels, length=length,
                  length_depth=length_depth, fpl=fpl)


if __name__ == '__main__':
    main()
