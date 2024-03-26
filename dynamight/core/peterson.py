import numpy as np
import matplotlib.pyplot as plt
equations = {
    'Peterson Table 17.1': [
        'K_t = c_1 + c_2(2h/D) + c_3 (2h/D)^2 + c_4 (2h/D)^3',
        '0.1 <= h/r <= 2.0'
        'c_1 = 0.850  + 2.628 \sqrt{h/r} - 0.413 (h/r)',
        'c_2 = -1.119 - 4.826 \sqrt{h/r} + 2.575 (h/r)',
        'c_3 = 3.563  - 0.514 \sqrt{h/r} - 2.402 (h/r)',
        'c_4 = -2.294 + 2.713 \sqrt{h/r} + 0.240 (h/r)',

        '2.0 <= h/r <= 50.0',
        'c_1 = 0.833 + 2.069 \sqrt{h/r} - 0.009 (h/r)',
        'c_2 = 2.732 - 4.157 \sqrt{h/r} + 0.176 (h/r)',
        'c_3 =-8.859 + 5.317 \sqrt{h/r} - 0.320 (h/r)',
        'c_4 = 6.294 - 3.239 \sqrt{h/r} + 0.154 (h/r)',
    ],
}
nhr = 5001
h_r = np.linspace(0.1, 50., num=nhr)
i = ((0.1 <= h_r) & (h_r <= 2.0))
#0.1 <= h/r <= 2.0'
c1 = np.zeros(nhr, dtype='float64')
c2 = np.zeros(nhr, dtype='float64')
c3 = np.zeros(nhr, dtype='float64')
c4 = np.zeros(nhr, dtype='float64')
h_ri = h_r[i]
c1[i] = 0.850  + 2.628 * np.sqrt(h_ri) - 0.413*(h_ri)
c2[i] = -1.119 - 4.826 * np.sqrt(h_ri) + 2.575*(h_ri)
c3[i] = 3.563  - 0.514 * np.sqrt(h_ri) - 2.402*(h_ri)
c4[i] = -2.294 + 2.713 * np.sqrt(h_ri) + 0.240*(h_ri)

j = ((2.0 <= h_r) & (h_r <= 50.))
#'2.0 <= h/r <= 50.0'
h_rj = h_r[j]
c1[j] = 0.833 + 2.069 * np.sqrt(h_rj) - 0.009*(h_rj)
c2[j] = 2.732 - 4.157 * np.sqrt(h_rj) + 0.176*(h_rj)
c3[j] =-8.859 + 5.317 * np.sqrt(h_rj) - 0.320*(h_rj)
c4[j] = 6.294 - 3.239 * np.sqrt(h_rj) + 0.154*(h_rj)

eqs = {}
for h_D in (0.1, 0.2, 0.4, 0.5):
    eq = c1 + c2 * (2*h_D) + c3*(2*h_D)**2 + c4*(2*h_D)**3
    eqs[h_D] = eq

ifig = 1
for name, data in equations.items():
    fig = plt.figure(ifig)
    ax = fig.gca()
    x, y = 0, 0
    s = ''
    for line in data:
        s += '$' + line + '$\n'
    #plt.text(x, y, s)
    plt.suptitle(s)

    for h_D, eq in eqs.items():
        plt.plot(h_r, eq, label='$h/D$ = %s' % h_D)
    #ax.set_title(s, loc='left')
    ifig += 1
plt.legend()
plt.show()
