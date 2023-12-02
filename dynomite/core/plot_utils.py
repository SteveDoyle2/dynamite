import numpy as np
import matplotlib.pyplot as plt


def _set_grid(ax: plt.Axes, xscale: str, yscale: str) -> None:
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xscale == 'linear':
        ax.grid(True)
    else:
        assert xscale == 'log', xscale
        ax.grid(True, which='both')

def _adjust_axes_limit(ax2: plt.Axes, imag: np.ndarray) -> None:
    if np.allclose(imag.min(), imag.max()):
        imag_min = imag.min()
        ax2.set_ylim([imag_min - 1, imag_min + 1])

