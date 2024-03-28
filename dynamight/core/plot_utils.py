from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def get_colors(colormap_name: str,
               xmin: float, xmax: float,
               nresponses: int) -> np.ndarray:
    if colormap_name:
        x = np.linspace(xmin, xmax, num=nresponses)[::-1]
        colormap = plt.get_cmap(colormap_name)
        colors = colormap(x)
    else:
        colors = cycle(['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9'])
    return colors

def _set_grid(ax: plt.Axes, xscale: str, yscale: str) -> None:
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xscale == 'linear':
        ax.grid(True)
    else:
        assert xscale == 'log', xscale
        ax.grid(True, which='both')
        #plt.ticklabel_format(axis="x", style="plain")
        #plt.ticklabel_format(axis="both", style="plain")

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        #ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        #plt.ticklabel_format(axis='x', style='plain')


def _adjust_axes_limit(ax2: plt.Axes, imag: np.ndarray) -> None:
    if np.allclose(imag.min(), imag.max()):
        imag_min = imag.min()
        ax2.set_ylim([imag_min - 1, imag_min + 1])

