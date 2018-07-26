""" convenient file for saving matplotlib figures
"""


import os
import matplotlib.pyplot as plt


_default_dpi = 200


def png(fig, fname):
    """ save as png
    """
    fname = os.path.expanduser(fname)
    plt.savefig('%s.png' % fname, format='png',
                dpi=_default_dpi, bbox_inches='tight')
    plt.close(fig)


def svg(fig, fname):
    """ save as svg
    """
    fname = os.path.expanduser(fname)
    plt.savefig('%s.svg' % fname, format='svg',
                dpi=_default_dpi, bbox_inches='tight')
    plt.close(fig)
