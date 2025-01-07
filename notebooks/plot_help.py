######################### Plotting helper functions ########################
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def mix_colors(color1, color2):
    """
    Mix two colors and return the result as a hex color code.

    Parameters:
    color1 (str): First color in a format recognized by matplotlib (e.g., 'blue', '#1f77b4').
    color2 (str): Second color in a format recognized by matplotlib.

    Returns:
    str: Hex color code of the mixed color.
    """
    c1 = np.array(mcolors.to_rgb(color1))
    c2 = np.array(mcolors.to_rgb(color2))
    return mcolors.to_hex((c1 + c2) / 2)
