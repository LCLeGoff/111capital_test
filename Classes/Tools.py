from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def get_list_color(cmap, idx_list):

        cols = dict()
        if isinstance(idx_list, int) or isinstance(idx_list, float):
            norm = colors.Normalize(0, int(idx_list) - 1)
            cmap = plt.get_cmap(cmap)
            for i in range(int(idx_list)):
                cols[str(i)] = cmap(norm(i))
        else:
            norm = colors.Normalize(0, len(idx_list) - 1)
            cmap = plt.get_cmap(cmap)
            for i, idx in enumerate(idx_list):
                if isinstance(idx, np.ndarray):
                    idx = tuple(idx)
                cols[str(idx)] = cmap(norm(i))
        return cols
