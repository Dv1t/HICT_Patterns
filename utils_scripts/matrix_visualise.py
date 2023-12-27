import string

import matplotlib.colors as clr
import numpy as np


class MatrixVisualise(object):

    @staticmethod
    def log_matrix(matrix: np.ndarray, log_base: float = 10, addition: float = 1,
                   remove_zeros: bool = True) -> np.ndarray:
        if remove_zeros:
            matrix[matrix == 0] = np.NaN
        return np.log(matrix + addition) / np.log(log_base)

    @staticmethod
    def get_colormap(start_color_hex: string, mid_color_hex: string, end_color_hex: string,
                     gradient_levels: tuple = (0, 0.5, 1)) -> clr.LinearSegmentedColormap:
        if len(gradient_levels) != 3:
            raise Exception("gradient_levels tuple should have 3 elements")
        return clr.LinearSegmentedColormap.from_list('custom colormap', [(gradient_levels[0], start_color_hex),
                                                                         (gradient_levels[1], mid_color_hex),
                                                                         (gradient_levels[2], end_color_hex)])

    @staticmethod
    def get_colormap_diverging(first_quarter: float = 0.250, second_quarter: float = 0.750, mid_position:float = 0.500,
                               start_color: tuple = (0.000, 0.145, 0.702), end_color: tuple = (0.780, 0.012, 0.051),
                               mid_color: tuple = (1.000, 1.000, 1.000)) \
            -> clr.LinearSegmentedColormap:
        return clr.LinearSegmentedColormap.from_list('diverging_clr', (
            (0.000, start_color),
            (first_quarter, start_color),
            (mid_position, mid_color),
            (second_quarter, end_color),
            (1.000, end_color)))

    @staticmethod
    def calculate_diag_means(matrix: np.ndarray, res: string = 'exp/obs') -> np.ndarray:
        result = np.zeros_like(matrix, dtype='float64')
        expected = np.zeros_like(matrix, dtype='float64')
        assert (
            matrix.shape[0] == matrix.shape[1]
        ), "Matrix must be square"
        if True:
            n = matrix.shape[0]
            expected = sum(
                (
                    np.diag(
                        [np.nanmean(matrix.diagonal(offset=i))] * (n-abs(i)), k=i
                    ) for i in range(1-n, n)
                )
            )
        else:
            expected = np.ones_like(matrix) * np.nanmean(matrix)

        if res == 'exp/obs':
            return expected/matrix

        if res == 'exp':
            return expected

        if res == 'exp-obs':
            return expected - matrix

        if res == 'obs-exp':
            return matrix - expected

        if res == 'obs/exp':
            return matrix/expected

        return result
