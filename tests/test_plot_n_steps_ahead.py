import os
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from plotly_utils.time_series import plot_n_steps_ahead_predictions


class TestPlotNStepsAhead(unittest.TestCase):
    def test(self) -> None:
        r"""Plot data following the model

        .. math::

            y = 2x + \epsilon

        with :math:`\epilon \approx \mathcal{N}(0, 1)`. In order to distinguish
        the predictions from the original data, the predictions will be equal
        to the last known point, i.e. we simulate a model that predicts a
        constant. The uncertanty of the model grows linearly and symmetrically
        starting from ±0.1 and growing by ±0.3 at each step

        """
        orig_data = 2 * np.arange(10, dtype=np.float64)
        pred_mean = np.empty((10, 3), dtype=orig_data.dtype)
        for idx, y in enumerate(orig_data):
            pred_mean[idx, :] = y
        conf_int = pred_mean.copy()[:, :, None].repeat(2, axis=2)
        uncertainty = np.array([0.1, 0.4, 0.7], dtype=conf_int.dtype)
        conf_int[:, :, 0] -= uncertainty[None, :]
        conf_int[:, :, 1] += uncertainty[None, :]
        fig = plot_n_steps_ahead_predictions(
            orig_data,
            pred_mean,
            conf_int,
            alpha=0.9,
            title="Title",
            xaxis_title="x axis title",
            yaxis_title="y axis title",
        )
        expected_plot = Image.open(
            Path(__file__).parent / "test_data" / "expected_n_steps_ahead_plot.png"
        )
        fig.write_image("actual_plot.png", width=853, height=480)
        actual_plot = Image.open("actual_plot.png")
        os.remove("actual_plot.png")
        self.assertTrue(np.all(np.array(expected_plot) == np.array(actual_plot)))
