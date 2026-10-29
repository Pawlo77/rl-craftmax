"""The two Spearman helpers of the distillation CLI: constant input is NaN, quietly."""

import math
import warnings

from wut_murl_craftax.interpretation import run as cli

NAN = (math.nan, math.nan)


def test_constant_levels_above_three_reached_give_nan_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cli.spearman([1.0, 2.0, 3.0, 4.0], [16, 16, 16, 16]) == NAN


def test_constant_scores_give_nan_without_a_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cli.spearman([2.0, 2.0, 2.0, 2.0], [16, 32, 64, 128]) == NAN
        assert cli._spearman_values([0.5] * 4, [1.0, 2.0, 3.0, 4.0]) == NAN
        assert cli._spearman_values([1.0, 2.0, 3.0, 4.0], [0.5] * 4) == NAN


def test_varying_input_still_ranks() -> None:
    rho, p = cli.spearman([1.0, 2.0, 3.0, 4.0], [16, 32, 64, 128])
    assert rho == 1.0
    assert p < 0.01
    assert cli._spearman_values([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])[0] == -1.0
