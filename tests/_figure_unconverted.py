"""Figure kinds the round-4 audit still fails: case -> checks expected to fail.

Generated from a run of `test_figure_style` before the figures were converted. Each
entry is a strict xfail: when a figure is converted its entries start to pass, the
test fails, and the entry is deleted here, so the check then guards that figure.
An empty dict means every figure follows the round-4 rules.
"""

UNCONVERTED: dict[str, tuple[str, ...]] = {}
