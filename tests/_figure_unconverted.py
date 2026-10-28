"""Figure kinds the round-4 audit still fails: case -> checks expected to fail.

Generated from a run of `test_figure_style` before the figures were converted. Each
entry is a strict xfail: when a figure is converted its entries start to pass, the
test fails, and the entry is deleted here, so the check then guards that figure.
An empty dict means every figure follows the round-4 rules.
"""

UNCONVERTED: dict[str, tuple[str, ...]] = {
    "F21-report-column": (
        "wrapped title",
        "method legend",
        "one axes",
        "caption",
        "size class",
    ),
    "F21-report-double": (
        "method legend",
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
    "F21-video-column": (
        "method legend",
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
    "F22-report-column": (
        "wrapped title",
        "method legend",
        "caption",
        "size class",
    ),
    "F22-report-double": (
        "wrapped title",
        "method legend",
        "text kinds",
        "caption",
        "size class",
    ),
    "F22-video-column": (
        "method legend",
        "text kinds",
        "caption",
        "size class",
    ),
    "F23-report-column": (
        "wrapped title",
        "method legend",
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
    "F23-report-double": (
        "method legend",
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
    "F23-video-column": (
        "method legend",
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
    "F24-report-column": (
        "wrapped title",
        "one axes",
        "caption",
        "size class",
    ),
    "F24-report-double": (
        "wrapped title",
        "one axes",
        "caption",
        "size class",
    ),
    "F24-video-column": (
        "wrapped title",
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
    "F25-report-column": (
        "wrapped title",
        "method legend",
        "caption",
        "size class",
    ),
    "F25-report-double": (
        "method legend",
        "text kinds",
        "caption",
        "size class",
    ),
    "F25-video-column": (
        "method legend",
        "text kinds",
        "caption",
        "size class",
    ),
    "F26-report-column": (
        "wrapped title",
        "method legend",
        "one axes",
        "caption",
        "size class",
    ),
    "F26-report-double": (
        "wrapped title",
        "method legend",
        "one axes",
        "caption",
        "size class",
    ),
    "F26-video-column": (
        "text overlap",
        "wrapped title",
        "method legend",
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
    "F27-report-column": (
        "wrapped title",
        "method legend",
        "caption",
        "size class",
    ),
    "F27-report-double": (
        "wrapped title",
        "method legend",
        "caption",
        "size class",
    ),
    "F27-video-column": (
        "method legend",
        "text kinds",
        "caption",
        "size class",
    ),
    "F28-report-column": (
        "wrapped title",
        "caption",
        "size class",
    ),
    "F28-report-double": (
        "text kinds",
        "caption",
        "size class",
    ),
    "F28-video-column": (
        "text kinds",
        "caption",
        "size class",
    ),
    "fidelity": (
        "method legend",
        "text kinds",
        "caption",
        "size class",
    ),
    "hard": (
        "text kinds",
        "caption",
        "size class",
    ),
    "hard-variants": (
        "one axes",
        "text kinds",
        "caption",
        "size class",
    ),
}
