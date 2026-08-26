# Shared plotting style for every report figure.
#
# Figures are drawn at close to the size they are printed (the report's text column is about
# 6.7 inches), so labels are not shrunk by the LaTeX \includegraphics scaling. The font sizes
# below are therefore also the sizes seen on the page, sitting just under the 12pt body text.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXTWIDTH = 6.7  # printed width of one text column, inches

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11.5,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "legend.framealpha": 0.92,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.7,
    "lines.markersize": 5.5,
    "errorbar.capsize": 3,
})


def wide(height=3.5):
    # size for a figure spanning the full text column
    return (TEXTWIDTH, height)


def half(height=3.4):
    # size for a figure spanning about two thirds of the column
    return (0.72 * TEXTWIDTH, height)


def save_exact(fig, paths, dpi=150):
    # Save at exactly the declared figsize.
    #
    # savefig.bbox is "tight" for most figures, which trims whitespace -- but "tight" also EXPANDS the
    # canvas to include anything that spills past the axes margins (a wide figure legend, a long y-axis
    # label).  A figure that comes out wider than the text column is then scaled down by
    # \includegraphics and every label in it shrinks by the same factor.  For figures with figure-level
    # legends the width must be pinned instead, and the margins set by subplots_adjust must be roomy
    # enough that nothing is clipped.
    import matplotlib.pyplot as _plt
    prev = _plt.rcParams["savefig.bbox"]
    _plt.rcParams["savefig.bbox"] = "standard"
    try:
        for path in ([paths] if isinstance(paths, str) else paths):
            fig.savefig(path, dpi=dpi)
    finally:
        _plt.rcParams["savefig.bbox"] = prev
