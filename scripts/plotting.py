"""Figure style and India maps for the paper on lightning exposure in India.

Liberation Sans (Arial metrics), 7 pt text, widths in mm (Nature: 89 mm one
column, 183 mm two). Every map shows India only, inside its official
(India-view) boundary, with the Survey of India based state boundaries.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import shapely
from cartopy.mpl.path import shapely_to_path

from common import FIG, LAT, LON, SUP, india_geometry, states

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.4,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlelocation": "left",
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42,
})
MM = 1 / 25.4
PC = ccrs.PlateCarree()
EXTENT = [67.6, 97.6, 6.3, 37.3]
INK, INK2, MUTED, GRID, LAND = "#0b0b0b", "#52514e", "#8a8985", "#dcdbd6", "#f3f2ee"

# one-hue ramps, light to dark (blue: the reference palette; the others matplotlib's)
BLUE = ["#e6f0fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b", "#07203f"]
REGION_COLOR = {"Himalaya": "#e87ba4", "Northwest": "#eda100", "Gangetic Plain": "#1baf7a",
                "Central": "#2a78d6", "East": "#eb6834", "Northeast": "#4a3aa7", "South": "#008300"}
SEASON_COLOR = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]     # winter, pre-monsoon, monsoon, post-monsoon
DRIVER_COLOR = {"CAPE": "#008300", "Rain": "#2a78d6", "CAPE times rain": "#e34948"}

_INDIA = india_geometry().simplify(0.005)          # about 500 m: invisible in print, small files
_STATES = states()
_INNER = shapely.intersection(shapely.line_merge(shapely.union_all([g.simplify(0.005).boundary for g in _STATES.geometry])),
                              _INDIA.buffer(-0.03))                # state lines inside India, simplified


def ramp(colors, n):
    """n colours evenly along a list of colours (a one-hue ramp)."""
    cm = mcolors.LinearSegmentedColormap.from_list("r", colors)
    return [cm(x) for x in np.linspace(0, 1, n)]


def classes(levels, colors, extend="max"):
    """Discrete colormap and norm for the class edges `levels`."""
    n = len(levels) - 1 + (extend in ("max", "both")) + (extend in ("min", "both"))
    cmap = mcolors.ListedColormap(ramp(colors, n) if isinstance(colors, list) else
                                  [plt.get_cmap(colors)(x) for x in np.linspace(0.08, 1, n)])
    return cmap, mcolors.BoundaryNorm(levels, cmap.N, extend=extend)


def diverging(levels, neg="#2a78d6", pos="#e34948", mid="#f0efec"):
    """Discrete blue-grey-red colormap and norm, grey around the middle class."""
    n = len(levels) + 1
    half = n // 2
    cols = ramp([neg, mid], half + 1)[:-1] + ([mid] if n % 2 else []) + ramp([mid, pos], half + 1)[1:]
    cmap = mcolors.ListedColormap(cols)
    return cmap, mcolors.BoundaryNorm(levels, cmap.N, extend="both")


def india_map(ax, ticks=True, states=True):
    """India's official boundary and state boundaries on a PlateCarree GeoAxes."""
    ax.set_extent(EXTENT, PC)
    ax.add_geometries([_INDIA], PC, facecolor=LAND, edgecolor="none", zorder=0)
    if states:
        ax.add_geometries([_INNER], PC, facecolor="none", edgecolor=MUTED, lw=0.25, zorder=3)
    ax.add_geometries([_INDIA], PC, facecolor="none", edgecolor=INK, lw=0.5, zorder=4)
    ax.spines["geo"].set_linewidth(0.5)
    ax.spines["geo"].set_edgecolor(MUTED)
    if ticks:
        ax.set_xticks([70, 80, 90], crs=PC)
        ax.set_yticks([10, 20, 30], crs=PC)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}°E")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}°N")
        ax.tick_params(length=2, labelsize=6, colors=INK2)


def field(ax, data, cmap, norm, lat=LAT, lon=LON):
    """A gridded field (cell centres lat, lon) drawn inside India's boundary only."""
    m = ax.pcolormesh(lon, lat, np.ma.masked_invalid(data), cmap=cmap, norm=norm, transform=PC,
                      shading="nearest", zorder=1, rasterized=True)
    m.set_clip_path(shapely_to_path(_INDIA), PC._as_mpl_transform(ax))
    return m


def colorbar(fig, m, cax, label, ticks=None, fmt=None, orientation="horizontal"):
    cb = fig.colorbar(m, cax=cax, orientation=orientation, ticks=ticks, format=fmt,
                      spacing="uniform", extendfrac=0.06)
    cb.set_label(label, fontsize=6.5, color=INK)
    cb.ax.tick_params(labelsize=6, length=2, width=0.5, colors=INK2)
    cb.outline.set_linewidth(0.4)
    return cb


def label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def save(fig, name, sup=False):
    """Save to manuscript/figures (or supplementary) as PNG and PDF; returns the PNG path."""
    out = (SUP if sup else FIG) / name
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return f"{out}.png"
