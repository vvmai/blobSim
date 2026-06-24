"""GIF rendering pipeline for blobsim simulations.

state → frame image (matplotlib) → animated GIF (Pillow).
Streams frames via generator — peak memory is ~2 frames.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image

from blobsim.config import SimulationConfig

if TYPE_CHECKING:
    from blobsim.simulation import SimulationResult
    from blobsim.types import BlobRecord, LedgerSnapshot

# ── Constants ────────────────────────────────────────────────────────
GRID_CMAP = "YlGn"
MARKER_SIZE_RANGE = (20, 80)
BLOB_MAX_ENERGY = 20.0
CELL_FILL_BG = (0.88, 0.88, 0.88)  # light grey background for cell_fill mode
ALPHA_LEVELS = 4  # energy-opacity fade is snapped to this many steps (GIF size)


# ── Public API ───────────────────────────────────────────────────────

def render_gif(
    result: SimulationResult,
    output_path: str | Path | None = None,
    *,
    fps: int = 10,
    dpi: int = 72,
    cell_pixels: int = 12,
    sample_every: int = 1,
    show_grid: bool = True,
    size_by_energy: bool = True,
    alpha_by_energy: bool = False,
    marker_size: float = 40.0,
    cell_fill: bool = False,
) -> Path:
    """Render simulation history to an animated GIF.

    Parameters
    ----------
    result : SimulationResult
        Completed simulation with recorded state history.
    output_path : path, optional
        Destination file. Default: ``blobsim_<timestamp>.gif`` in cwd.
    fps : int
        Frames per second (default 10).
    dpi : int
        Figure resolution (default 72).
    cell_pixels : int
        Pixels per grid cell (default 12).
    sample_every : int
        Render every Nth frame (default 1 = all frames).

    Returns
    -------
    Path
        Resolved path to the written GIF file.
    """
    recorder = result.recorder
    grid_size = result.config.grid_size

    # Deterministic palette (once)
    species = {r.species for step in recorder.blob_records for r in step}
    palette = _build_palette(species)

    # Global vmax for consistent colorscale across all frames
    grid_vmax = max(
        (g.max() for g in recorder.grid_records if g is not None),
        default=1.0,
    )

    # Compact config annotation
    info_lines = _build_info_lines(result.config)

    # Generator — yields RGBA frames, never accumulates
    def frames() -> Iterator[np.ndarray]:
        for t in range(0, len(recorder.blob_records), sample_every):
            yield _render_frame(
                recorder.grid_records[t],
                recorder.blob_records[t],
                recorder.ledger_records[t],
                grid_size,
                palette,
                grid_vmax,
                dpi,
                cell_pixels,
                info_lines,
                show_grid=show_grid,
                size_by_energy=size_by_energy,
                alpha_by_energy=alpha_by_energy,
                marker_size=marker_size,
                cell_fill=cell_fill,
            )

    path = Path(output_path) if output_path else Path(f"blobsim_{int(time.time())}.gif")
    return _write_gif(frames(), path, duration_ms=1000 // fps)


# ── Private helpers ──────────────────────────────────────────────────

def _build_palette(species_names: set[str]) -> dict[str, tuple[float, ...]]:
    """Map sorted species names → deterministic RGBA colors via tab10."""
    cmap = colormaps["tab10"]
    return {name: cmap(i % 10) for i, name in enumerate(sorted(species_names))}


def _build_info_lines(config: SimulationConfig) -> list[str]:
    """Extract compact config annotation strings."""
    lines: list[str] = []

    # Environment info
    env = config.environment
    if hasattr(env, "rate"):
        lines.append(f"grid:{config.grid_size} regen(r={env.rate},cap={env.capacity})")
    else:
        lines.append(f"grid:{config.grid_size} decay(e0={env.initial_cell_energy})")

    # Per-species summary
    for sc in config.species:
        a = sc.attributes
        lines.append(
            f"{a.species}(n={sc.count},bmr={a.base_metabolic_cost},max_e={a.max_energy})"
        )

    return lines


def _render_frame(
    grid: np.ndarray | None,
    blob_records: list[BlobRecord],
    ledger: LedgerSnapshot,
    grid_size: int,
    palette: dict[str, tuple[float, ...]],
    grid_vmax: float,
    dpi: int,
    cell_pixels: int,
    info_lines: list[str] | None = None,
    *,
    show_grid: bool = True,
    size_by_energy: bool = True,
    alpha_by_energy: bool = False,
    marker_size: float = 40.0,
    cell_fill: bool = False,
) -> np.ndarray:
    """Render a single simulation step to an RGBA numpy array."""
    fig_w = grid_size * cell_pixels / dpi
    fig_h = fig_w  # square
    n_info = len(info_lines) if info_lines else 0
    info_h = 0.13 * n_info
    total_h = fig_h + 0.55 + info_h
    total_w = fig_w + 1.0
    fig = Figure(figsize=(total_w, total_h), dpi=dpi, facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax_bottom = info_h / total_h + 0.02
    ax = fig.add_axes((0.08, ax_bottom, fig_w / total_w, fig_h / total_h))

    # Cell-fill mode: each live blob fills its whole cell with its solid
    # species colour on a light grey background; a dead cell is just the
    # background, so deaths read as cells winking back to grey.
    if cell_fill:
        bg = CELL_FILL_BG
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)
        img = np.ones((grid_size, grid_size, 3)) * np.array(bg)
        bg_arr = np.array(bg)
        for r in blob_records:
            if not r.alive:
                continue
            base = np.array(palette.get(r.species, (1.0, 1.0, 1.0, 1.0))[:3])
            if alpha_by_energy:
                # composite species colour over the background by energy:
                # >=50% energy renders at full colour; below 50% it fades
                # toward the background ("dead disappears"). Snap to a few
                # discrete levels so the GIF palette stays small (continuous
                # alpha gives every cell a unique shade and kills compression).
                f = min(max(r.energy / BLOB_MAX_ENERGY, 0.0), 1.0)
                a = round(min(f / 0.5, 1.0) * ALPHA_LEVELS) / ALPHA_LEVELS
                base = bg_arr * (1.0 - a) + base * a
            img[r.position[0], r.position[1]] = base
        ax.imshow(img, origin="lower", interpolation="nearest")
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"Step {ledger.step} | Alive: {ledger.n_alive}",
            color="black", fontsize=9, pad=4,
        )
        if info_lines:
            fig.text(
                0.08, 0.01, "\n".join(info_lines),
                color="black", fontsize=9, alpha=0.7,
                verticalalignment="bottom", family="monospace",
            )
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba()).copy()
        fig.clear()
        return buf

    # Grid heatmap (optional — hide for a clean black background)
    if show_grid and grid is not None:
        im = ax.imshow(
            grid, cmap=GRID_CMAP, vmin=0, vmax=grid_vmax,
            origin="lower", interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.0)
        cbar.set_ticks([0, grid_vmax])
        cbar.set_ticklabels(["0", f"{grid_vmax:.1f}"])
        cbar.ax.tick_params(labelsize=9, colors="white")
        cbar.set_label("Grid energy", color="white", fontsize=9)
    elif show_grid:
        ax.imshow(
            np.zeros((grid_size, grid_size)), cmap="Greys", vmin=0, vmax=1,
            origin="lower", interpolation="nearest",
        )
        ax.text(
            grid_size / 2, grid_size / 2, "no grid data",
            ha="center", va="center", color="white", fontsize=8, alpha=0.6,
        )
    else:
        ax.set_facecolor("black")

    # Blobs (alive only)
    alive = [r for r in blob_records if r.alive]
    if alive:
        xs = [r.position[1] for r in alive]  # col → x
        ys = [r.position[0] for r in alive]  # row → y

        # Marker size: fixed, or proportional to energy
        if size_by_energy:
            lo, hi = MARKER_SIZE_RANGE
            sizes = [
                lo + (hi - lo) * min(r.energy / BLOB_MAX_ENERGY, 1.0)
                for r in alive
            ]
        else:
            sizes = marker_size

        # Marker colour: base species colour, with opacity ∝ energy if asked
        # (so low-energy / dying blobs fade out against the black background).
        if alpha_by_energy:
            colors = []
            for r in alive:
                base = palette.get(r.species, (1.0, 1.0, 1.0, 1.0))
                a = min(max(r.energy / BLOB_MAX_ENERGY, 0.0), 1.0)
                colors.append((base[0], base[1], base[2], a))
            edge = "none"
        else:
            colors = [palette.get(r.species, (1, 1, 1, 1)) for r in alive]
            edge = "white"

        ax.scatter(
            xs, ys, s=sizes, c=colors,
            edgecolors=edge, linewidths=0.5, zorder=2,
        )

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Step {ledger.step} | Alive: {ledger.n_alive}",
        color="white", fontsize=9, pad=4,
    )

    # Config annotation
    if info_lines:
        fig.text(
            0.08, 0.01, "\n".join(info_lines),
            color="white", fontsize=9, alpha=0.7,
            verticalalignment="bottom", family="monospace",
        )

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba()).copy()
    fig.clear()
    return buf


def _write_gif(frames: Iterator[np.ndarray], path: Path, duration_ms: int) -> Path:
    """Stream RGBA frames to an animated GIF via Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    first = next(frames)
    img = Image.fromarray(first, "RGBA")
    img.save(
        path,
        format="GIF",
        save_all=True,
        append_images=(Image.fromarray(f, "RGBA") for f in frames),
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return path.resolve()
