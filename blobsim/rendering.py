"""GIF rendering pipeline for blobsim simulations.

state → frame image (matplotlib) → animated GIF (Pillow).
Streams frames via generator — peak memory is ~2 frames.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from PIL import Image

if TYPE_CHECKING:
    from blobsim.simulation import SimulationResult
    from blobsim.types import BlobRecord, LedgerSnapshot

# ── Constants ────────────────────────────────────────────────────────
GRID_CMAP = "YlGn"
MARKER_SIZE_RANGE = (20, 80)
BLOB_MAX_ENERGY = 20.0


# ── Public API ───────────────────────────────────────────────────────

def render_gif(
    result: SimulationResult,
    output_path: str | Path | None = None,
    *,
    fps: int = 10,
    dpi: int = 72,
    cell_pixels: int = 12,
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

    # Generator — yields RGBA frames, never accumulates
    def frames() -> Iterator[np.ndarray]:
        for t in range(len(recorder.blob_records)):
            yield _render_frame(
                recorder.grid_records[t],
                recorder.blob_records[t],
                recorder.ledger_records[t],
                grid_size,
                palette,
                grid_vmax,
                dpi,
                cell_pixels,
            )

    path = Path(output_path) if output_path else Path(f"blobsim_{int(time.time())}.gif")
    return _write_gif(frames(), path, duration_ms=1000 // fps)


# ── Private helpers ──────────────────────────────────────────────────

def _build_palette(species_names: set[str]) -> dict[str, tuple[float, ...]]:
    """Map sorted species names → deterministic RGBA colors via tab10."""
    cmap = get_cmap("tab10")
    return {name: cmap(i % 10) for i, name in enumerate(sorted(species_names))}


def _render_frame(
    grid: np.ndarray | None,
    blob_records: list[BlobRecord],
    ledger: LedgerSnapshot,
    grid_size: int,
    palette: dict[str, tuple[float, ...]],
    grid_vmax: float,
    dpi: int,
    cell_pixels: int,
) -> np.ndarray:
    """Render a single simulation step to an RGBA numpy array."""
    fig_w = grid_size * cell_pixels / dpi
    fig_h = fig_w  # square
    fig = Figure(figsize=(fig_w + 0.8, fig_h + 0.6), dpi=dpi, facecolor="black")
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0.08, 0.08, fig_w / (fig_w + 0.8), fig_h / (fig_h + 0.6)))

    # Grid heatmap
    if grid is not None:
        im = ax.imshow(
            grid, cmap=GRID_CMAP, vmin=0, vmax=grid_vmax,
            origin="lower", interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Grid energy")
    else:
        ax.imshow(
            np.zeros((grid_size, grid_size)), cmap="Greys", vmin=0, vmax=1,
            origin="lower", interpolation="nearest",
        )
        ax.text(
            grid_size / 2, grid_size / 2, "no grid data",
            ha="center", va="center", color="white", fontsize=8, alpha=0.6,
        )

    # Blobs (alive only)
    alive = [r for r in blob_records if r.alive]
    if alive:
        xs = [r.position[1] for r in alive]  # col → x
        ys = [r.position[0] for r in alive]  # row → y
        colors = [palette.get(r.species, (1, 1, 1, 1)) for r in alive]

        # Size proportional to energy
        lo, hi = MARKER_SIZE_RANGE
        sizes = [
            lo + (hi - lo) * min(r.energy / BLOB_MAX_ENERGY, 1.0)
            for r in alive
        ]

        ax.scatter(
            xs, ys, s=sizes, c=colors,
            edgecolors="white", linewidths=0.5, zorder=2,
        )

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Step {ledger.step} | Alive: {ledger.n_alive}",
        color="white", fontsize=9, pad=4,
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
