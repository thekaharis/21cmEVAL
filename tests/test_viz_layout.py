"""Layout rules: one shared model legend above the panels, per-panel values below, square cyl maps."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eval21.viz.common import model_handles, pane_values, shared_legend, style_key, wrap_label  # noqa: E402


class _Cat:
    models = ["a", "b"]

    def color(self, m):
        return {"a": "r", "b": "k"}[m]

    def ls(self, m):
        return {"a": "-", "b": "--"}[m]

    def label(self, m):
        return {"a": "fine-tuned, 44 blocks, pointwise (LoReLi v3)", "b": "reference (21cmFAST, zero-shot)"}[m]


def _figure(tmp_path):
    from eval21.viz.common import _place_below_boxes, _place_shared_legend

    fig, axes = plt.subplots(2, 3, figsize=(9, 5))
    for ax in axes.ravel():
        ax.plot([0, 1], [0, 1], color="r", label=_Cat().label("a"))
        ax.set_title("panel")
        ax.set_xlabel("x")
        pane_values(ax, [("r", "-", 0.123), ("k", "--", 0.456)], label="W1")
    fig.tight_layout()
    shared_legend(fig, handles=model_handles(_Cat()),
                  extra=style_key([("16–84 %", {"kind": "band", "color": "0.5", "alpha": 0.3})]))
    _place_below_boxes(fig)                     # what save() does before writing (save() clears the figure)
    _place_shared_legend(fig, fig._eval21_legend)
    out = tmp_path / "layout.png"
    fig.savefig(out, bbox_inches="tight")
    return fig, axes, out


def test_shared_legend_is_single_and_outside(tmp_path):
    fig, axes, out = _figure(tmp_path)
    assert out.exists()
    assert all(ax.get_legend() is None for ax in axes.ravel())       # no per-panel legends
    assert len(fig.legends) == 1
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert texts[0] == "truth" and "16–84 %" in texts
    renderer = fig.canvas.get_renderer()
    legend_box = fig.legends[0].get_window_extent(renderer)
    top = max(ax.get_tightbbox(renderer).y1 for ax in axes.ravel())
    assert legend_box.y0 >= top - 1.0                                  # above every panel (titles included)


def test_pane_values_below_axis_label(tmp_path):
    fig, axes, _ = _figure(tmp_path)
    fig.canvas.draw()                           # anchored boxes take their final offset on draw
    renderer = fig.canvas.get_renderer()
    for row in (0, 1):
        ax = axes[row, 0]
        box = ax._eval21_below[0].get_window_extent(renderer)
        assert box.y1 <= ax.xaxis.label.get_window_extent(renderer).y0 + 1.0
    # no collision with the title of the panel underneath
    upper = axes[0, 0]._eval21_below[0].get_window_extent(renderer)
    assert upper.y0 >= axes[1, 0].title.get_window_extent(renderer).y1 - 0.5


def test_wrap_label():
    assert max(len(s) for s in wrap_label(_Cat().label("a"), 22).split("\n")) <= 22
    assert wrap_label("short") == "short"


def test_cyl_maps_square(viz_fixture, tmp_path, monkeypatch):
    import eval21.viz.cylindrical as cylindrical
    import eval21.viz.spectra as spectra
    from eval21.viz import registry
    from eval21.viz.common import Catalog, find_runs

    cat = Catalog(find_runs([viz_fixture / "results"]), quiet=True)
    ctx = registry.PlotContext(catalog=cat, out=tmp_path, quiet=True)
    shapes = {}

    def grab(fig, out, name, **kw):          # measure the map panels as drawn, then skip writing
        fig.canvas.draw()
        shapes[name] = [ax.get_window_extent() for ax in fig.axes
                        if ax.get_xscale() == "log" and ax.get_yscale() == "log" and ax.collections]
        plt.close(fig)
        return out / f"{name}.png"

    figures = registry.load_all()
    for module in (spectra, cylindrical):
        monkeypatch.setattr(module, "save", grab)
    for name in ("ps_cyl_r", "cyl"):
        figures[name].func(ctx)
    assert any(name.startswith("ps_cyl_r") for name in shapes)
    assert any(name.startswith("cyl_error") for name in shapes)
    for name, boxes in shapes.items():
        assert boxes, name
        for box in boxes:
            assert abs(box.width / box.height - 1.0) < 0.02, (name, box.width, box.height)
