"""ARC grid validity and rule-preserving augmentation, shared by the induction and transduction lines.

Validity: `normalize_grid` returns a JSON-compatible grid or None; `grid_key` is a hashable canonical form.

Rule-preserving test-time views shared by grid and code generation.

A *view* is an ARC-rule-preserving disguise of a whole task: a D8 dihedral
symmetry (optional transpose then k 90-degree rotations) x a color permutation
(the 10 colors are arbitrary labels) x a demonstration-order shuffle. The frozen
model solves each disguised view; `invert_grid` maps its predicted grid back to
the canonical frame so candidates from different views pool together.

These are the same symmetries NVARC's per-task TTT augments over (cf.
`arc_loader.permute_mod` and `ArcDataset.augment`); here they drive prompt-level
TTA instead of building a training set. Color and geometry commute, so forward
applies color then geometry and `invert_grid` undoes geometry then color.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Grid = list[list[int]]


def normalize_grid(value: Any) -> Grid | None:
    """Return a JSON-compatible ARC grid, or ``None`` for invalid values."""
    try:
        array = np.asarray(value)
        if array.ndim != 2 or array.size == 0 or any(not 0 < size <= 30 for size in array.shape):
            return None
        if not np.issubdtype(array.dtype, np.integer):
            if not np.all(np.equal(array, np.floor(array))):
                return None
        array = array.astype(int)
        if array.min() < 0 or array.max() > 9:
            return None
        return array.tolist()
    except (TypeError, ValueError, OverflowError):
        return None


def serialize_spaced(grid) -> str:
    """Serialize a grid as rows of space-separated digits."""
    return "\n".join(" ".join(str(int(cell)) for cell in row) for row in grid)


def is_valid_grid(value: Any) -> bool:
    return normalize_grid(value) is not None


def grid_key(value: Any) -> tuple[tuple[int, ...], ...]:
    grid = normalize_grid(value)
    if grid is None:
        raise ValueError("value is not a valid ARC grid")
    return tuple(tuple(row) for row in grid)

# The 8 elements of the dihedral group of the square: optionally transpose, then
# rotate by k * 90 degrees. (transpose, k) enumerates all of D8 exactly once.
D8 = [(t, k) for t in (False, True) for k in range(4)]


@dataclass(frozen=True)
class View:
    """One TTA disguise. Frozen so it is hashable (used for de-duplication).

    transpose / k: the D8 geometry. color: a length-10 permutation, color[c] is
    the new label for cell value c. demo_order: a permutation of the demo indices.
    """
    transpose: bool
    k: int
    color: tuple  # length-10 permutation of 0..9
    demo_order: tuple  # permutation of range(n_demos)


def d8_transform(grid, transform_index: int) -> list:
    """Apply one of the eight square-grid symmetries."""
    if not 0 <= transform_index < len(D8):
        raise ValueError("transform_index must be in [0, 7]")
    g = np.asarray(grid, dtype=int)
    if transform_index >= 4:
        g = np.fliplr(g)
    return np.rot90(g, transform_index % 4).tolist()


def apply_grid(grid, view: View) -> list:
    """Forward: relabel colors, then apply the D8 geometry. Returns nested ints."""
    g = np.asarray(grid, dtype=int)
    g = np.asarray(view.color, dtype=int)[g]  # color[c] relabel, per cell
    if view.transpose:
        g = g.T
    if view.k:
        g = np.rot90(g, view.k)
    return g.tolist()


def invert_grid(grid, view: View) -> list:
    """Inverse of `apply_grid`: undo the geometry, then undo the color relabel."""
    g = np.asarray(grid, dtype=int)
    if view.k:
        g = np.rot90(g, -view.k)
    if view.transpose:
        g = g.T
    inv_color = np.argsort(np.asarray(view.color, dtype=int))  # inverse permutation
    g = inv_color[g]
    return g.tolist()


def apply_task(task: dict, view: View) -> dict:
    """Disguise a whole task: reorder demos and transform every grid. Test outputs
    are unknown, so only test *inputs* are transformed."""
    train = [task["train"][i] for i in view.demo_order]
    new_train = [
        {"input": apply_grid(p["input"], view), "output": apply_grid(p["output"], view)}
        for p in train
    ]
    new_test = [{"input": apply_grid(t["input"], view)} for t in task["test"]]
    return {"train": new_train, "test": new_test}


def make_views(n_demos: int, n_views: int, seed: int = 0) -> list:
    """Sample `n_views` distinct views. The identity view is always first so the
    untransformed task is one of the candidates."""
    rng = np.random.default_rng(seed)
    identity = View(False, 0, tuple(range(10)), tuple(range(n_demos)))
    views = [identity]
    seen = {identity}
    max_attempts = max(50, n_views * 50)
    attempts = 0
    while len(views) < n_views and attempts < max_attempts:
        attempts += 1
        transpose, k = D8[int(rng.integers(len(D8)))]
        color = tuple(int(x) for x in rng.permutation(10))
        demo_order = tuple(int(x) for x in rng.permutation(n_demos))
        v = View(bool(transpose), int(k), color, demo_order)
        if v in seen:
            continue
        seen.add(v)
        views.append(v)
    return views


__all__ = [
    "D8",
    "View",
    "apply_grid",
    "apply_task",
    "d8_transform",
    "invert_grid",
    "is_valid_grid",
    "make_views",
]
