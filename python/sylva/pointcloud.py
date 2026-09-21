"""Numpy-backed point cloud container."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointCloud:
    """A set of 3D points with optional per-point attributes.

    Parameters
    ----------
    xyz
        ``(N, 3)`` array of coordinates, stored as C-contiguous float64.
    attrs
        Mapping of attribute name to a length-``N`` array (``intensity``,
        ``classification``, ``height``, ...). Names follow laspy conventions
        so LAS files round-trip.
    """

    xyz: np.ndarray
    attrs: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.xyz = np.ascontiguousarray(np.asarray(self.xyz, dtype=np.float64))
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {self.xyz.shape}")
        self.attrs = {k: np.ascontiguousarray(np.asarray(v)) for k, v in dict(self.attrs).items()}
        for name, values in self.attrs.items():
            if values.ndim != 1 or len(values) != len(self.xyz):
                raise ValueError(
                    f"attribute {name!r} has shape {values.shape}, expected ({len(self.xyz)},)"
                )

    def __len__(self) -> int:
        return len(self.xyz)

    def __repr__(self) -> str:
        return f"PointCloud(n={len(self):,}, attrs={sorted(self.attrs)})"

    def __getitem__(self, index) -> PointCloud:
        """Subset by boolean mask, integer indices or slice."""
        return PointCloud(self.xyz[index], {k: v[index] for k, v in self.attrs.items()})

    @property
    def x(self) -> np.ndarray:
        return self.xyz[:, 0]

    @property
    def y(self) -> np.ndarray:
        return self.xyz[:, 1]

    @property
    def z(self) -> np.ndarray:
        return self.xyz[:, 2]

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(min_xyz, max_xyz)``."""
        return self.xyz.min(axis=0), self.xyz.max(axis=0)

    def heights(self, attr: str = "height") -> np.ndarray:
        """The ``height`` attribute if present, else z (assumed normalised)."""
        return np.asarray(self.attrs[attr], dtype=np.float64) if attr in self.attrs else self.z

    def copy(self) -> PointCloud:
        return PointCloud(self.xyz.copy(), {k: v.copy() for k, v in self.attrs.items()})

    def with_attrs(self, **attrs: np.ndarray) -> PointCloud:
        """Return a shallow copy with attributes added or replaced."""
        return PointCloud(self.xyz, {**self.attrs, **attrs})

    def without(self, *names: str) -> PointCloud:
        return PointCloud(self.xyz, {k: v for k, v in self.attrs.items() if k not in names})

    def transform(self, matrix: np.ndarray) -> PointCloud:
        """Apply a 4x4 homogeneous transform and return a new cloud."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("matrix must be 4x4")
        xyz = self.xyz @ matrix[:3, :3].T + matrix[:3, 3]
        return PointCloud(xyz, dict(self.attrs))

    @classmethod
    def concatenate(cls, clouds: Iterable[PointCloud]) -> PointCloud:
        """Merge clouds, keeping only attributes present in all of them."""
        clouds = list(clouds)
        if not clouds:
            raise ValueError("no clouds to concatenate")
        common = set.intersection(*(set(c.attrs) for c in clouds))
        return cls(
            np.vstack([c.xyz for c in clouds]),
            {k: np.concatenate([c.attrs[k] for c in clouds]) for k in sorted(common)},
        )

    @classmethod
    def from_array(cls, array: np.ndarray, names: Mapping[int, str] | None = None) -> PointCloud:
        """Build from an ``(N, 3 + k)`` array; extra columns become attributes."""
        array = np.asarray(array)
        names = names or {}
        attrs = {names.get(i, f"col{i}"): array[:, i] for i in range(3, array.shape[1])}
        return cls(array[:, :3], attrs)
