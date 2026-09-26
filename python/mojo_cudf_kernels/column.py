"""A cudf-shaped column: a float64 buffer plus a packed validity bitmask.

cudf stores validity as a packed bitmask of `bitmask_type` (one bit per row,
LSB first within each byte) alongside the data buffer, and that is exactly the
layout the Mojo kernels read and write, so the two agree bit for bit.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["Column", "pack_validity", "unpack_validity"]


def pack_validity(valid: np.ndarray | None) -> np.ndarray | None:
    """Pack a boolean validity array into cudf's `bitmask_type` layout."""
    if valid is None:
        return None
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 1:
        raise ValueError("validity must be one-dimensional")
    if valid.size == 0:
        return np.zeros(0, dtype=np.uint8)
    words = np.packbits(valid.astype(np.uint8), bitorder="little")
    return np.ascontiguousarray(words)


def unpack_validity(bitmask: np.ndarray | None, size: int) -> np.ndarray:
    """Unpack a cudf validity bitmask; a missing mask means every row is valid."""
    if bitmask is None or bitmask.size == 0:
        return np.ones(size, dtype=bool)
    bits = np.unpackbits(np.asarray(bitmask, dtype=np.uint8), bitorder="little")
    return bits[:size].astype(bool)


class Column:
    """One float64 column with an optional packed validity mask.

    `nulls` is the cudf `size_type` count of invalid entries; the packed mask
    is always exactly `ceil(size / 8)` bytes.
    """

    __slots__ = ("data", "mask", "size", "nulls")

    def __init__(
        self,
        data: Any,
        valid: np.ndarray | None = None,
        nulls: int | None = None,
        mask: np.ndarray | None = None,
    ) -> None:
        values = np.asarray(data, dtype=np.float64).reshape(-1)
        self.data = np.ascontiguousarray(values)
        self.size = int(self.data.size)
        if mask is not None:
            if valid is not None:
                raise ValueError("pass either a validity array or a packed mask")
            self.mask = np.ascontiguousarray(mask, dtype=np.uint8)
        else:
            self.mask = pack_validity(valid)
        if self.mask is not None and self.size:
            expected = (self.size + 7) >> 3
            if self.mask.size != expected:
                raise ValueError(
                    f"validity mask has {self.mask.size} bytes, expected {expected}"
                )
        self.nulls = (
            int(nulls) if nulls is not None else self._count_nulls()
        )
        if self.nulls < 0 or self.nulls > self.size:
            raise ValueError("nulls must be between 0 and the column size")

    def _count_nulls(self) -> int:
        if self.mask is None:
            return 0
        valid = unpack_validity(self.mask, self.size)
        return int(self.size - valid.sum())

    @property
    def valid(self) -> np.ndarray:
        return unpack_validity(self.mask, self.size)

    @classmethod
    def from_mask(cls, data: Any, mask: np.ndarray) -> "Column":
        """Rebuild a column from raw values and a packed mask, as the kernels
        produce them. The null count is recovered by popcount rather than by a
        Python-level scan of every row."""
        return cls(data, mask=mask)

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return f"Column(size={self.size}, nulls={self.nulls})"
