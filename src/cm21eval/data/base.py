"""Dataset interface."""
from __future__ import annotations

from typing import Iterator, Protocol

from ..spec import Sample


class Dataset(Protocol):
    def __iter__(self) -> Iterator[Sample]:
        ...

    def __len__(self) -> int:
        ...

    def describe(self) -> dict:
        ...


class BaseDataset:
    """Common split/limit handling."""

    def __init__(self, split: str | None = None, limit: int | None = None,
                 indices: list[int] | None = None):
        self.split = split
        self.limit = limit
        self.indices = indices

    def _visible(self, samples: list[Sample]) -> list[Sample]:
        if self.indices is not None:
            samples = [samples[i] for i in self.indices if 0 <= i < len(samples)]
        if self.split:
            if samples and all(sample.split is None for sample in samples):
                raise ValueError(
                    f"this dataset does not declare splits, so --split "
                    f"{self.split!r} cannot be honoured. Pass --indices or "
                    f"--indices-json to select the evaluation rows explicitly.")
            samples = [s for s in samples if s.split == self.split]
        if self.limit is not None:
            samples = samples[:self.limit]
        return samples

    def __iter__(self) -> Iterator[Sample]:
        yield from self._visible(self._samples())

    def __len__(self) -> int:
        return len(self._visible(self._samples()))

    def _samples(self) -> list[Sample]:
        raise NotImplementedError
