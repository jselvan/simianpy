from os import PathLike
from pathlib import Path
from typing import Tuple, Iterator, overload, Literal, Any

import numpy as np
from numpy.typing import ArrayLike
from tqdm import tqdm

@overload
def extract_snippets(path: PathLike[str] | str,
    indices: ArrayLike,
    width: int,
    n_channels: int,
    pbar: bool,
    scale: float,
    dask: Literal[False]
) -> np.ndarray: ...

@overload
def extract_snippets(path: PathLike[str] | str,
    indices: ArrayLike,
    width: int,
    n_channels: int,
    pbar: bool,
    scale: float,
    dask: Literal[True]
) -> da.Array: ...

def extract_snippets(
    path: PathLike[str] | str,
    indices: ArrayLike,
    width: int,
    n_channels: int,
    pbar: bool = False,
    scale: float = 1.0,
    dask: bool = False,
) -> Any:
    alldata = []
    datasize = np.dtype(np.int16).itemsize
    filesize = Path(path).stat().st_size
    indices = np.asarray(indices)

    if dask:
        import dask.array as da
        alldata = da.full(
            (*indices.shape, width, n_channels),
            np.nan,
            dtype=np.float16,
            chunks=("auto",) * indices.ndim + (-1, -1),
        )
    else:
        alldata = np.full((*indices.shape, width, n_channels), np.nan, dtype=np.float16)

    indices_sort_idx = np.argsort(indices, axis=None)
    if (indices.max() + width) * n_channels * datasize > filesize:
        raise ValueError("Indices and width exceed file size.")
    if indices.min() < 0:
        raise ValueError("Indices must be non-negative.")
    if pbar is True:
        indices_wrapped: Iterator[int] = tqdm(
            indices_sort_idx, desc="Extracting snippets"
        )  # type:ignore
    else:
        indices_wrapped: Iterator[int] = iter(indices_sort_idx)
    with open(path, "rb") as f:
        for idx in indices_wrapped:
            idx_unravelled = np.unravel_index(idx, indices.shape)
            i = indices[idx_unravelled].item()
            if np.isnan(i):
                continue
            else:
                position = int(i * n_channels * datasize)
            f.seek(position)
            data = np.fromfile(f, dtype=np.int16, count=n_channels * width).reshape(
                (width, n_channels)
            )
            data = (data * scale).astype(np.float16)
            alldata[*idx_unravelled, ...] = data
    return alldata


def extract_windows_seconds(
    path: PathLike[str] | str,
    times: ArrayLike,
    window: Tuple[float, float],
    n_channels: int,
    sampling_rate: float,
    scale: float = 1.0,
    pbar: bool = False,
    dask: bool = False
):
    """Extract snippets from a binary file based on specified times and window.

    :param path: Path to the binary file.
    :param times: Array of times in seconds at which to extract snippets.
    :param window: Tuple specifying the left and right bounds of the window in seconds (e.g., -1 to 1).
    :param n_channels: Number of channels in the binary file.
    :param sampling_rate: Sampling rate of the data in Hz.
    :return: Extracted snippets as a NumPy array.
    """
    left, right = window[0], window[1]
    width = int((right - left) * sampling_rate)
    times = np.asarray(times)
    indices = np.round((times + left) * sampling_rate)
    return extract_snippets(path, indices, width, n_channels, pbar=pbar, scale=scale, dask=dask)


def extract_windows_samples(
    path: PathLike[str] | str,
    samples: ArrayLike,
    window: Tuple[int, int],
    n_channels: int,
    scale: float = 1.0,
    pbar: bool = False,
    dask: bool = False
):
    """Extract snippets from a binary file based on specified sample indices and window.

    :param path: Path to the binary file.
    :param samples: Array of sample indices at which to extract snippets.
    :param window: Tuple specifying the left and right bounds of the window in samples (e.g., -1000 to 1000).
    :param n_channels: Number of channels in the binary file.
    :return: Extracted snippets as a NumPy array.
    """
    left, right = window[0], window[1]
    width = right - left
    samples = np.asarray(samples)
    indices = samples + left
    return extract_snippets(path, indices, width, n_channels, pbar=pbar, scale=scale, dask=dask)
