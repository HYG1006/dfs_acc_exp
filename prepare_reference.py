#!/usr/bin/env python3
"""Stream ADM's ImageNet reference NPZ into a memory-mapped NPY."""

import argparse
from pathlib import Path
import zipfile

import numpy as np
from tqdm.auto import tqdm


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"expected {size} bytes, got {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def array_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version in {(2, 0), (3, 0)}:
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f"unsupported NPY version in NPZ: {version}")


def convert(input_path, output_path, array_name="arr_0", chunk_size=32):
    with zipfile.ZipFile(input_path, "r") as archive:
        member_name = f"{array_name}.npy"
        if member_name not in archive.namelist():
            raise ValueError(f"{member_name} not found in {input_path}")
        with archive.open(member_name, "r") as stream:
            shape, fortran, dtype = array_header(stream)
            if fortran or dtype != np.uint8 or len(shape) != 4 or shape[-1] != 3:
                raise ValueError(f"expected C-order NHWC uint8 reference images, got {shape}, {dtype}")
            output = np.lib.format.open_memmap(output_path, mode="w+", dtype=dtype, shape=shape)
            pixels_per_image = int(np.prod(shape[1:]))
            for start in tqdm(range(0, shape[0], chunk_size), desc="extracting reference"):
                count = min(chunk_size, shape[0] - start)
                raw = read_exact(stream, count * pixels_per_image * dtype.itemsize)
                output[start : start + count] = np.frombuffer(raw, dtype=dtype).reshape((count, *shape[1:]))
            output.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    convert(args.input, output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

