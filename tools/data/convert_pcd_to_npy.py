#!/usr/bin/env python3
"""Convert MarScenes3D PCD files to OpenPCDet-compatible N x 4 NumPy arrays."""

import argparse
from pathlib import Path
import struct
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _read_header(stream) -> Tuple[Dict[str, List[str]], bytes]:
    header = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD header has no DATA line")
        decoded = line.decode("ascii").strip()
        if not decoded or decoded.startswith("#"):
            continue
        fields = decoded.split()
        key = fields[0].upper()
        header[key] = fields[1:]
        if key == "DATA":
            return header, stream.read()


def _field_layout(header: Dict[str, List[str]]) -> Tuple[List[str], List[int], List[str], List[int]]:
    names = header.get("FIELDS")
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = header.get("TYPE")
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(names or []))]
    if not names or len(names) != len(sizes) or len(names) != len(types or []) or len(names) != len(counts):
        raise ValueError("PCD FIELDS, SIZE, TYPE, and COUNT headers are inconsistent")
    return names, sizes, types, counts


def _as_float32(values: np.ndarray, field: str) -> np.ndarray:
    if isinstance(values, dict):
        if field not in values:
            raise ValueError(f"PCD is missing required field {field!r}")
        result = np.asarray(values[field], dtype=np.float32)
    else:
        if field not in values.dtype.names:
            raise ValueError(f"PCD is missing required field {field!r}")
        result = np.asarray(values[field], dtype=np.float32)
    if result.ndim > 1:
        result = result[:, 0]
    return result


def read_pcd(path: Path) -> np.ndarray:
    """Read ASCII or binary PCD into an ``N x 4`` float32 array."""
    with path.open("rb") as stream:
        header, payload = _read_header(stream)

    names, sizes, types, counts = _field_layout(header)
    data_type = header["DATA"][0].lower()
    if data_type == "binary_compressed":
        raise ValueError("binary_compressed PCD is not supported; export ASCII or binary PCD first")

    points = int(header.get("POINTS", [header.get("WIDTH", ["0"])[0]])[0])
    scalar_count = sum(counts)
    if data_type == "ascii":
        values = np.fromstring(payload.decode("ascii"), sep=" ", dtype=np.float32)
        expected = points * scalar_count
        if values.size != expected:
            raise ValueError(f"{path}: expected {expected} ASCII values, found {values.size}")
        values = values.reshape(points, scalar_count)
        offsets = np.cumsum([0, *counts])
        columns = {}
        for index, name in enumerate(names):
            start, end = offsets[index], offsets[index + 1]
            columns[name] = values[:, start:end][:, 0] if end - start == 1 else values[:, start:end]
    elif data_type == "binary":
        dtype_fields = []
        for name, size, value_type, count in zip(names, sizes, types, counts):
            if value_type == "F" and size == 4:
                dtype = "f4"
            elif value_type == "F" and size == 8:
                dtype = "f8"
            elif value_type == "U" and size == 1:
                dtype = "u1"
            elif value_type == "I" and size == 1:
                dtype = "i1"
            elif value_type == "I" and size == 2:
                dtype = "i2"
            elif value_type == "I" and size == 4:
                dtype = "i4"
            else:
                raise ValueError(f"Unsupported PCD field type: {name} {value_type}{size}")
            dtype_fields.append((name, dtype, (count,)) if count > 1 else (name, dtype))
        dtype = np.dtype(dtype_fields)
        expected = points * dtype.itemsize
        if len(payload) != expected:
            raise ValueError(f"{path}: expected {expected} binary bytes, found {len(payload)}")
        columns = np.frombuffer(payload, dtype=dtype, count=points)
    else:
        raise ValueError(f"Unsupported PCD DATA mode: {data_type}")

    intensity_name = next((name for name in ("intensity", "reflectivity") if name in names), None)
    if intensity_name is None:
        raise ValueError("PCD must contain an intensity or reflectivity field")
    return np.column_stack([
        _as_float32(columns, "x"),
        _as_float32(columns, "y"),
        _as_float32(columns, "z"),
        _as_float32(columns, intensity_name),
    ]).astype(np.float32, copy=False)


def convert(input_path: Path, output_path: Path, overwrite: bool = False) -> int:
    files = [input_path] if input_path.is_file() else sorted(input_path.rglob("*.pcd"))
    if not files:
        raise FileNotFoundError(f"No PCD files found under {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    for source in files:
        relative = source.relative_to(input_path) if input_path.is_dir() else Path(source.name)
        target = (output_path / relative).with_suffix(".npy")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Output exists (use --overwrite): {target}")
        np.save(target, read_pcd(source))
    return len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PCD file or directory")
    parser.add_argument("output", type=Path, help="Output file or directory")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    count = convert(args.input, args.output, overwrite=args.overwrite)
    print(f"Converted {count} PCD file(s) to float32 N x 4 NumPy arrays")


if __name__ == "__main__":
    main()
