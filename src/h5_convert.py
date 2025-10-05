#!/usr/bin/env python3
"""
h5_convert.py — Pick a random valid dataset (CT image or mask) from an HDF5 file
and convert it to NIfTI (.nii.gz), skipping metadata entries like "*_mask_name_lst".

Run :
    python h5_convert.py <abs_path_h5_file> --out <output_dir_path>
"""

import argparse
import os
import re
from typing import List, Tuple, Optional, Dict

import h5py
import numpy as np
import nibabel as nib


def is_metadata_name(name: str) -> bool:
    """Heuristic to skip metadata datasets by name."""
    return name.endswith("_mask_name_lst") or name.lower().endswith("_names")


def is_numeric_dtype(dt) -> bool:
    """True if dtype is numeric (int/float)."""
    kind = np.dtype(dt).kind
    return kind in ("i", "u", "f")


def is_mask_name(name: str) -> bool:
    return "_mask_" in name


def is_3d_like_shape(shape: Tuple[int, ...]) -> bool:
    """Must have at least 3 dims; last-3 dims all > 0."""
    if len(shape) < 3:
        return False
    zyx = shape[-3:]
    return all(int(d) > 0 for d in zyx)


def dataset_kind(name: str, dtype) -> str:
    """Classify dataset name/dtype into 'mask' or 'image' (heuristic)."""
    if is_mask_name(name):
        return "mask"
    # Integer dtypes often indicate labelmaps
    if np.dtype(dtype).kind in ("i", "u"):
        return "mask"
    return "image"


def list_valid_datasets(h5: "h5py.File",
                        force_kind: str = "any") -> List[Dict[str, object]]:
    """
    Return a list of valid datasets with metadata for selection.
    Each entry: {"name": str, "shape": tuple, "dtype": np.dtype, "kind": "image"|"mask"}
    """
    out = []

    def visitor(name, obj):
        if not isinstance(obj, h5py.Dataset):
            return
        # Exclude by name/dtype
        if is_metadata_name(name):
            return
        if not is_numeric_dtype(obj.dtype):
            return
        if not is_3d_like_shape(obj.shape):
            return

        kind = dataset_kind(name, obj.dtype)
        if force_kind in ("image", "mask") and kind != force_kind:
            return

        out.append({
            "name": name,
            "shape": tuple(int(x) for x in obj.shape),
            "dtype": obj.dtype,
            "kind": kind,
        })

    h5.visititems(visitor)
    return out


def squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
    """Squeeze to (Z,Y,X)."""
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D after squeeze, got shape {arr.shape}")
    return arr


def sanitize_filename(name: str) -> str:
    """Make a safe filename from a dataset path/name."""
    base = name.strip("/")
    base = base.replace("/", "_")
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    return base


def convert_one(h5_path: str,
                out_dir: str,
                force_kind: str = "any",
                seed: Optional[int] = None,
                mask_mode: str = "auto",
                threshold: float = 0.5,
                dry_run: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """
    Pick one dataset at random and convert it.
    Returns (output_path, dataset_name) or (None, None) if nothing converted.
    """
    rng = np.random.default_rng(seed)
    with h5py.File(h5_path, "r") as f:
        candidates = list_valid_datasets(f, force_kind=force_kind)

        if not candidates:
            print("[ERROR] No valid datasets found with the given criteria.")
            return None, None

        choice = candidates[rng.integers(0, len(candidates))]
        name = choice["name"]
        kind = choice["kind"]
        shape = choice["shape"]
        dtype = choice["dtype"]
        print(f"[INFO] Selected dataset: {name}  kind={kind}  shape={shape}  dtype={dtype}")

        if dry_run:
            print("[DRY-RUN] No files written.")
            return None, name

        ds = f[name]
        arr = squeeze_to_3d(ds[()])

        os.makedirs(out_dir, exist_ok=True)
        base = sanitize_filename(name)

        if kind == "mask":
            # Convert to labelmap uint8
            if mask_mode == "auto":
                if np.issubdtype(ds.dtype, np.integer):
                    out_arr = arr.astype(np.uint8, copy=False)
                else:
                    out_arr = (arr > threshold).astype(np.uint8, copy=False)
            elif mask_mode == "threshold":
                out_arr = (arr > threshold).astype(np.uint8, copy=False)
            elif mask_mode == "round":
                out_arr = np.rint(arr).astype(np.uint8, copy=False)
            else:
                raise ValueError(f"Unknown mask_mode: {mask_mode}")

            out_path = os.path.join(out_dir, f"{base}.nii.gz")
            nib.save(nib.Nifti1Image(out_arr, np.eye(4)), out_path)
        else:
            # Image volume as float32
            out_arr = arr.astype(np.float32, copy=False)
            out_path = os.path.join(out_dir, f"{base}.nii.gz")
            nib.save(nib.Nifti1Image(out_arr, np.eye(4)), out_path)

        print(f"[OK] Wrote: {out_path}")
        return out_path, name


def main():
    ap = argparse.ArgumentParser(description="Pick a random valid dataset from an HDF5 file and convert to NIfTI.")
    ap.add_argument("h5_path", help="Path to .h5/.hdf5 file")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--kind", choices=["any", "image", "mask"], default="any", help="Which kind of dataset to select")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--mask-mode", choices=["auto", "threshold", "round"], default="auto", help="How to convert mask values")
    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing float masks (if used)")
    ap.add_argument("--dry-run", action="store_true", help="Do selection only, do not write files")
    args = ap.parse_args()

    if not os.path.exists(args.h5_path):
        ap.error(f"File not found: {args.h5_path}")

    convert_one(args.h5_path, args.out, force_kind=args.kind, seed=args.seed,
                mask_mode=args.mask_mode, threshold=args.threshold, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
