#!/usr/bin/env python3
"""
the main functionality of this script is to pick a random 3D image from an h5 file
and display it in 3D slicer. If the randomly picked image is a segmentation script
display the image using "view_seg.py" and if its a CT image (preprocessed but not masked)
script display that image via "view_img.py"

Run:
    python pick_and_display.py /abs/path/to/h5file

Optional arguments:
    -- kind <mask|image> -- image is picked randomly by default, one can use kind argument to constrain the selection
    -- bg                -- if selected image is a segmentation, you can display the CT image as background if --bg is set

*view main() for other type of optional arguments (mostly path related)

"""

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from typing import List, Optional

import numpy as np
import h5py
import SimpleITK as sitk


META_SUFFIX   = "_mask_name_lst"
MASK_TOKEN    = "mask_tibia"
IMG_NAME_RE   = re.compile(r"^\d+_tibia_[LR](?:\.nii(?:\.gz)?)?$")

def strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"): return name[:-7]
    if name.endswith(".nii"):    return name[:-4]
    return name


def is_metadata(name: str) -> bool:
    return name.endswith(META_SUFFIX)


def classify_kind(name: str) -> str:
    base = strip_ext(name)
    if MASK_TOKEN in base:
        return "mask"
    if IMG_NAME_RE.match(name) or IMG_NAME_RE.match(base):
        return "image"
    return "other"

def is_numeric_dtype(dt) -> bool:
    return np.dtype(dt).kind in ("i", "u", "f")


def squeeze_to_3d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    a = np.squeeze(a)
    if a.ndim != 3:
        raise ValueError(f"Expected 3D after squeeze; got {a.shape}")
    return a


def list_candidates(h5: "h5py.File", force_kind: Optional[str] = None):
    out = []
    def visit(name, obj):
        import h5py as _h5py
        if not isinstance(obj, _h5py.Dataset):
            return
        if is_metadata(name):
            return
        if not is_numeric_dtype(obj.dtype):
            return
        kind = classify_kind(name)
        if kind == "other":
            return
        if force_kind in ("image", "mask") and kind != force_kind:
            return
        shape = tuple(int(x) for x in obj.shape)
        if len(shape) < 3 or any(int(d) <= 0 for d in shape[-3:]):
            return
        out.append({"name": name, "kind": kind, "shape": shape, "dtype": str(obj.dtype)})
    h5.visititems(visit)
    return out


def base_from_mask(name: str) -> str:
    # "9_tibia_R_mask_tibia_R" -> "9_tibia_R"
    return name.split("_" + MASK_TOKEN, 1)[0] if MASK_TOKEN in name else name


def temp_nifti_from_array(arr: np.ndarray) -> str:
    """Write array (Z,Y,X) to a temporary .nii.gz. Returns path; caller should delete it later."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr))
    f = tempfile.NamedTemporaryFile(prefix="slicer_tmp_", suffix=".nii.gz", delete=False)
    p = f.name
    f.close()
    sitk.WriteImage(img, p)
    return p


def default_slicer_path() -> str:
    # Try a smart default for macOS; otherwise use "Slicer" on PATH
    mac = "/Applications/Slicer.app/Contents/MacOS/Slicer"
    return mac if sys.platform == "darwin" and os.path.exists(mac) else "Slicer"


def ensure_exists(p: Optional[str], description: str):
    if p is None or not os.path.exists(p):
        raise FileNotFoundError(f"{description} not found: {p}")


def run_slicer(slicer_bin: str, args: List[str]) -> int:
    cmd = [slicer_bin] + args
    print("[SLICER]", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


def main():
    here = os.path.abspath(os.path.dirname(__file__))

    ap = argparse.ArgumentParser(description="Randomly pick a dataset from HDF5 and view it in Slicer (no persistent files).")
    ap.add_argument("h5_path", help="Absolute path to .h5/.hdf5 file")
    ap.add_argument("--slicer", default=default_slicer_path(), help="Path to Slicer executable")
    ap.add_argument("--view-seg", default=os.path.join(here, "view_seg.py"), help="Path to view_seg.py")
    ap.add_argument("--view-img", default=os.path.join(here, "view_img.py"), help="Path to view_img.py")
    ap.add_argument("--kind", choices=["image", "mask"], default=None, help="Restrict random selection to type")
    ap.add_argument("--threshold", type=float, default=0.5, help="Float mask binarization threshold")
    ap.add_argument("--bg", action="store_true", help="If a mask is picked, also load its matching CT as background (default: off)")

    args = ap.parse_args()

    if not os.path.isabs(args.h5_path):
        print("Please provide an absolute path to the HDF5 file.", file=sys.stderr)
        sys.exit(2)

    ensure_exists(args.h5_path, "HDF5 file")
    ensure_exists(args.slicer, "Slicer executable")
    ensure_exists(args.view_seg, "view_seg.py")
    ensure_exists(args.view_img, "view_img.py")

    tmp_paths: List[str] = []
    try:
        with h5py.File(args.h5_path, "r") as f:
            # choose candidate
            cands = list_candidates(f, force_kind=args.kind)
            if not cands:
                print("ERROR: No valid datasets found. Check naming rules and --kind.", file=sys.stderr)
                sys.exit(3)

            rng = np.random.default_rng()
            pick = cands[rng.integers(0, len(cands))]
            name, kind = pick["name"], pick["kind"]
            print(f"[PICK] {name}  kind={kind}  shape={pick['shape']}  dtype={pick['dtype']}")

            arr = squeeze_to_3d(f[name][()])

            if kind == "image":
                img_path = temp_nifti_from_array(arr.astype(np.float32, copy=False))
                tmp_paths.append(img_path)
                rc = run_slicer(args.slicer, ["--python-script", args.view_img, "--", "--image", img_path])
                sys.exit(rc)

            # kind == "mask": convert mask + try to find background
            dt = f[name].dtype
            if np.dtype(dt).kind in ("i", "u"):
                lab = (arr > 0).astype(np.uint8, copy=False)
            else:
                lab = (arr > float(args.threshold)).astype(np.uint8, copy=False)
            mask_path = temp_nifti_from_array(lab)
            tmp_paths.append(mask_path)

            # Optional background CT only if --bg was passed
            bg_path = None
            if args.bg:
                base = base_from_mask(name)  # e.g., "9_tibia_R" from "9_tibia_R_mask_tibia_R"
                # Be tolerant of dataset keys with extensions
                for bg_key in (base, base + ".nii", base + ".nii.gz"):
                    if bg_key in f and classify_kind(bg_key) == "image":
                        bg_arr = squeeze_to_3d(f[bg_key][()])
                        bg_path = temp_nifti_from_array(bg_arr.astype(np.float32, copy=False))
                        tmp_paths.append(bg_path)
                        break

            slicer_args = ["--python-script", args.view_seg, "--", "--mask", mask_path]
            if args.bg and bg_path:
                slicer_args += ["--background", bg_path]
            rc = run_slicer(args.slicer, slicer_args)
            sys.exit(rc)

    finally:
        # cleanup temp files
        for p in tmp_paths:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


if __name__ == "__main__":
    main()
