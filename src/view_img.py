#!/usr/bin/env python3


"""
Display the CT image in 3D slicer

Run:
    /Applications/Slicer.app/Contents/MacOS/Slicer \
    --python-script /abs/path/view_img.py \
    --image /abs/path/ct_file.nii.gz
"""


import sys
import argparse
import numpy as np
import slicer


def volume_info(node, title="IMAGE VOLUME INFO"):
    img = node.GetImageData()
    dims = img.GetDimensions()
    spacing = node.GetSpacing()
    origin = node.GetOrigin()
    stype = img.GetScalarTypeAsString()
    print(f"\n=== {title} ===")
    print(f"Node name:           {node.GetName()}")
    print(f"Dimensions (voxels): {dims}")
    print(f"Spacing (mm/voxel):  {spacing}")
    print(f"Origin (RAS, mm):    {origin}")
    print(f"Scalar type:         {stype}")
    print(f"Voxel count:         {dims[0]*dims[1]*dims[2]:,}")
    print("================================\n")


def auto_window_level(node, pmin=0.5, pmax=99.5):
    """Set window/level using robust percentiles."""
    arr = slicer.util.arrayFromVolume(node)
    if arr.size == 0:
        return
    arr = arr.astype(np.float32, copy=False)
    lo = float(np.percentile(arr, pmin))
    hi = float(np.percentile(arr, pmax))
    if not node.GetDisplayNode():
        node.CreateDefaultDisplayNodes()
    dn = node.GetDisplayNode()
    if hi <= lo:
        hi = lo + 1.0
    dn.AutoWindowLevelOff()
    dn.SetWindow(hi - lo)
    dn.SetLevel((hi + lo) / 2.0)
    print(f"[window/level] p{pmin}={lo:.3f}, p{pmax}={hi:.3f} -> window={hi-lo:.3f}, level={(hi+lo)/2.0:.3f}")


def enable_volume_rendering(volume_node, preset_name="CT-Bone"):
    """
    Enable 3D volume rendering on the given scalar volume node.
    Tries to apply a CT preset (e.g., 'CT-Bone'); falls back to defaults if unavailable.
    """
    vrLogic = slicer.modules.volumerendering.logic()
    # Create default VR nodes (display + ROI + property)
    display_node = vrLogic.CreateDefaultVolumeRenderingNodes(volume_node)

    # Try to copy a preset volume property into the main scene
    try:
        presets_scene = vrLogic.GetPresetsScene()  # a separate MRML scene containing presets
        if presets_scene:
            preset = presets_scene.GetFirstNodeByName(preset_name)
            if preset:
                vp_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLVolumePropertyNode', f'{preset_name}_VP')
                vp_node.Copy(preset)
                display_node.SetAndObserveVolumePropertyNodeID(vp_node.GetID())
                print(f"[VR] Applied preset: {preset_name}")
            else:
                print(f"[VR] Preset '{preset_name}' not found; using default volume property.")
    except Exception as e:
        print(f"[VR] Could not apply preset '{preset_name}': {e}")

    # Fit ROI and make it visible (optional)
    try:
        vrLogic.FitROIToVolume(display_node)
        display_node.SetCroppingEnabled(False)
    except Exception as e:
        print(f"[VR] Fit ROI failed (non-fatal): {e}")

    # Show in 3D
    display_node.SetVisibility(True)
    return display_node


def show_image(image_path):
    node = slicer.util.loadVolume(image_path)
    if not node:
        raise RuntimeError(f"Failed to load image: {image_path}")
    volume_info(node)
    auto_window_level(node, pmin=0.5, pmax=99.5)
    slicer.util.setSliceViewerLayers(background=node, fit=True)

    # Switch to Four-Up and reset 3D camera
    lm = slicer.app.layoutManager()
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    lm.threeDWidget(0).threeDView().resetCamera()

    # Enable 3D volume rendering with a bone-friendly preset (good for tibia)
    enable_volume_rendering(node, preset_name="CT-Bone")

    print("[DONE] Shown CT image with auto window/level and 3D volume rendering.")
    return node


def parse_args(argv):
    p = argparse.ArgumentParser(description="Display a CT NIfTI in Slicer with auto window/level and 3D rendering.")
    p.add_argument("--image", required=True, help="Path to .nii or .nii.gz image")
    return p.parse_args(argv)


def main(argv):
    # In Slicer, script args appear after a bare "--"
    try:
        idx = argv.index("--")
        args = argv[idx+1:]
    except ValueError:
        args = argv[1:]
    ns = parse_args(args)
    show_image(ns.image)


if __name__ == "__main__":
    main(sys.argv)
