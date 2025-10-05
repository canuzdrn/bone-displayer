"""
Display the segmentation mask in 3D slicer

Run:
    /Applications/Slicer.app/Contents/MacOS/Slicer \
    --python-script /abs/path/view_seg.py \
    --mask /abs/path/segmentation_file.nii.gz

check parse_args() for optional arguments such as smoothing, threshold, etc.
"""


import sys 
import argparse
import numpy as np 
import slicer

def to_labelmap(label_or_scalar_node, threshold=0.5):
    """Ensure we have a LabelMapVolumeNode; if scalar, binarize > threshold."""
    if label_or_scalar_node.IsA("vtkMRMLLabelMapVolumeNode"):
        return label_or_scalar_node
    # Convert scalar → labelmap via threshold
    arr = slicer.util.arrayFromVolume(label_or_scalar_node)
    bin_arr = (arr > threshold).astype(np.uint8)
    lm = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode', label_or_scalar_node.GetName()+"_LM")
    lm.CopyOrientation(label_or_scalar_node)
    slicer.util.updateVolumeFromArray(lm, bin_arr)
    lm.SetSpacing(label_or_scalar_node.GetSpacing())
    lm.SetOrigin(label_or_scalar_node.GetOrigin())
    return lm

def volume_info(node, title="VOLUME INFO"):
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
    print("======================\n")

def mask_stats(label_node):
    arr_zyx = slicer.util.arrayFromVolume(label_node)
    fg = int(np.count_nonzero(arr_zyx))
    sx, sy, sz = label_node.GetSpacing()
    vol_ml = fg * (sx*sy*sz) / 1000.0
    print(f"Foreground voxels:   {fg:,}")
    print(f"Foreground volume:   {vol_ml:.2f} mL\n")

def show_segmentation(mask_path, background_path=None, color=(0.1,1.0,0.1),
                      smoothing=0.12, decimation=0.65, oversample=2.0, threshold=0.5):
    # Try to load as labelmap first
    label_node = slicer.util.loadLabelVolume(mask_path)
    if not label_node:
        scalar = slicer.util.loadVolume(mask_path)
        if not scalar:
            raise RuntimeError(f"Failed to load mask: {mask_path}")
        print("[info] Loaded mask as scalar volume; binarizing…")
        label_node = to_labelmap(scalar, threshold=threshold)

    volume_info(label_node, "SEGMENTATION LABELMAP")
    mask_stats(label_node)

    # Optional background CT
    bg_node = None
    if background_path:
        bg_node = slicer.util.loadVolume(background_path)
        if bg_node:
            volume_info(bg_node, "BACKGROUND CT")

    # Show in 2D (overlay if background provided)
    slicer.util.setSliceViewerLayers(background=bg_node or label_node,
                                     label=label_node if bg_node else None,
                                     fit=True)

    # LabelMap -> Segmentation
    seg_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLSegmentationNode', 'Seg')
    seg_node.CreateDefaultDisplayNodes()
    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(label_node)
    slicer.vtkSlicerSegmentationsModuleLogic.ImportLabelmapToSegmentationNode(label_node, seg_node)

    # Display settings
    disp = seg_node.GetDisplayNode()
    disp.SetVisibility2D(True)
    disp.SetVisibility3D(True)
    disp.SetOpacity3D(0.75)

    seg = seg_node.GetSegmentation()
    if seg.GetNumberOfSegments() > 0:
        first_id = seg.GetNthSegmentID(0)
        seg.GetSegment(first_id).SetColor(color)

    # Nicer surface
    seg.SetConversionParameter('Oversampling factor', str(float(oversample)))
    seg.SetConversionParameter('Smoothing factor',    str(float(smoothing)))
    seg.SetConversionParameter('Decimation factor',   str(float(decimation)))
    seg_node.CreateClosedSurfaceRepresentation()

    # Reset 3D camera
    lm = slicer.app.layoutManager()
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
    lm.threeDWidget(0).threeDView().resetCamera()

    print(f"Shown segmentation overlay + smoothed (s={smoothing}), decimated (d={decimation}), oversampled (x{oversample}) surface.")
    return label_node, seg_node, bg_node

def parse_args(argv):
    p = argparse.ArgumentParser(description="Display a segmentation NIfTI in Slicer (2D overlay + 3D surface).")
    p.add_argument("--mask", required=True, help="Path to segmentation .nii/.nii.gz")
    p.add_argument("--background", help="Optional background CT .nii/.nii.gz to overlay on")
    p.add_argument("--smooth", type=float, default=0.12, help="Smoothing factor (0..1)")
    p.add_argument("--decimate", type=float, default=0.65, help="Decimation factor (0..1, fraction removed)")
    p.add_argument("--oversample", type=float, default=2.0, help="Oversampling factor (>=1)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing scalar masks")
    return p.parse_args(argv)

def main(argv):
    args = parse_args(argv)
    show_segmentation(args.mask, background_path=args.background,
                      smoothing=args.smooth, decimation=args.decimate,
                      oversample=args.oversample, threshold=args.threshold)

if __name__ == "__main__":
    # In Slicer: args come after a bare "--"
    try:
        idx = sys.argv.index("--")
        argv = sys.argv[idx+1:]
    except ValueError:
        argv = sys.argv[1:]
    main(argv)
