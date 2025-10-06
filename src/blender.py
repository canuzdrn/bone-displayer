import tempfile
import subprocess
import slicer

"""
idea of this script is to display the mesh of the give segmentation using Blender. Assuming you are currently displaying a segmentation
in 3D Slicer, you can use Slicer's Python console to execute this file and you can view the mesh in new Blender window.

Run (in Slicer's Python console):
    1- exec(open("/abs/path/blender.py").read())
    2- export_current_seg_to_blender("$BLENDER_PATH$")
    
    default Path for Blender in macOS : /Applications/Blender.app/Contents/MacOS/Blender
"""


def _ensure_closed_surface(seg_node, oversample=2.0, smoothing=0.12, decimation=0.65):
    """
    configure segmentation to mesh conversion (oversampling, smoothing, etc.) and
    generate closed surface representation (triangular mesh ready for 3D viewing) of the segmentation
    """
    seg = seg_node.GetSegmentation()
    seg.SetConversionParameter("Oversampling factor", str(float(oversample)))
    seg.SetConversionParameter("Smoothing factor",    str(float(smoothing)))
    seg.SetConversionParameter("Decimation factor",   str(float(decimation)))
    seg_node.CreateClosedSurfaceRepresentation()

def _export_segments_to_temp_models(seg_node, out_format="stl"):
    """
    export all segments* from the given segmentation node to temporary mesh files. 
    We'll use those temp mesh files to display the mesh of the current segmentation.

    *: function is implemented in a way it exports meshes of different segmentations
    such as (tibia, fibula, cartilage,...). By default we have one segmentation for each display
    which is tibia. If necessary, we can modify the implementation where we export only the first
    segmentation.
    """
    out_format = out_format.lower()
    if out_format not in ("stl","obj","ply"):
        out_format = "stl"

    before = {n.GetID() for n in slicer.util.getNodesByClass("vtkMRMLModelNode")}
    slicer.modules.segmentations.logic().ExportAllSegmentsToModels(seg_node, 1)
    new_models = [
        n for n in slicer.util.getNodesByClass("vtkMRMLModelNode")
        if n.GetID() not in before
    ]
    if not new_models:
        raise RuntimeError("No model nodes exported.")

    paths = []
    for i, mn in enumerate(new_models, 1):
        tmp = tempfile.NamedTemporaryFile(prefix=f"segmesh_{i:02d}_", suffix=f".{out_format}", delete=False)
        tmp_path = tmp.name; tmp.close()
        if not slicer.util.saveNode(mn, tmp_path):
            raise RuntimeError(f"Failed to save: {tmp_path}")
        paths.append(tmp_path)
    return paths

def _write_blender_loader_script():
    """
    build a temporary Blender script and import
    the mesh using Blender
    """

    import textwrap, tempfile
    code = textwrap.dedent(r"""
        import sys, os, bpy

        mesh_paths = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []

        def import_one(p):
            ext = os.path.splitext(p)[1].lower()
            try:
                if ext == ".stl":
                    bpy.ops.wm.stl_import(filepath=p)
                elif ext == ".obj":
                    bpy.ops.wm.obj_import(filepath=p)
                elif ext == ".ply":
                    bpy.ops.wm.ply_import(filepath=p)
                else:
                    print(f"[Blender] Unsupported format: {ext} for {p}")
            except AttributeError:
                if ext == ".stl":
                    bpy.ops.import_mesh.stl(filepath=p)
                elif ext == ".obj":
                    bpy.ops.import_scene.obj(filepath=p)
                elif ext == ".ply":
                    bpy.ops.import_mesh.ply(filepath=p)

        for p in mesh_paths:
            import_one(p)

        for obj in bpy.context.selected_objects:
            if obj.type == "MESH":
                bpy.ops.object.shade_smooth()
                # obj.data.use_auto_smooth = True ------ became obsolete
                bpy.ops.object.shade_auto_smooth(angle=0.523599)

        for area in bpy.context.screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.shading.type = "MATERIAL"
                break

        print("[Blender] Imported", len(mesh_paths), "mesh(es).")
    """)
    f = tempfile.NamedTemporaryFile(prefix="blender_import_", suffix=".py", delete=False, mode="w", encoding="utf-8")
    f.write(code) 
    path = f.name
    f.close()
    return path

def export_current_seg_to_blender(blender_bin="blender",
                                  out_format="stl",
                                  oversample=2.0,
                                  smoothing=0.12,
                                  decimation=0.65,
                                  verbose=True):
    
    """
    find a segmentation already in the Slicer scene, turns it into a surface mesh, 
    saves the mesh as a temporary file and then launch Blender to import them
    """

    seg_nodes = slicer.util.getNodesByClass("vtkMRMLSegmentationNode")
    if not seg_nodes:
        if verbose: print("No segmentation in scene."); return []
    seg_node = seg_nodes[-1]
    if verbose: print("Using segmentation:", seg_node.GetName())
    _ensure_closed_surface(seg_node, oversample, smoothing, decimation)
    mesh_paths = _export_segments_to_temp_models(seg_node, out_format)
    if verbose:
        for p in mesh_paths: print("Mesh:", p)
    loader = _write_blender_loader_script()
    cmd = [blender_bin, "--python", loader, "--"] + mesh_paths
    if verbose: print("Launching Blender:", " ".join(cmd))
    try:
        subprocess.Popen(cmd)
    except Exception as e:
        print("Failed to launch Blender:", e)
    return mesh_paths
