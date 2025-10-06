# bone-displayer

## Tibia CT & Segmentation — Quick Viewer

This project lets you **preview tibia CT volumes and segmentation masks** stored in an **HDF5 (`.h5`)** file using **3D Slicer**. It can:

- Show **CT** with smart window/level and **3D volume rendering**
- Show **segmentations** as 2D overlays + **smoothed 3D surface**
- **Randomly pick** one dataset from an `.h5`, convert to **temporary** NIfTI, open it in Slicer, and **clean up** (no files left behind)

---

## Repository

- `view_img.py` — View a **CT NIfTI** in Slicer (auto window/level + 3D VR)
- `view_seg.py` — View a **segmentation NIfTI** in Slicer (2D label overlay + 3D surface)
- `pick_and_display.py` — **Main entry**: open `.h5`, skip metadata, randomly pick **CT or mask**, write **temp** `.nii.gz`, launch Slicer with the right viewer, then delete temps
- `h5_convert.py` — (Optional) Convert datasets from `.h5` → **NIfTI** on disk
- `blender.py` — Given segmentation image, it saves the mesh of the surface as a temp file and display the mesh in Blender.

---

## Installation

### 1) Install 3D Slicer (external app)
Download and install from <https://www.slicer.org/>.

Typical executable paths:
- **macOS:** `/Applications/Slicer.app/Contents/MacOS/Slicer`
- **Windows:** `C:\Program Files\Slicer <version>\Slicer.exe`
- **Linux:** `Slicer` (on PATH) or `/path/to/Slicer`

### 2) Set up Python environment & install deps
Use system Python or a virtual environment:

```bash
# create and activate a venv (recommended)
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate

# upgrade pip and install all Python deps
pip install --upgrade pip
pip install -r requirements.txt
