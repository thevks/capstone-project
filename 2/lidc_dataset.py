# lidc_dataset.py
"""
PyTorch Dataset to read CT DICOM folders (LIDC-IDRI style) and return:
 - per-slice 2D images (grayscale, normalized after HU windowing)
 - OR 3D sub-volumes (n_slices x H x W) for 3D U-Net

Requirements:
 pip install pydicom numpy scipy torch torchvision SimpleITK
"""
import os
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pydicom
import SimpleITK as sitk
from torch.utils.data import Dataset

def read_dicom_volume(dirpath: str) -> Tuple[np.ndarray, dict]:
    """
    Read DICOM series in a folder into a 3D numpy volume (z,y,x) and metadata dict.
    Works if folder contains a single series (typical per-study folder).
    """
    reader = sitk.ImageSeriesReader()
    series_IDs = reader.GetGDCMSeriesIDs(str(dirpath))
    if not series_IDs:
        raise ValueError(f"No DICOM series found in {dirpath}")
    series_file_names = reader.GetGDCMSeriesFileNames(str(dirpath), series_IDs[0])
    reader.SetFileNames(series_file_names)
    image = reader.Execute()
    array = sitk.GetArrayFromImage(image)  # shape: (slices, rows, cols)
    meta = {
        "origin": image.GetOrigin(),
        "spacing": image.GetSpacing(),  # (x,y,z) in mm
        "direction": image.GetDirection()
    }
    # if pixel representation needs rescale, read from first slice via pydicom
    ds = pydicom.dcmread(series_file_names[0], stop_before_pixels=True)
    meta["RescaleIntercept"] = float(getattr(ds, "RescaleIntercept", 0.0))
    meta["RescaleSlope"] = float(getattr(ds, "RescaleSlope", 1.0))
    return array.astype(np.float32), meta

def apply_hu_window(volume: np.ndarray, rescale_slope=1.0, rescale_intercept=0.0,
                    win_center: float = -600, win_width: float = 1500) -> np.ndarray:
    """
    Convert raw pixel values to Hounsfield Units (HU), then apply window center/width,
    clip and normalize to [0,1] float32.
    Default window is wide (useful for whole-chest CT). Adjust for soft-tissue if needed.
    """
    hu = volume * rescale_slope + rescale_intercept
    low = win_center - win_width/2.0
    high = win_center + win_width/2.0
    hu_clipped = np.clip(hu, low, high)
    norm = (hu_clipped - low) / (high - low)
    return norm.astype(np.float32)

class LIDCSliceDataset(Dataset):
    """
    Per-slice dataset:
    - root_dir: each subfolder contains a DICOM series (one CT volume)
    - mode: '2d' returns single slices; '3d' returns small volumes (n_slices, H, W)
    - n_slices_3d: number of slices to return when mode='3d' (must be odd)
    - transform: optional callable(img) -> img
    """
    def __init__(self, root_dir: str, mode: str = '2d', n_slices_3d: int = 5,
                 window_center: float = -600, window_width: float = 1500,
                 transform=None):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise ValueError("root_dir does not exist")
        self.series_dirs = [p for p in self.root.iterdir() if p.is_dir()]
        self.mode = mode
        self.n_slices = n_slices_3d if mode == '3d' else 1
        if self.n_slices % 2 == 0:
            raise ValueError("n_slices_3d must be odd")
        self.window_center = window_center
        self.window_width = window_width
        self.transform = transform

        # Preload file lists to accelerate indexing
        self.volumes = []  # list of (volume_numpy, meta, series_path)
        for s in self.series_dirs:
            try:
                vol, meta = read_dicom_volume(str(s))
                self.volumes.append((vol, meta, s))
            except Exception as e:
                print(f"Skipping {s}: {e}")

        # Build an index mapping (series_idx, slice_idx) for easy per-slice access
        self.index = []
        for si, (vol, meta, sp) in enumerate(self.volumes):
            n = vol.shape[0]
            for z in range(n):
                # skip tiny volumes if needed
                self.index.append((si, z))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        si, z = self.index[idx]
        vol, meta, spath = self.volumes[si]
        slope = meta.get("RescaleSlope", 1.0)
        intercept = meta.get("RescaleIntercept", 0.0)
        # prepare slices
        if self.mode == '2d':
            slice_img = vol[z, :, :]
            img_norm = apply_hu_window(slice_img, slope, intercept,
                                       win_center=self.window_center, win_width=self.window_width)
            if self.transform:
                img_norm = self.transform(img_norm)
            return img_norm, {"series_dir": str(spath), "slice_index": z}
        else:
            half = self.n_slices // 2
            z0 = max(0, z - half)
            z1 = min(vol.shape[0]-1, z + half)
            # if near boundaries, pad by repeating edge slices
            slices = []
            for zz in range(z - half, z + half + 1):
                zz_clamped = min(max(zz, 0), vol.shape[0]-1)
                slices.append(vol[zz_clamped, :, :])
            vol3d = np.stack(slices, axis=0)
            vol3d = apply_hu_window(vol3d, slope, intercept,
                                    win_center=self.window_center, win_width=self.window_width)
            if self.transform:
                vol3d = self.transform(vol3d)
            return vol3d, {"series_dir": str(spath), "center_slice": z}


'''
Notes:

This dataset reads DICOM series using SimpleITK, converts to HU using RescaleSlope/Intercept, applies windowing, and normalizes to [0,1].

Use mode='2d' (per-slice) to reuse the earlier 2D pipeline (LBP + 2D U-Net). Use mode='3d' to feed small 3D blocks to the 3D U-Net.

If you have PNG/JPEG slice images instead, write a thin wrapper to load them and skip the HU conversion.

'''
