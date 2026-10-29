"""
Module 2: SAMGeo Segmentation with Intelligent GPU (CUDA) Memory Management
LAPIG - Image Processing and Geoprocessing Laboratory
"""

import os
import gc
import time
from datetime import timedelta
import torch

def clear_gpu_memory(verbose=False):
    """
    Completely flushes video memory (VRAM) and PyTorch CUDA cache.
    1. Forces garbage collection of unreferenced Python objects (gc.collect).
    2. Empties cached blocks from CUDA caching allocator (torch.cuda.empty_cache).
    3. Collects shared IPC memory blocks (torch.cuda.ipc_collect).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        if verbose:
            try:
                vram_mb = torch.cuda.memory_allocated() / (1024**2)
                res_mb = torch.cuda.memory_reserved() / (1024**2)
                print(f"[VRAM Clean] GPU Memory released: Allocated={vram_mb:.1f} MB | Reserved={res_mb:.1f} MB")
            except Exception:
                pass

def get_gpu_info():
    """Returns available GPU diagnostic information."""
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU", "vram_gb": 0.0}
    device_id = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_id)
    vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
    vram_gb = vram_bytes / (1024**3)
    return {"available": True, "name": name, "vram_gb": vram_gb}

def run_samgeo_segmentation(image_path, output_mask_path, model_type='auto', checkpoint_dir='models', force_rerun=False):
    """
    Executes parcel segmentation using Segment Anything Model for Geospatial (SAMGeo).

    Automated hardware adaptation:
    - RTX 2080 Ti (11GB): Runs ViT-H or ViT-L with high headroom.
    - RTX 3060 Ti (8GB): Runs ViT-H with calibrated batch size or ViT-B.
    - Guaranteed zero-leak VRAM deallocation upon completion of each run.
    """
    # Ensure clean VRAM prior to any operation
    clear_gpu_memory()

    if os.path.exists(output_mask_path) and os.path.getsize(output_mask_path) > 1000 and not force_rerun:
        print(f"SAMGeo mask already exists: {output_mask_path}. Skipping inference.")
        clear_gpu_memory()
        return output_mask_path

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[-] Input image not found: {image_path}")

    gpu_info = get_gpu_info()
    print(f"\nHardware Diagnostic:")
    print(f"   Device:      {gpu_info['name']}")
    if gpu_info['available']:
        print(f"   Total VRAM:  {gpu_info['vram_gb']:.2f} GB")
    else:
        print("   Warning: CUDA not available. Running on CPU (inference may be slow).")

    # Automated architecture selection
    if model_type == 'auto':
        if gpu_info['vram_gb'] >= 10.0:
            chosen_model = 'vit_h'
        elif gpu_info['vram_gb'] >= 7.5:
            # RTX 3060 Ti (8GB): ViT-H with calibrated batch size
            chosen_model = 'vit_h'
        elif gpu_info['vram_gb'] >= 5.0:
            chosen_model = 'vit_l'
        else:
            chosen_model = 'vit_b'
    else:
        chosen_model = model_type

    print(f"Initializing SAMGeo with architecture: {chosen_model.upper()}...")

    # Calibrated parameters (LAPIG Setup) for subtle fence line detection
    # points_per_batch set to 64 to ensure safety margin against VRAM overflow on 8GB cards
    batch_size = 64 if gpu_info['vram_gb'] < 10.0 else 128

    sam_kwargs = {
        'points_per_side': 100,
        'points_per_batch': batch_size,
        'pred_iou_thresh': 0.70,
        'stability_score_thresh': 0.85,
        'stability_score_offset': 1.0,
        'crop_n_layers': 1,
        'crop_overlap_ratio': 512 / 1500,
        'crop_n_points_downscale_factor': 1,
        'min_mask_region_area': 3000,
        'box_nms_thresh': 0.5,
    }

    # Suppress OpenCV/libtiff warnings when reading GeoTIFF cartographic tags (33550, 33922, 34735, 34737)
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    try:
        import cv2
        cv2.setLogLevel(0)
    except Exception:
        pass

    from samgeo import SamGeo

    os.makedirs(os.path.dirname(os.path.abspath(output_mask_path)), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    t0 = time.time()
    sam = None
    try:
        sam = SamGeo(
            model_type=chosen_model,
            checkpoint_dir=checkpoint_dir,
            device='cuda' if gpu_info['available'] else 'cpu',
            sam_kwargs=sam_kwargs
        )

        print(f"Processing parcel segmentation on: {os.path.basename(image_path)}...")
        sam.generate(image_path, output=output_mask_path, foreground=True, unique=True)
        elapsed = str(timedelta(seconds=int(time.time() - t0)))
        print(f"Segmentation completed successfully in {elapsed}!")
        print(f"Raster mask saved to: {output_mask_path}")
    finally:
        # Explicit unbinding and reference deletion for complete VRAM deallocation
        if sam is not None:
            try:
                if hasattr(sam, 'mask_generator'):
                    del sam.mask_generator
                if hasattr(sam, 'predictor'):
                    del sam.predictor
                if hasattr(sam, 'sam'):
                    del sam.sam
            except Exception:
                pass
            del sam

        # Immediately flush VRAM so subsequent steps and batches start clean
        clear_gpu_memory(verbose=True)

    return output_mask_path
