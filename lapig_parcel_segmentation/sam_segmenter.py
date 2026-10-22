"""
Módulo 2: Segmentação SAMGeo com Gerenciamento Inteligente de GPU (CUDA)
LAPIG - Laboratório de Processamento de Imagens e Geoprocessamento
"""

import os
import gc
import time
from datetime import timedelta
import torch

def clear_gpu_memory(verbose=False):
    """
    Libera completamente a memória de vídeo (VRAM) e cache do PyTorch CUDA.
    1. Força a coleta de lixo de objetos Python desacoplados (gc.collect).
    2. Esvazia o cache alocado pelo caching allocator do CUDA (torch.cuda.empty_cache).
    3. Coleta blocos IPC compartilhados (torch.cuda.ipc_collect).
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
                print(f"🧹 [VRAM Clean] Memória GPU liberada: Alocada={vram_mb:.1f} MB | Reservada={res_mb:.1f} MB")
            except Exception:
                pass

def get_gpu_info():
    """Retorna informações da GPU disponível para o usuário."""
    if not torch.cuda.is_available():
        return {"available": False, "name": "CPU", "vram_gb": 0.0}
    device_id = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_id)
    vram_bytes = torch.cuda.get_device_properties(device_id).total_memory
    vram_gb = vram_bytes / (1024**3)
    return {"available": True, "name": name, "vram_gb": vram_gb}

def run_samgeo_segmentation(image_path, output_mask_path, model_type='auto', checkpoint_dir='models', force_rerun=False):
    """
    Executa a segmentação de piquetes com o Segment Anything Model for Geospatial (SAMGeo).
    
    Gerenciamento automático de hardware:
    - RTX 2080 Ti (11GB): Roda ViT-H ou ViT-L com folga.
    - RTX 3060 Ti (8GB): Roda ViT-H otimizado ou ViT-B.
    - Purga completa de VRAM garantida ao término de cada execução.
    """
    # Garante VRAM limpa antes de qualquer verificação
    clear_gpu_memory()

    if os.path.exists(output_mask_path) and os.path.getsize(output_mask_path) > 1000 and not force_rerun:
        print(f"ℹ️ Máscara do SAMGeo já existente: {output_mask_path}. Pulando inferência.")
        clear_gpu_memory()
        return output_mask_path

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Imagem de entrada não encontrada: {image_path}")

    gpu_info = get_gpu_info()
    print(f"\n🔍 Diagnóstico de Hardware:")
    print(f"   Dispositivo: {gpu_info['name']}")
    if gpu_info['available']:
        print(f"   VRAM Total:  {gpu_info['vram_gb']:.2f} GB")
    else:
        print("   ⚠️ CUDA não disponível. Rodando na CPU (pode ser lento).")

    # Seleção automática de modelo se solicitado
    if model_type == 'auto':
        if gpu_info['vram_gb'] >= 10.0:
            chosen_model = 'vit_h'
        elif gpu_info['vram_gb'] >= 7.5:
            # RTX 3060 Ti (8GB): Suporta ViT-H com batch calibrado
            chosen_model = 'vit_h'
        elif gpu_info['vram_gb'] >= 5.0:
            chosen_model = 'vit_l'
        else:
            chosen_model = 'vit_b'
    else:
        chosen_model = model_type

    print(f"🚀 Inicializando SAMGeo com arquitetura: {chosen_model.upper()}...")

    # Parâmetros calibrados pelo Vinícius (LAPIG Setup) para detecção de lances de cerca sutis
    # Ajustamos points_per_batch para 64 para garantir margem de segurança contra estouro de VRAM na 3060 Ti
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

    # Suprime avisos do OpenCV/libtiff ao ler tags cartográficas de GeoTIFF (33550, 33922, 34735, 34737)
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

        print(f"⏳ Processando segmentação em: {os.path.basename(image_path)}...")
        sam.generate(image_path, output=output_mask_path, foreground=True, unique=True)
        elapsed = str(timedelta(seconds=int(time.time() - t0)))
        print(f"✅ Segmentação concluída com sucesso em {elapsed}!")
        print(f"📁 Máscara raster salva em: {output_mask_path}")
    finally:
        # Desvinculação forçada e exclusão de referências para liberação de VRAM
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

        # Esvazia a VRAM imediatamente para que as etapas seguintes e próximas parcelas comecem limpas
        clear_gpu_memory(verbose=True)

    return output_mask_path
