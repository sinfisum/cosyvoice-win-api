"""
Model Manager for CosyVoice3
Handles model downloading, caching, and loading
"""

import os
import torch
from typing import Optional, Dict, Any
from tqdm import tqdm

# Global model cache
_MODEL_CACHE = {}

# Model configurations
MODEL_CONFIGS = {
    "Fun-CosyVoice3-0.5B": {
        "modelscope_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "huggingface_id": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
        "recommended": True,
    },
    "CosyVoice2-0.5B": {
        "modelscope_id": "FunAudioLLM/CosyVoice2-0.5B",
        "huggingface_id": "FunAudioLLM/CosyVoice2-0.5B",
        "recommended": False,
    },
    "CosyVoice-300M": {
        "modelscope_id": "FunAudioLLM/CosyVoice-300M",
        "huggingface_id": "FunAudioLLM/CosyVoice-300M",
        "recommended": False,
    },
}


def get_models_directory() -> str:
    """Get the base models directory for CosyVoice models"""
    # Directly use pretrained_models directory
    base_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models"))
    cosyvoice_dir = os.path.join(base_models_dir, "cosyvoice")
    os.makedirs(cosyvoice_dir, exist_ok=True)
    print(f"[UTILS] Models directory: {cosyvoice_dir}")
    return cosyvoice_dir


def check_model_exists(model_dir: str) -> bool:
    """Check if model files exist in the directory"""
    if not os.path.exists(model_dir):
        return False
    # Check for any CosyVoice config file (v1, v2, or v3)
    config_files = ['cosyvoice.yaml', 'cosyvoice2.yaml', 'cosyvoice3.yaml']
    # Note: Some models use llm.rl.pt instead of llm.pt (e.g., Fun-CosyVoice3 on HuggingFace)
    llm_files = ['llm.pt', 'llm.rl.pt']
    flow_file = 'flow.pt'
    # Recursively search for files
    for root, dirs, files in os.walk(model_dir):
        has_config = any(f in files for f in config_files)
        has_llm = any(f in files for f in llm_files)
        has_flow = flow_file in files
        if has_config and has_llm and has_flow:
            return True
    return False


def get_model_path(model_version: str, download_source: str = "ModelScope", force_redownload: bool = False) -> str:
    """Get the path to a CosyVoice model from local directory"""
    if model_version not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model version: {model_version}. Available: {list(MODEL_CONFIGS.keys())}")
    # Use pretrained_models directory directly
    base_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pretrained_models"))
    model_dir = os.path.join(base_models_dir, model_version)
    # Check if model exists
    if not force_redownload and check_model_exists(model_dir):
        print(f"\n{'='*60}")
        print(f"[UTILS] Model found locally: {model_dir}")
        print(f"{'='*60}\n")
        return model_dir
    raise FileNotFoundError(f"Model {model_version} not found in {base_models_dir}. Please ensure the model is downloaded to the pretrained_models directory.")


def load_cosyvoice_model(model_path: str, device: Optional[torch.device] = None) -> Any:
    """Load a CosyVoice model from disk"""
    print(f"\n{'='*60}")
    print(f"[UTILS] Loading CosyVoice model from: {model_path}")
    print(f"{'='*60}\n")
    try:
        # Import from vendored cosyvoice package (bundled with this node pack)
        import sys
        vendored_path = os.path.join(os.path.dirname(__file__), "..")
        if vendored_path not in sys.path:
            sys.path.insert(0, vendored_path)
        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2, CosyVoice3, AutoModel
        # Determine device if device is None
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[UTILS] Target device: {device}")
        # Search for model directory containing cosyvoice.yaml or cosyvoice3.yaml
        # Also handles HuggingFace cache structure with snapshots/ subdirectory
        model_dir = None
        config_names = ['cosyvoice.yaml', 'cosyvoice3.yaml', 'cosyvoice2.yaml']
        for root, dirs, files in os.walk(model_path):
            for config_name in config_names:
                if config_name in files:
                    model_dir = root
                    print(f"[UTILS] Found config: {os.path.join(root, config_name)}")
                    break
            if model_dir:
                break
        if model_dir is None:
            raise FileNotFoundError(f"Could not find cosyvoice.yaml/cosyvoice3.yaml in {model_path}")
        # Use AutoModel to automatically detect and load the correct model type
        print(f"[UTILS] Using AutoModel to load from: {model_dir}")
        model = AutoModel(model_dir=model_dir, load_trt=False, fp16=False)
        # Note: CosyVoice handles device placement internally, no need to call .to()
        print(f"\n{'='*60}")
        print(f"[UTILS] Model loaded successfully!")
        print(f"{'='*60}\n")
        return model
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"[UTILS] ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise


def get_cached_model(model_version: str, download_source: str = "ModelScope", device: Optional[torch.device] = None, force_redownload: bool = False, force_reload: bool = False) -> Dict[str, Any]:
    """Get a CosyVoice model, using cache if available"""
    cache_key = f"{model_version}_{device}"
    # Check cache
    if not force_reload and cache_key in _MODEL_CACHE:
        print(f"\n{'='*60}")
        print(f"[UTILS] Using cached model: {model_version}")
        print(f"{'='*60}\n")
        return _MODEL_CACHE[cache_key]
    # Get model path (download if needed)
    model_path = get_model_path(model_version, download_source, force_redownload)
    # Load model
    model = load_cosyvoice_model(model_path, device)
    # Detect model version to use
    version_lower = model_version.lower()
    is_cosyvoice3 = "cosyvoice3" in version_lower or "fun-cosyvoice3" in version_lower
    is_cosyvoice2 = "cosyvoice2" in version_lower and not is_cosyvoice3
    # Create model info dict
    model_info = {
        "model": model,
        "model_name": model_version,
        "model_version": model_version,
        "model_path": model_path,
        "device": device,
        "sample_rate": model.sample_rate,
        # Use actual model sample rate (24000 for v2/v3, 22050 for v1)
        "is_cosyvoice3": is_cosyvoice3,
        "is_cosyvoice2": is_cosyvoice2,
    }
    print(f"[UTILS] Model sample rate: {model.sample_rate} Hz")
    print(f"[UTILS] Model type: {'CosyVoice3' if is_cosyvoice3 else 'CosyVoice2' if is_cosyvoice2 else 'CosyVoice v1'}")
    # Cache model
    _MODEL_CACHE[cache_key] = model_info
    return model_info


def clear_model_cache():
    """Clear the model cache"""
    global _MODEL_CACHE
    _MODEL_CACHE.clear()
    print(f"\n{'='*60}")
    print(f"[UTILS] Model cache cleared")
    print(f"{'='*60}\n")