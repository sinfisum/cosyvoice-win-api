#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice3 Gradio Web Interface
==============================
Полный веб-интерфейс со всеми возможностями CosyVoice3
Для запуска используйте .\start_web.bat
"""

import os
import sys
import re
import json
import tempfile
import random
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import soundfile as sf  # type: ignore[import-not-found]
import torch
import torchaudio
import whisper  # type: ignore[import-not-found]

# ── Ensure vendored cosyvoice / utils / nodes are importable ──────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print(f"[SCRIPT_DIR]: {SCRIPT_DIR}")
print(f"[os.path.abspath(__file__)]: {os.path.abspath(__file__)}")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── CosyVoice imports ─────────────────────────────────────────────────────
from utils.model_manager import (
    MODEL_CONFIGS,
    get_cached_model,
    clear_model_cache,
)
from utils.audio_utils import (
    save_raw_audio_to_tempfile,
    cleanup_temp_file,
    tensor_to_comfyui_audio,
    load_audio_from_path,
)

# Whisper (lazy-loaded)
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        try:
            import whisper
            print("[GradioApp] Loading Whisper model for auto-transcription...")
            _whisper_model = whisper.load_model("base")
            print("[GradioApp] Whisper model loaded.")
        except Exception as e:
            print(f"[GradioApp] Failed to load Whisper: {e}")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper_model()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_path, language=None)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[GradioApp] Transcription error: {e}")
        return ""


# ── Globals ───────────────────────────────────────────────────────────────
_current_model_info: Optional[Dict[str, Any]] = None


def is_v3_model(model_info: Dict[str, Any]) -> bool:
    version = model_info.get("model_version", "").lower()
    return "cosyvoice3" in version or "fun-cosyvoice3" in version


# 
#  Helpers
# 

def _np_to_wav_path(arr: np.ndarray, sr: int) -> str:
    """Save a numpy audio array to a temp WAV file; returns path."""
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        # channels-first → channels-last for soundfile
        arr = arr.T
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    sf.write(tmp.name, arr, sr)
    return tmp.name


def _audio_dict_to_tuple(audio_dict: Dict[str, Any]) -> Tuple[int, np.ndarray]:
    """Convert ComfyUI-style audio dict → (sample_rate, np.ndarray channels-last)."""
    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu()
    if hasattr(waveform, "numpy"):
        waveform = waveform.numpy()
    # Expected shape: [batch, channels, samples]
    if waveform.ndim == 3:
        waveform = waveform[0]  # drop batch
    if waveform.ndim == 2 and waveform.shape[0] <= 2:
        waveform = waveform.T   # channels-first → channels-last
    return sr, waveform


def _collect_output(generator) -> torch.Tensor:
    """Drain a CosyVoice generator and return concatenated waveform."""
    chunks = []
    for out in generator:
        speech = out["tts_speech"]
        chunks.append(speech)
    if not chunks:
        raise RuntimeError("Model produced no audio chunks.")
    waveform = torch.cat(chunks, dim=-1)
    return waveform


# 
#  Model management
# 

def load_model(model_version: str, source: str, device: str,
               force_redownload: bool, force_reload: bool) -> str:
    global _current_model_info
    try:
        if device == "auto":
            target_device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device(device)
    except Exception:
        target_device = torch.device("cpu")

    try:
        _current_model_info = get_cached_model(
            model_version=model_version,
            download_source=source,
            device=target_device,
            force_redownload=force_redownload,
            force_reload=force_reload,
        )
        msg = (f"✅ Model loaded: {_current_model_info['model_name']}\n"
               f"Device: {_current_model_info['device']}\n"
               f"Sample rate: {_current_model_info['sample_rate']} Hz\n"
               f"Path: {_current_model_info['model_path']}")
        print(msg)
        return msg
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error loading model:\n{type(e).__name__}: {e}"


def unload_model() -> str:
    global _current_model_info
    _current_model_info = None
    clear_model_cache()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "🗑️ Model unloaded and cache cleared."


# 
#  Audio preprocessing (crop, etc.)
# 

def parse_time_string(time_str: str) -> float:
    time_str = time_str.strip()
    if not time_str:
        raise ValueError("Time string cannot be empty")
    parts = time_str.split(':')
    try:
        if len(parts) == 2:  # MM:SS
            minutes = int(parts[0])
            seconds = float(parts[1])
            if minutes < 0 or seconds < 0 or seconds >= 60:
                raise ValueError
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            if hours < 0 or minutes < 0 or minutes >= 60 or seconds < 0 or seconds >= 60:
                raise ValueError
            return hours * 3600 + minutes * 60 + seconds
        else:
            raise ValueError
    except Exception:
        raise ValueError(f"Invalid time format: '{time_str}'. Use MM:SS or HH:MM:SS")


def crop_audio(audio_tuple: Optional[Tuple[int, np.ndarray]],
               start_time: str, end_time: str) -> Optional[Tuple[int, np.ndarray]]:
    """
    Gradio-style audio crop.
    Input:  (sample_rate, np_array) from gr.Audio
    Output: (sample_rate, np_array)
    """
    if audio_tuple is None:
        return None
    sr, data = audio_tuple
    if data is None or data.size == 0:
        return None

    start_s = parse_time_string(start_time)
    end_s = parse_time_string(end_time)

    # Ensure channels-last numpy
    if data.ndim == 1:
        data = data[:, np.newaxis]

    total_samples = data.shape[0]
    start_frame = int(start_s * sr)
    end_frame = int(end_s * sr)
    start_frame = max(0, min(start_frame, total_samples - 1))
    end_frame = max(start_frame + 1, min(end_frame, total_samples))

    cropped = data[start_frame:end_frame]
    return sr, cropped


# 
#  Core inference wrappers (return Gradio-compatible audio tuples)
# 

def _ensure_model_loaded() -> Dict[str, Any]:
    if _current_model_info is None:
        raise RuntimeError("No model loaded. Please load a model first in the 'Model' tab.")
    return _current_model_info


def _seed_all(seed: int):
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _run_inference_and_return_audio(generator, model_info) -> Tuple[int, np.ndarray]:
    """Drain generator, convert to Gradio audio format."""
    waveform = _collect_output(generator)
    if waveform.device != torch.device("cpu"):
        waveform = waveform.cpu()
    audio_dict = tensor_to_comfyui_audio(waveform, model_info["sample_rate"])
    sr, arr = _audio_dict_to_tuple(audio_dict)
    return sr, arr


# ── Zero-Shot ─────────────────────────────────────────────────────────────
def zero_shot(
    text: str,
    ref_audio: Optional[Tuple[int, np.ndarray]],
    speed: float,
    seed: int,
    text_frontend: bool,
    auto_transcribe: bool,
    ref_text: str,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    model_info = _ensure_model_loaded()
    if ref_audio is None:
        return None, "❌ Reference audio is required."

    sr, data = ref_audio
    # Build temp dict and file
    if data.ndim == 1:
        data = data[:, np.newaxis]
    audio_dict = {"waveform": torch.from_numpy(data.T).unsqueeze(0), "sample_rate": sr}

    # Check duration
    duration = data.shape[0] / sr
    if duration > 30:
        return None, (f"❌ Reference audio too long ({duration:.1f}s). "
                      "Max 30s. Use the Audio Crop tool first.")

    tmp_path = save_raw_audio_to_tempfile(audio_dict)
    try:
        _seed_all(seed)
        model = model_info["model"]
        is_v3 = is_v3_model(model_info)

        # Resolve prompt text
        if auto_transcribe or not ref_text.strip():
            prompt_text = transcribe_audio(tmp_path)
            if not prompt_text:
                # Fallback cross-lingual
                if is_v3:
                    tts_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                else:
                    tts_text = text
                gen = model.inference_cross_lingual(
                    tts_text=tts_text,
                    prompt_wav=tmp_path,
                    stream=False,
                    speed=speed,
                    text_frontend=text_frontend,
                )
                audio_out = _run_inference_and_return_audio(gen, model_info)
                return audio_out, f"✅ Zero-shot (cross-lingual fallback, no transcript). Duration: {audio_out[1].shape[0]/audio_out[0]:.2f}s"
        else:
            prompt_text = ref_text.strip()

        if is_v3:
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"

        gen = model.inference_zero_shot(
            tts_text=text,
            prompt_text=prompt_text,
            prompt_wav=tmp_path,
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )
        audio_out = _run_inference_and_return_audio(gen, model_info)
        return audio_out, f"✅ Zero-shot synthesis complete. Duration: {audio_out[1].shape[0]/audio_out[0]:.2f}s"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {e}"
    finally:
        cleanup_temp_file(tmp_path)


# ── Cross-Lingual ─────────────────────────────────────────────────────────
def cross_lingual(
    text: str,
    ref_audio: Optional[Tuple[int, np.ndarray]],
    speed: float,
    seed: int,
    text_frontend: bool,
    target_language: str = "auto"
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    model_info = _ensure_model_loaded()
    if ref_audio is None:
        return None, "❌ Reference audio is required."

    sr, data = ref_audio
    if data.ndim == 1:
        data = data[:, np.newaxis]
    audio_dict = {"waveform": torch.from_numpy(data.T).unsqueeze(0), "sample_rate": sr}

    duration = data.shape[0] / sr
    if duration > 30:
        return None, f"❌ Reference audio too long ({duration:.1f}s). Max 30s."

    tmp_path = save_raw_audio_to_tempfile(audio_dict)
    try:
        _seed_all(seed)
        model = model_info["model"]
        is_v3 = is_v3_model(model_info)

        if is_v3:
            tts_text = f"You are a helpful assistant.<|endofprompt|>{text}"
        else:
            lang_tags = {
                "en": "<|en|>", "zh": "<|zh|>", "ja": "<|jp|>",
                "ko": "<|ko|>", "yue": "<|yue|>", "de": "<|de|>",
                "es": "<|es|>", "fr": "<|fr|>", "it": "<|it|>", "ru": "<|ru|>"
            }
            lang_tag = lang_tags.get(target_language, "")
            tts_text = f"{lang_tag}{text}"

        gen = model.inference_cross_lingual(
            tts_text=tts_text,
            prompt_wav=tmp_path,
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )
        audio_out = _run_inference_and_return_audio(gen, model_info)
        return audio_out, f"✅ Cross-lingual synthesis complete. Duration: {audio_out[1].shape[0]/audio_out[0]:.2f}s"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {e}"
    finally:
        cleanup_temp_file(tmp_path)


# ── Instruct2 ─────────────────────────────────────────────────────────────
def instruct2(
    text: str,
    instruct_text: str,
    ref_audio: Optional[Tuple[int, np.ndarray]],
    speed: float,
    seed: int,
    text_frontend: bool,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    model_info = _ensure_model_loaded()
    if not instruct_text or not instruct_text.strip():
        return None, "❌ instruct_text cannot be empty."
    if ref_audio is None:
        return None, "❌ Reference audio is required."

    sr, data = ref_audio
    if data.ndim == 1:
        data = data[:, np.newaxis]
    audio_dict = {"waveform": torch.from_numpy(data.T).unsqueeze(0), "sample_rate": sr}

    duration = data.shape[0] / sr
    if duration > 30:
        return None, f"❌ Reference audio too long ({duration:.1f}s). Max 30s."

    tmp_path = save_raw_audio_to_tempfile(audio_dict)
    try:
        _seed_all(seed)
        model = model_info["model"]
        if not hasattr(model, "inference_instruct2"):
            return None, "❌ inference_instruct2 is not available. Use CosyVoice2 or CosyVoice3."

        is_v3 = is_v3_model(model_info)
        SYSTEM_PROMPT = "You are a helpful assistant."
        ENDOFPROMPT = "<|endofprompt|>"
        raw = instruct_text.strip()
        if raw.startswith(SYSTEM_PROMPT):
            raw = raw[len(SYSTEM_PROMPT):].lstrip("\n")
        if raw.endswith(ENDOFPROMPT):
            raw = raw[:-len(ENDOFPROMPT)].rstrip()

        if is_v3:
            formatted = SYSTEM_PROMPT + "\n" + raw + ENDOFPROMPT
        else:
            formatted = raw + ENDOFPROMPT

        gen = model.inference_instruct2(
            tts_text=text,
            instruct_text=formatted,
            prompt_wav=tmp_path,
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )
        audio_out = _run_inference_and_return_audio(gen, model_info)
        return audio_out, f"✅ Instruct2 synthesis complete. Duration: {audio_out[1].shape[0]/audio_out[0]:.2f}s"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {e}"
    finally:
        cleanup_temp_file(tmp_path)


# ── Voice Conversion ──────────────────────────────────────────────────────
def voice_conversion(
    source_audio: Optional[Tuple[int, np.ndarray]],
    target_audio: Optional[Tuple[int, np.ndarray]],
    speed: float,
    seed: int,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    model_info = _ensure_model_loaded()
    if source_audio is None or target_audio is None:
        return None, "❌ Both source and target audio are required."

    sr_s, data_s = source_audio
    sr_t, data_t = target_audio
    for label, data, sr in [("source", data_s, sr_s), ("target", data_t, sr_t)]:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        dur = data.shape[0] / sr
        if dur > 30:
            return None, f"❌ {label} audio too long ({dur:.1f}s). Max 30s."

    src_dict = {"waveform": torch.from_numpy(data_s.T).unsqueeze(0), "sample_rate": sr_s}
    tgt_dict = {"waveform": torch.from_numpy(data_t.T).unsqueeze(0), "sample_rate": sr_t}

    src_tmp = save_raw_audio_to_tempfile(src_dict)
    tgt_tmp = save_raw_audio_to_tempfile(tgt_dict)
    try:
        _seed_all(seed)
        model = model_info["model"]
        if not hasattr(model, "inference_vc"):
            return None, "❌ inference_vc is not available on this model."

        gen = model.inference_vc(
            source_wav=src_tmp,
            prompt_wav=tgt_tmp,
            stream=False,
            speed=speed,
        )
        audio_out = _run_inference_and_return_audio(gen, model_info)
        return audio_out, f"✅ Voice conversion complete. Duration: {audio_out[1].shape[0]/audio_out[0]:.2f}s"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {e}"
    finally:
        cleanup_temp_file(src_tmp)
        cleanup_temp_file(tgt_tmp)


# ── Speaker management (save / list / clone) ──────────────────────────────
# "./pretrained_models/Fun-CosyVoice3-0.5B"
def _get_speaker_dir() -> str:
    models_dir = "./pretrained_models/cosyvoice/speaker"
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _list_speaker_presets() -> List[str]:
    speaker_dir = _get_speaker_dir()
    if not os.path.isdir(speaker_dir):
        return []
    return sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(speaker_dir)
        if f.endswith(".pt")
    ])


def save_speaker(
    ref_audio: Optional[Tuple[int, np.ndarray]],
    ref_text: str,
    speaker_name: str,
) -> str:
    model_info = _ensure_model_loaded()
    if ref_audio is None:
        return "❌ Reference audio is required."
    speaker_name = speaker_name.strip()
    if not speaker_name:
        return "❌ Speaker name cannot be empty."

    sr, data = ref_audio
    if data.ndim == 1:
        data = data[:, np.newaxis]
    audio_dict = {"waveform": torch.from_numpy(data.T).unsqueeze(0), "sample_rate": sr}

    duration = data.shape[0] / sr
    if duration > 30:
        return f"❌ Reference audio too long ({duration:.1f}s). Max 30s."

    tmp_path = save_raw_audio_to_tempfile(audio_dict)
    try:
        model = model_info["model"]
        # Auto-transcribe if empty
        if not ref_text.strip():
            ref_text = transcribe_audio(tmp_path)
            if ref_text:
                print(f"[GradioApp] Auto-transcribed: {ref_text[:80]}{'...' if len(ref_text)>80 else ''}")

        is_v3 = is_v3_model(model_info)
        if is_v3:
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{ref_text}"
        else:
            prompt_text = ref_text

        model_input = model.frontend.frontend_zero_shot(
            "", prompt_text, tmp_path, model.sample_rate, ""
        )
        del model_input["text"]
        del model_input["text_len"]
        spk2info = {speaker_name: model_input}

        speaker_dir = _get_speaker_dir()
        save_path = os.path.join(speaker_dir, f"{speaker_name}.pt")
        torch.save(spk2info, save_path)
        return f"✅ Speaker saved: {save_path}\nKeys: {list(spk2info[speaker_name].keys())}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error saving speaker: {e}"
    finally:
        cleanup_temp_file(tmp_path)


def speaker_clone(
    text: str,
    speaker_preset: str,
    speed: float,
    seed: int,
    text_frontend: bool,
) -> Tuple[Optional[Tuple[int, np.ndarray]], str]:
    model_info = _ensure_model_loaded()
    if not speaker_preset:
        return None, "❌ No speaker preset selected."

    speaker_dir = _get_speaker_dir()
    pt_path = os.path.join(speaker_dir, f"{speaker_preset}.pt")
    if not os.path.isfile(pt_path):
        return None, f"❌ Preset not found: {pt_path}"

    try:
        _seed_all(seed)
        model = model_info["model"]
        load_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        spk2info = torch.load(pt_path, map_location=load_device)
        spk_id = next(iter(spk2info))
        model.frontend.spk2info = spk2info

        gen = model.inference_zero_shot(
            tts_text=text,
            prompt_text="",
            prompt_wav=None,
            zero_shot_spk_id=spk_id,
            stream=False,
            speed=speed,
            text_frontend=text_frontend,
        )
        audio_out = _run_inference_and_return_audio(gen, model_info)
        return audio_out, f"✅ Speaker clone complete. Duration: {audio_out[1].shape[0]/audio_out[0]:.2f}s"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {e}"


# ── Multi-Speaker Dialog Generation ──────────────────────────────────────
def generate_dialog(
    dialog_text: str,
    speaker_a_audio: Optional[Tuple[int, np.ndarray]],
    speaker_b_audio: Optional[Tuple[int, np.ndarray]],
    speaker_c_audio: Optional[Tuple[int, np.ndarray]],
    speaker_d_audio: Optional[Tuple[int, np.ndarray]],
    speed: float = 1.0,
    seed: int = -1
) -> Tuple[
    Optional[Tuple[int, np.ndarray]],
    Optional[Tuple[int, np.ndarray]],
    Optional[Tuple[int, np.ndarray]],
    Optional[Tuple[int, np.ndarray]],
    Optional[Tuple[int, np.ndarray]],
    str
]:
    model_info = _ensure_model_loaded()
    
    speakers = {
        "A": speaker_a_audio,
        "B": speaker_b_audio,
        "C": speaker_c_audio,
        "D": speaker_d_audio,
    }
    
    # Проверяем что хотя бы А и Б есть
    if speaker_a_audio is None or speaker_b_audio is None:
        return None, None, None, None, None, "❌ Требуется минимум 2 голоса (SPEAKER A и SPEAKER B)"
    
    lines = dialog_text.strip().splitlines()
    valid_lines = []
    
    # Парсим строки диалога
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for sid in ["SPEAKER A:", "SPEAKER B:", "SPEAKER C:", "SPEAKER D:"]:
            if line.upper().startswith(sid):
                speaker_id = sid.split()[1].replace(":", "")
                content = line[len(sid):].strip()
                valid_lines.append((speaker_id, content))
                break
    
    if not valid_lines:
        return None, None, None, None, None, "❌ Нет корректных строк диалога. Используйте формат SPEAKER X: Текст"
    
    try:
        _seed_all(seed)
        model = model_info["model"]
        sample_rate = model.sample_rate
        is_v3 = is_v3_model(model_info)
        
        temp_files = {}
        speaker_waveforms = {"A": [], "B": [], "C": [], "D": []}
        all_waveforms = []
        
        # Подготавливаем все голоса во временные файлы
        for speaker_id, audio in speakers.items():
            if audio is None:
                continue
            sr, data = audio
            if data.ndim == 1:
                data = data[:, np.newaxis]
            audio_dict = {"waveform": torch.from_numpy(data.T).unsqueeze(0), "sample_rate": sr}
            tmp_path = save_raw_audio_to_tempfile(audio_dict)
            temp_files[speaker_id] = tmp_path
        
        try:
            # Генерируем каждую реплику
            for idx, (speaker_id, text) in enumerate(valid_lines):
                print(f"[Dialog] Генерация реплики {idx+1}/{len(valid_lines)}: SPEAKER {speaker_id}")
                
                if speaker_id not in temp_files:
                    print(f"[Dialog] Пропуск реплики для SPEAKER {speaker_id}: нет голоса")
                    continue
                
                tmp_path = temp_files[speaker_id]
                
                if is_v3:
                    tts_text = f"You are a helpful assistant.<|endofprompt|>{text}"
                else:
                    tts_text = text
                
                gen = model.inference_cross_lingual(
                    tts_text=tts_text,
                    prompt_wav=tmp_path,
                    stream=False,
                    speed=speed,
                    text_frontend=True,
                )
                
                waveform = _collect_output(gen)
                if waveform.device != torch.device('cpu'):
                    waveform = waveform.cpu()
                
                speaker_waveforms[speaker_id].append(waveform)
                all_waveforms.append(waveform)
            
            # Собираем все треки
            def concat_or_empty(wf_list):
                if not wf_list:
                    return np.zeros((1, 1), dtype=np.float32)
                return torch.cat(wf_list, dim=-1).numpy()
            
            combined = torch.cat(all_waveforms, dim=-1) if all_waveforms else torch.zeros(1, 1)
            
            # Конвертируем в формат Gradio
            sr = sample_rate
            return (
                (sr, combined.numpy().T),
                (sr, concat_or_empty(speaker_waveforms["A"]).T),
                (sr, concat_or_empty(speaker_waveforms["B"]).T),
                (sr, concat_or_empty(speaker_waveforms["C"]).T),
                (sr, concat_or_empty(speaker_waveforms["D"]).T),
                f"✅ Диалог сгенерирован успешно! Реплик: {len(valid_lines)}, Длительность: {combined.shape[-1]/sample_rate:.2f} сек"
            )
            
        finally:
            # Чистим временные файлы
            for tmp_path in temp_files.values():
                cleanup_temp_file(tmp_path)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, f"❌ Ошибка генерации диалога: {str(e)}"


# 
#  Gradio UI
# 

def build_ui() -> Any:
    import gradio as gr  # type: ignore[import-not-found]

    css = """
    .gradio-container { max-width: auto; margin: 20px; }
    h1 { text-align: center; color: #4a90d9; }
    .tab-label { font-weight: 600; }
    """

    with gr.Blocks(title="CosyVoice3 Web UI", css=css) as demo:
        gr.Markdown("""
        # 🔊 CosyVoice3 Gradio Interface
        **Zero-shot voice cloning, cross-lingual synthesis, instruct control & voice conversion**
        """)

        # ── Model Tab ─────────────────────────────────────────────────────
        with gr.Tab("🧠 Model"):
            gr.Markdown("### Load or unload the CosyVoice model")
            with gr.Row():
                model_version = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="Fun-CosyVoice3-0.5B",
                    label="Model Version",
                )
                download_source = gr.Dropdown(
                    choices=["ModelScope", "HuggingFace"],
                    value="ModelScope",
                    label="Download Source",
                )
                device_choice = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    label="Device",
                )
            with gr.Row():
                force_redownload = gr.Checkbox(label="Force Re-download", value=False)
                force_reload = gr.Checkbox(label="Force Reload", value=False)
            with gr.Row():
                load_btn = gr.Button("📥 Load Model", variant="primary")
                unload_btn = gr.Button("🗑️ Unload Model", variant="secondary")
            model_status = gr.Textbox(label="Status", lines=6, interactive=False)

            load_btn.click(
                fn=lambda mv, ds, dev, frd, frl: load_model(mv, ds, dev, frd, frl),
                inputs=[model_version, download_source, device_choice, force_redownload, force_reload],
                outputs=model_status,
            )
            unload_btn.click(fn=unload_model, outputs=model_status)

        # ── Audio Tools ───────────────────────────────────────────────────
        with gr.Tab("✂️ Audio Crop"):
            gr.Markdown("Trim audio to a specific time range before using it as reference.")
            audio_in = gr.Audio(label="Input Audio", type="numpy")
            with gr.Row():
                start_time = gr.Textbox(label="Start Time", value="0:00", placeholder="MM:SS")
                end_time = gr.Textbox(label="End Time", value="0:10", placeholder="MM:SS")
            crop_btn = gr.Button("✂️ Crop Audio", variant="primary")
            audio_out_crop = gr.Audio(label="Cropped Audio", type="numpy")
            crop_btn.click(fn=crop_audio, inputs=[audio_in, start_time, end_time],
                          outputs=audio_out_crop)

        # ── Zero-Shot ─────────────────────────────────────────────────────
        with gr.Tab("🎙️ Zero-Shot Clone"):
            gr.Markdown("Clone a voice from a short reference audio clip.")
            zs_text = gr.Textbox(label="Text to synthesize", value="Hello, this is my cloned voice speaking.", lines=3)
            zs_ref_audio = gr.Audio(label="Reference Audio (3–10s recommended, max 30s)", type="numpy")
            with gr.Row():
                zs_auto_transcribe = gr.Checkbox(label="Auto-transcribe with Whisper", value=True)
                zs_ref_text = gr.Textbox(label="Reference Text (optional if auto-transcribe is on)", value="", lines=2)
            with gr.Row():
                zs_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                zs_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                zs_text_frontend = gr.Checkbox(label="Text Frontend", value=True)
            zs_btn = gr.Button("🎙️ Synthesize", variant="primary")
            zs_audio = gr.Audio(label="Output Audio", type="numpy")
            zs_status = gr.Textbox(label="Status", interactive=False)
            zs_btn.click(fn=zero_shot,
                        inputs=[zs_text, zs_ref_audio, zs_speed, zs_seed, zs_text_frontend, zs_auto_transcribe, zs_ref_text],
                        outputs=[zs_audio, zs_status])

        # ── Cross-Lingual ─────────────────────────────────────────────────
        with gr.Tab("🌐 Cross-Lingual"):
            gr.Markdown("Speak text in a different language using the reference voice.")
            cl_text = gr.Textbox(label="Text to synthesize", value="Hello, this is cross-lingual speech synthesis.", lines=3)
            cl_ref_audio = gr.Audio(label="Reference Audio (max 30s)", type="numpy")
            cl_target_lang = gr.Dropdown(
                choices=["auto", "zh", "en", "ja", "ko", "de", "es", "fr", "it", "ru"],
                value="auto",
                label="Target Language"
            )
            with gr.Row():
                cl_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                cl_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                cl_text_frontend = gr.Checkbox(label="Text Frontend", value=True)
            cl_btn = gr.Button("🌐 Synthesize", variant="primary")
            cl_audio = gr.Audio(label="Output Audio", type="numpy")
            cl_status = gr.Textbox(label="Status", interactive=False)
            cl_btn.click(fn=cross_lingual,
                        inputs=[cl_text, cl_ref_audio, cl_speed, cl_seed, cl_text_frontend, cl_target_lang],
                        outputs=[cl_audio, cl_status])

        # ── Instruct2 ─────────────────────────────────────────────────────
        with gr.Tab("🎭 Instruct2"):
            gr.Markdown("Control speaking style and emotion with instructions. Requires CosyVoice2 or CosyVoice3.")
            i2_text = gr.Textbox(label="Text to synthesize", value="Hello, this is my cloned voice speaking.", lines=3)
            i2_instruct = gr.Textbox(label="Instruct Text",
                                     value="Speak in a warm and friendly tone.", lines=2)
            i2_ref_audio = gr.Audio(label="Reference Audio (max 30s)", type="numpy")
            with gr.Row():
                i2_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                i2_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                i2_text_frontend = gr.Checkbox(label="Text Frontend", value=True)
            i2_btn = gr.Button("🎭 Synthesize", variant="primary")
            i2_audio = gr.Audio(label="Output Audio", type="numpy")
            i2_status = gr.Textbox(label="Status", interactive=False)
            i2_btn.click(fn=instruct2,
                        inputs=[i2_text, i2_instruct, i2_ref_audio, i2_speed, i2_seed, i2_text_frontend],
                        outputs=[i2_audio, i2_status])

        # ── Voice Conversion ──────────────────────────────────────────────
        with gr.Tab("🔄 Voice Conversion"):
            gr.Markdown("Convert source voice to sound like target voice.")
            vc_source = gr.Audio(label="Source Audio", type="numpy")
            vc_target = gr.Audio(label="Target Voice Reference", type="numpy")
            with gr.Row():
                vc_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                vc_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
            vc_btn = gr.Button("🔄 Convert", variant="primary")
            vc_audio = gr.Audio(label="Output Audio", type="numpy")
            vc_status = gr.Textbox(label="Status", interactive=False)
            vc_btn.click(fn=voice_conversion,
                        inputs=[vc_source, vc_target, vc_speed, vc_seed],
                        outputs=[vc_audio, vc_status])

        # ── Speaker Presets ───────────────────────────────────────────────
        with gr.Tab("💾 Speaker Presets"):
            gr.Markdown("Save a speaker preset from reference audio, then clone it later without re-uploading.")
            with gr.Row():
                sp_ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                sp_ref_text = gr.Textbox(label="Reference Text (leave blank to auto-transcribe)", value="", lines=2)
                sp_name = gr.Textbox(label="Speaker Name", value="my_speaker")
            sp_save_btn = gr.Button("💾 Save Speaker", variant="primary")
            sp_save_status = gr.Textbox(label="Save Status", interactive=False)

            sp_save_btn.click(fn=save_speaker,
                             inputs=[sp_ref_audio, sp_ref_text, sp_name],
                             outputs=sp_save_status)

            gr.Markdown("---\n### Clone from saved preset")
            with gr.Row():
                sp_preset = gr.Dropdown(label="Speaker Preset", choices=_list_speaker_presets(), value="")
                sp_refresh = gr.Button("🔄 Refresh List")
                sp_text = gr.Textbox(label="Text to synthesize", value="Hello from my saved speaker preset.", lines=2)
            with gr.Row():
                sp_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")
                sp_seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                sp_text_frontend = gr.Checkbox(label="Text Frontend", value=True)
            sp_clone_btn = gr.Button("🎙️ Clone from Preset", variant="primary")
            sp_audio = gr.Audio(label="Output Audio", type="numpy")
            sp_status = gr.Textbox(label="Status", interactive=False)

            def _refresh_presets():
                return gr.Dropdown(choices=_list_speaker_presets())

            sp_refresh.click(fn=_refresh_presets, outputs=sp_preset)
            sp_clone_btn.click(fn=speaker_clone,
                              inputs=[sp_text, sp_preset, sp_speed, sp_seed, sp_text_frontend],
                              outputs=[sp_audio, sp_status])

        # ── Dialog Multi-Speaker ───────────────────────────────────────────
        with gr.Tab("💬 Multi-Speaker Dialog"):
            gr.Markdown("Генерация диалога с несколькими спикерами (до 4 голосов)")
            
            with gr.Row():
                speaker_a = gr.Audio(label="SPEAKER A Голос", type="numpy")
                speaker_b = gr.Audio(label="SPEAKER B Голос", type="numpy")
            with gr.Row():
                speaker_c = gr.Audio(label="SPEAKER C Голос (опционально)", type="numpy")
                speaker_d = gr.Audio(label="SPEAKER D Голос (опционально)", type="numpy")
            
            dialog_text = gr.Textbox(
                label="Текст диалога",
                value="SPEAKER A: Привет! Как дела?\nSPEAKER B: Отлично, спасибо! А у тебя?\nSPEAKER A: Тоже хорошо, спасибо что спросил.",
                lines=8,
                placeholder="Используйте формат:\nSPEAKER A: Текст спикера A\nSPEAKER B: Текст спикера B\nSPEAKER C: Текст спикера C\nSPEAKER D: Текст спикера D"
            )
            
            with gr.Row():
                dialog_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Скорость речи")
                dialog_seed = gr.Number(value=-1, label="Сид (-1 = случайный)", precision=0)
            
            dialog_btn = gr.Button("💬 Сгенерировать диалог", variant="primary")
            dialog_audio = gr.Audio(label="Итоговый диалог", type="numpy")
            speaker_a_track = gr.Audio(label="Трек SPEAKER A", type="numpy")
            speaker_b_track = gr.Audio(label="Трек SPEAKER B", type="numpy")
            speaker_c_track = gr.Audio(label="Трек SPEAKER C", type="numpy")
            speaker_d_track = gr.Audio(label="Трек SPEAKER D", type="numpy")
            dialog_status = gr.Textbox(label="Статус", lines=3, interactive=False)
            
            dialog_btn.click(fn=generate_dialog,
                            inputs=[dialog_text, speaker_a, speaker_b, speaker_c, speaker_d, dialog_speed, dialog_seed],
                            outputs=[dialog_audio, speaker_a_track, speaker_b_track, speaker_c_track, speaker_d_track, dialog_status])

        gr.Markdown("""
        ---
        *Tip: Use **Audio Crop** first to trim reference audio to 3–10 seconds for best quality.*
        """)

    return demo


# 
#  Entry point
# 

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CosyVoice3 Gradio Web UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create a public gradio link")
    args = parser.parse_args()

    print("=" * 60)
    print(" CosyVoice3 Gradio Web Interface")
    print("=" * 60)
    print(f"Launching on http://{args.host}:{args.port}")
    print("=" * 60)

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=True
    )


if __name__ == "__main__":
    main()
