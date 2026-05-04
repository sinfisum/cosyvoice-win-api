#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice3 REST API Server
==========================
Полнофункциональный API сервер для CosyVoice3
Поддерживает все возможности: клонирование голоса, синтез, диалоги, пресеты

Запуск:
    python_embeded/python.exe api_server.py

Конфигурация через переменные окружения:
    MODEL_PATH: Путь к уже скачанной модели
    HOST: Хост для прослушивания (по умолчанию 127.0.0.1)
    PORT: Порт (по умолчанию 8000)
    DEVICE: Устройство auto/cuda/cpu
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
import base64
import io

import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn



# ── Ensure vendored cosyvoice / utils / nodes are importable ──────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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

# ── Configuration ─────────────────────────────────────────────────────────
MODEL_PATH = "./pretrained_models/Fun-CosyVoice3-0.5B"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
DEVICE = os.getenv("DEVICE", "auto")

app = FastAPI(title="CosyVoice3 API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ───────────────────────────────────────────────────────────────
_current_model_info: Optional[Dict[str, Any]] = None
_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        print("[API] Whisper model is None.")
        try:
            import whisper
            print("[API] Loading Whisper model for auto-transcription...")
            _whisper_model = whisper.load_model("base")
            print("[API] Whisper model loaded.")
        except Exception as e:
            print(f"[API] Failed to load Whisper: {e}")
    return _whisper_model


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper_model()
    if model is None:
        return ""
    try:
        result = model.transcribe(audio_path, language=None)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[API] Transcription error: {e}")
        return ""


def is_v3_model(model_info: Dict[str, Any]) -> bool:
    version = model_info.get("model_version", "").lower()
    return "cosyvoice3" in version or "fun-cosyvoice3" in version


# 
#  Helpers
# 

def _collect_output(generator) -> torch.Tensor:
    chunks = []
    for out in generator:
        speech = out["tts_speech"]
        chunks.append(speech)
    if not chunks:
        raise RuntimeError("Model produced no audio chunks.")
    waveform = torch.cat(chunks, dim=-1)
    return waveform


def _ensure_model_loaded() -> Dict[str, Any]:
    global _current_model_info
    if _current_model_info is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Call /load_model first.")
    return _current_model_info


def _seed_all(seed: int):
    if seed >= 0:
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _audio_to_base64(waveform: np.ndarray, sample_rate: int) -> str:
    buffer = io.BytesIO()
    sf.write(buffer, waveform, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _save_uploaded_audio(upload_file: UploadFile) -> Tuple[int, np.ndarray, str]:
    suffix = Path(upload_file.filename).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload_file.file.read())
    tmp.close()
    
    try:
        data, sr = sf.read(tmp.name)
        return sr, data, tmp.name
    except Exception as e:
        os.unlink(tmp.name)
        raise HTTPException(status_code=400, detail=f"Invalid audio file: {str(e)}")


# 
#  API Models
# 

class LoadModelRequest(BaseModel):
    model_path: str = MODEL_PATH
    device: str = DEVICE


class ZeroShotRequest(BaseModel):
    text: str
    speed: float = 1.0
    seed: int = -1
    text_frontend: bool = True
    auto_transcribe: bool = True
    ref_text: str = ""


class CrossLingualRequest(BaseModel):
    text: str
    speed: float = 1.0
    seed: int = -1
    text_frontend: bool = True
    target_language: str = "auto"


class Instruct2Request(BaseModel):
    text: str
    instruct_text: str
    speed: float = 1.0
    seed: int = -1
    text_frontend: bool = True


class VoiceConversionRequest(BaseModel):
    speed: float = 1.0
    seed: int = -1


class DialogLine(BaseModel):
    speaker_id: str
    text: str


class DialogRequest(BaseModel):
    lines: List[DialogLine]
    speed: float = 1.0
    seed: int = -1


class SaveSpeakerRequest(BaseModel):
    speaker_name: str
    ref_text: str = ""


# 
#  API Endpoints
# 

@app.on_event("startup")
async def startup_event():
    global _current_model_info
    print("=" * 60)
    print(" CosyVoice3 API Server")
    print("=" * 60)
    
    # Автоматическая загрузка модели при старте
    if os.path.exists(MODEL_PATH):
        try:
            print(f"[API] Auto-loading model from: {MODEL_PATH}")
            print(f"[API] check cuda: {torch.cuda.is_available()}")

            if DEVICE == "auto":
                target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                target_device = torch.device(DEVICE)
            
            _current_model_info = get_cached_model(
                model_version="Fun-CosyVoice3-0.5B",
                download_source="HuggingFace",
                device=target_device,
                force_redownload=False,
                force_reload=False,
            )
            print(f"[API] Model loaded successfully!")
            print(f"[API] Device: {_current_model_info['device']}")
            print(f"[API] Sample rate: {_current_model_info['sample_rate']} Hz")
        except Exception as e:
            print(f"[API] Failed to auto-load model: {e}")
    else:
        print(f"[API] Model not found at {MODEL_PATH}, use /load_model endpoint to load")
    
    print(f"[API] Server running on http://{HOST}:{PORT}")
    print(f"[API] Documentation: http://{HOST}:{PORT}/docs")
    print("=" * 60)


@app.post("/load_model", summary="Загрузить модель в память")
async def load_model(request: LoadModelRequest):
    global _current_model_info
    try:
        if request.device == "auto":
            target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            target_device = torch.device(request.device)

        _current_model_info = get_cached_model(
            model_version="Fun-CosyVoice3-0.5B",
            download_source="HuggingFace",
            device=target_device,
            force_redownload=False,
            force_reload=False,
        )
        return {
            "status": "success",
            "device": str(_current_model_info["device"]),
            "sample_rate": _current_model_info["sample_rate"],
            "model_path": _current_model_info["model_path"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload_model", summary="Выгрузить модель из памяти")
async def unload_model():
    global _current_model_info
    _current_model_info = None
    clear_model_cache()
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "success", "message": "Model unloaded"}


@app.get("/status", summary="Статус сервера")
async def get_status():
    global _current_model_info
    return {
        "server": "running",
        "model_loaded": _current_model_info is not None,
        "model_info": _current_model_info["model_name"] if _current_model_info else None,
        "device": str(_current_model_info["device"]) if _current_model_info else None,
        "sample_rate": _current_model_info["sample_rate"] if _current_model_info else None
    }

@app.post("/zero_shot", summary="Zero-Shot клонирование голоса")
async def api_zero_shot(
    text: str = Form(...),
    speed: float = Form(1.0),
    seed: int = Form(-1),
    text_frontend: bool = Form(True),
    auto_transcribe: bool = Form(True),
    ref_text: str = Form(""),
    audio_file: UploadFile = File(...)
):
    model_info = _ensure_model_loaded()
    sr, data, tmp_path = _save_uploaded_audio(audio_file)
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
    duration = data.shape[0] / sr
    if duration > 30:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Reference audio too long ({duration:.1f}s). Max 30s.")
    
    try:
        _seed_all(seed)
        model = model_info["model"]
        is_v3 = is_v3_model(model_info)

        if auto_transcribe or not ref_text.strip():
            prompt_text = transcribe_audio(tmp_path)
            if not prompt_text:
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
                waveform = _collect_output(gen)
                if waveform.device != torch.device('cpu'):
                    waveform = waveform.cpu()
                return {
                    "status": "success",
                    "mode": "cross_lingual_fallback",
                    "sample_rate": model_info["sample_rate"],
                    "duration": waveform.shape[-1] / model_info["sample_rate"],
                    "audio_base64": _audio_to_base64(waveform.numpy().T, model_info["sample_rate"])
                }
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
        waveform = _collect_output(gen)
        if waveform.device != torch.device('cpu'):
            waveform = waveform.cpu()
        
        return {
            "status": "success",
            "mode": "zero_shot",
            "sample_rate": model_info["sample_rate"],
            "duration": waveform.shape[-1] / model_info["sample_rate"],
            "transcript": prompt_text,
            "audio_base64": _audio_to_base64(waveform.numpy().T, model_info["sample_rate"])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)



@app.post("/cross_lingual", summary="Многоязычный синтез")
async def api_cross_lingual(
    text: str = Form(...),
    target_language: str = Form("auto"),
    speed: float = Form(1.0),
    seed: int = Form(-1),
    text_frontend: bool = Form(True),
    audio_file: UploadFile = File(...)
):
    model_info = _ensure_model_loaded()
    sr, data, tmp_path = _save_uploaded_audio(audio_file)
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
    duration = data.shape[0] / sr
    if duration > 30:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Reference audio too long ({duration:.1f}s). Max 30s.")
    
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
        waveform = _collect_output(gen)
        if waveform.device != torch.device('cpu'):
            waveform = waveform.cpu()
        
        return {
            "status": "success",
            "sample_rate": model_info["sample_rate"],
            "duration": waveform.shape[-1] / model_info["sample_rate"],
            "audio_base64": _audio_to_base64(waveform.numpy().T, model_info["sample_rate"])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


@app.post("/instruct2", summary="Синтез с контролем стиля")
async def api_instruct2(
    text: str = Form(...),
    instruct_text: str = Form(...),
    speed: float = Form(1.0),
    seed: int = Form(-1),
    text_frontend: bool = Form(True),
    audio_file: UploadFile = File(...)
):
    model_info = _ensure_model_loaded()
    sr, data, tmp_path = _save_uploaded_audio(audio_file)
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
    duration = data.shape[0] / sr
    if duration > 30:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Reference audio too long ({duration:.1f}s). Max 30s.")
    
    try:
        _seed_all(seed)
        model = model_info["model"]
        if not hasattr(model, "inference_instruct2"):
            raise HTTPException(status_code=501, detail="inference_instruct2 is not available. Use CosyVoice2 or CosyVoice3.")

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
        waveform = _collect_output(gen)
        if waveform.device != torch.device('cpu'):
            waveform = waveform.cpu()
        
        return {
            "status": "success",
            "sample_rate": model_info["sample_rate"],
            "duration": waveform.shape[-1] / model_info["sample_rate"],
            "audio_base64": _audio_to_base64(waveform.numpy().T, model_info["sample_rate"])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


@app.post("/voice_conversion", summary="Конвертация голоса")
async def api_voice_conversion(
    request: VoiceConversionRequest = Body(...),
    source_file: UploadFile = File(...),
    target_file: UploadFile = File(...)
):
    model_info = _ensure_model_loaded()
    
    sr_s, data_s, src_tmp = _save_uploaded_audio(source_file)
    sr_t, data_t, tgt_tmp = _save_uploaded_audio(target_file)
    
    for label, data, sr in [("source", data_s, sr_s), ("target", data_t, sr_t)]:
        if data.ndim == 1:
            data = data[:, np.newaxis]
        dur = data.shape[0] / sr
        if dur > 30:
            cleanup_temp_file(src_tmp)
            cleanup_temp_file(tgt_tmp)
            raise HTTPException(status_code=400, detail=f"{label} audio too long ({dur:.1f}s). Max 30s.")
    
    try:
        _seed_all(request.seed)
        model = model_info["model"]
        if not hasattr(model, "inference_vc"):
            raise HTTPException(status_code=501, detail="inference_vc is not available on this model.")

        gen = model.inference_vc(
            source_wav=src_tmp,
            prompt_wav=tgt_tmp,
            stream=False,
            speed=request.speed,
        )
        waveform = _collect_output(gen)
        if waveform.device != torch.device('cpu'):
            waveform = waveform.cpu()
        
        return {
            "status": "success",
            "sample_rate": model_info["sample_rate"],
            "duration": waveform.shape[-1] / model_info["sample_rate"],
            "audio_base64": _audio_to_base64(waveform.numpy().T, model_info["sample_rate"])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(src_tmp)
        cleanup_temp_file(tgt_tmp)




@app.post("/dialog", summary="Многоголосовой диалог")
async def api_dialog(
    lines: str = Form(...),
    speed: float = Form(1.0),
    seed: int = Form(42),
    speaker_a: Optional[UploadFile] = File(None),
    speaker_b: Optional[UploadFile] = File(None),
    speaker_c: Optional[UploadFile] = File(None),
    speaker_d: Optional[UploadFile] = File(None),
):
    model_info = _ensure_model_loaded()
    model = model_info["model"]
    sample_rate = model.sample_rate
    is_v3 = is_v3_model(model_info)
    
    speakers = {}
    temp_files = []
    
    # 1. Безопасный парсинг входящих строк
    try:
        parsed_lines = json.loads(lines)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка в формате JSON lines: {str(e)}")
    
    try:
        # 2. Сохраняем только загруженные голоса
        available_uploads = [
            ("A", speaker_a), ("B", speaker_b), 
            ("C", speaker_c), ("D", speaker_d)
        ]
        
        for sid, file in available_uploads:
            # Проверяем, что файл реально передан (есть имя и контент)
            if file and file.filename:
                # Предполагаем, что функция возвращает (sr, data, tmp_path)
                _, _, tmp_path = _save_uploaded_audio(file)
                speakers[sid] = tmp_path
                temp_files.append(tmp_path)

        if not speakers:
            raise HTTPException(status_code=400, detail="Не загружено ни одного аудио-файла для спикеров")

        # 3. Настройка генератора
        if seed is not None:
            _seed_all(seed)
            
        all_waveforms = []

        # 4. Генерация по каждой реплике
        for idx, line in enumerate(parsed_lines):
            sid = line.get("speaker_id")
            text = line.get("text", "").strip()

            if not text:
                continue
                
            print(f"[API] Реплика {idx+1}/{len(parsed_lines)}: SPEAKER {sid}")
            
            # Пропускаем, если для этого ID не загружен файл
            if sid not in speakers:
                print(f"[API] Пропуск: Голос для спикера {sid} не найден")
                continue
            
            tmp_path = speakers[sid]
            
            # Форматирование для v3
            tts_text = f"You are a helpful assistant.<|endofprompt|>{text}" if is_v3 else text
            
            # Выполнение инференса
            # Используем cross_lingual или zero_shot в зависимости от ваших нужд
            gen = model.inference_cross_lingual(
                tts_text=tts_text,
                prompt_wav=tmp_path,
                stream=False,
                speed=speed,
                text_frontend=True,
            )
            
            waveform = _collect_output(gen)
            
            # Перенос на CPU для конкатенации
            if waveform.device.type != 'cpu':
                waveform = waveform.cpu()
            
            all_waveforms.append(waveform)
        
        # 5. Сборка финального аудио
        if not all_waveforms:
            raise ValueError("Ни одна реплика не была успешно сгенерирована")
            
        combined = torch.cat(all_waveforms, dim=-1)
        
        return {
            "status": "success",
            "sample_rate": sample_rate,
            "duration": combined.shape[-1] / sample_rate,
            "lines_processed": len(all_waveforms),
            "audio_base64": _audio_to_base64(combined.numpy().T, sample_rate)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Возвращаем 500 ошибку с деталями для отладки
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")
        
    finally:
        # 6. Гарантированная очистка временных файлов
        for tmp_path in temp_files:
            try:
                cleanup_temp_file(tmp_path)
            except:
                pass

@app.post("/save_speaker", summary="Сохранить пресет спикера")
async def api_save_speaker(
    speaker_name: str = Form(...),
    ref_text: str = Form(""),
    audio_file: UploadFile = File(...)
):
    model_info = _ensure_model_loaded()
    sr, data, tmp_path = _save_uploaded_audio(audio_file)
    
    if data.ndim == 1:
        data = data[:, np.newaxis]
    duration = data.shape[0] / sr
    if duration > 30:
        os.unlink(tmp_path)
        raise HTTPException(status_code=400, detail=f"Reference audio too long ({duration:.1f}s). Max 30s.")
    
    try:
        model = model_info["model"]
        if not ref_text.strip():
            ref_text = transcribe_audio(tmp_path)

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

        speaker_dir = "./pretrained_models/cosyvoice/speaker"
        os.makedirs(speaker_dir, exist_ok=True)
        save_path = os.path.join(speaker_dir, f"{speaker_name}.pt")
        torch.save(spk2info, save_path)
        
        return {
            "status": "success",
            "speaker_name": speaker_name,
            "saved_path": save_path,
            "transcript": ref_text
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cleanup_temp_file(tmp_path)


@app.get("/list_speakers", summary="Список сохраненных пресетов спикеров")
async def api_list_speakers():
    speaker_dir = "./pretrained_models/cosyvoice/speaker"
    if not os.path.isdir(speaker_dir):
        return {"speakers": []}
    speakers = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(speaker_dir)
        if f.endswith(".pt")
    ])
    return {"speakers": speakers}


@app.post("/speaker_clone", summary="Синтез из сохраненного пресета")
async def api_speaker_clone(
    text: str = Form(...),
    speaker_name: str = Form(...),
    speed: float = Form(1.0),
    seed: int = Form(-1),
    text_frontend: bool = Form(True)
):
    model_info = _ensure_model_loaded()
    
    speaker_dir = "./pretrained_models/cosyvoice/speaker"
    pt_path = os.path.join(speaker_dir, f"{speaker_name}.pt")
    if not os.path.isfile(pt_path):
        raise HTTPException(status_code=404, detail=f"Speaker preset not found: {speaker_name}")
    
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
        waveform = _collect_output(gen)
        if waveform.device != torch.device('cpu'):
            waveform = waveform.cpu()
        
        return {
            "status": "success",
            "sample_rate": model_info["sample_rate"],
            "duration": waveform.shape[-1] / model_info["sample_rate"],
            "audio_base64": _audio_to_base64(waveform.numpy().T, model_info["sample_rate"])
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/dialog-zero-shot")
async def api_dialog_zero_shot(
    lines: str = Form(...),          # JSON string: [{"speaker_id": "A", "text": "..."}]
    speed: float = Form(1.0),
    seed: int = Form(42),
    speaker_a: UploadFile = File(None),
    speaker_b: UploadFile = File(None),
):
    model_info = _ensure_model_loaded()
    model = model_info["model"]
    is_v3 = is_v3_model(model_info)
    
    # 1. Parse lines
    import json
    try:
        data_lines = json.loads(lines)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON in lines field")

    # 2. Save uploaded files to temp storage
    speakers_paths = {}
    temp_files = []
    
    for sid, file in [("A", speaker_a), ("B", speaker_b)]:
        if file:
            _, _, tmp_path = _save_uploaded_audio(file)
            speakers_paths[sid] = tmp_path
            temp_files.append(tmp_path)

    try:
        _seed_all(seed)
        all_waveforms = []

        for line in data_lines:
            sid = line.get("speaker_id")
            text = line.get("text")
            
            if sid not in speakers_paths:
                continue

            # v3 Formatting
            tts_text = f"You are a helpful assistant.<|endofprompt|>{text}" if is_v3 else text
            
            # Zero-shot inference using the prompt_wav
            gen = model.inference_zero_shot(
                tts_text=tts_text,
                prompt_text="", # Optional: text content of the prompt_wav for better quality
                prompt_wav=speakers_paths[sid],
                speed=speed
            )
            
            waveform = _collect_output(gen)
            all_waveforms.append(waveform.cpu())

        combined = torch.cat(all_waveforms, dim=-1) if all_waveforms else torch.zeros(1, 1)
        return {"audio_base64": _audio_to_base64(combined.numpy().T, model.sample_rate)}

    finally:
        for p in temp_files: cleanup_temp_file(p)

@app.post("/dialog-sft")
async def api_dialog_sft(
    lines: str = Form(...), # JSON string
    speed: float = Form(1.0),
    seed: int = Form(42)
):
    model_info = _ensure_model_loaded()
    model = model_info["model"]
    is_v3 = is_v3_model(model_info)

    import json
    data_lines = json.loads(lines)

    _seed_all(seed)
    all_waveforms = []

    # Map Speaker A/B to internal CosyVoice IDs
    voice_mapping = {"A": "speech_edit", "B": "chinese_female"} 

    for line in data_lines:
        sid = line.get("speaker_id")
        text = line.get("text")
        
        target_voice = voice_mapping.get(sid, "speech_edit")
        tts_text = f"You are a helpful assistant.<|endofprompt|>{text}" if is_v3 else text

        # SFT inference: no files needed, only speaker_id
        gen = model.inference_sft(
            tts_text=tts_text,
            speaker_id=target_voice,
            speed=speed
        )
        
        waveform = _collect_output(gen)
        all_waveforms.append(waveform.cpu())

    combined = torch.cat(all_waveforms, dim=-1)
    return {"audio_base64": _audio_to_base64(combined.numpy().T, model.sample_rate)}



if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)