# CosyVoice3 API server and web ui for Windows

Advanced text-to-speech nodes for ComfyUI powered by the CosyVoice3 model family. Features zero-shot voice cloning, cross-lingual synthesis, and voice conversion.

## Features

- **Zero-Shot Voice Cloning** - Clone any voice from 3-30 seconds of reference audio
- **Cross-Lingual Synthesis** - Speak different languages while preserving voice characteristics
- **Voice Conversion** - Transform one voice to sound like another
- **9 Languages** - Chinese, English, Japanese, Korean, German, Spanish, French, Italian, Russian
- **Auto Transcription** - Built-in Whisper integration for reference audio
- **Speed Control** - Adjustable speech rate (0.5x - 2.0x)

## fast Installation

- donwload cosyVoiceApi_v01.7z
- unpack using WinRar or 7zip to D:\AI\cosyvoiceapi or other path

1. #launch web app
```bash
python_embeded\python.exe web_app.py
```
---------
or
---------

1. #launch API server
```bash
python_embeded\python.exe api_server.py
```

2. #launch test client
```bash
python_embeded\python.exe api_test.py
```

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/sinfisum/CosyVoice3API.git
cd CosyVoice3API

 C:\Python313\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
python web_app.py
```
---------
or
---------

1. #launch API server
```bash
python api_server.py
```

2. #launch test client
```bash
python api_test.py
```

## Models

| Model | Size | Notes |
|-------|------|-------|
| Fun-CosyVoice3-0.5B-2512 | ~2GB | Recommended |
| Fun-CosyVoice3-0.5B | ~2GB | Alternative |
| CosyVoice2-0.5B | ~2GB | Alternative |
| CosyVoice-ttsfrd | ~350MB | for linux |
| SenseVoiceSmall | ~900MB | ASR |
| Whisper | ~1600MB | ASR |

Download models to:
CosyVoice-ttsfrd --> `pretrained_models/CosyVoice-ttsfrd/`
SenseVoiceSmall --> `pretrained_models/SenseVoiceSmall/`

Fun-CosyVoice3-0.5B-2512 --> `pretrained_models/Fun-CosyVoice3-0.5B/`
CosyVoice2-0.5B --> `pretrained_models/CosyVoice2-0.5B/`

## Requirements

- Python 3.13
- 8GB RAM minimum (12GB+ recommended)
- NVIDIA GPU with 8GB+ VRAM recommended (CPU and Mac MPS supported)

## License

Apache 2.0