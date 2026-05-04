#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice3 API Test Script
==========================
Скрипт для полного тестирования всех функций API сервера
Автоматически генерирует тестовые данные, вызывает все эндпоинты и сохраняет результаты
"""

import os
import sys
import json
import base64
import random
import tempfile
from datetime import datetime
import requests
from pathlib import Path
import uuid


API_URL = "http://127.0.0.1:8000"
OUTPUT_DIR = "OUTPUT"

# Примеры текстов для генерации голосов
SPEAKER_EXAMPLES = [
    "Добрый день! Рад приветствовать вас. Меня зовут Алексей, я буду вашим гидом по этому проекту.",
    "Привет! Я очень рад вас видеть. Давайте сегодня обсудим планы на следующую неделю.",
    "Здравствуйте! Сегодня мы будем говорить про новые технологии в области искусственного интеллекта.",
    "Доброе утро! Надеюсь у вас прекрасное настроение. Сегодня отличный день для новых начинаний.",
    "Приветствую! Мне очень нравится работать с вами. Давайте создадим что-то замечательное вместе.",
    "Всем привет! Сегодня у нас очень интересная тема для обсуждения. Пожалуйста, задавайте вопросы.",
    "Добрый вечер! Сегодня мы поговорим про будущее голосовых технологий и как они изменят нашу жизнь.",
    "Здравствуйте! Я ваш персональный ассистент. Чем я могу вам помочь сегодня?",
    "Привет! Очень рад нашему знакомству. Давайте начнём наше сотрудничество с приятного разговора.",
    "Добрый день! Сегодня я расскажу вам про самые интересные возможности нашего сервиса."
]

DIALOG_EXAMPLE = """
SPEAKER A: Привет! Как дела?
SPEAKER B: Отлично, спасибо! А у тебя?
SPEAKER A: Тоже хорошо, спасибо что спросил.
SPEAKER B: Знаешь, я сегодня начал изучать новый язык программирования.
SPEAKER A: Правда? Это здорово! Какой именно?
SPEAKER B: Rust. Мне очень нравится как он спроектирован.
SPEAKER A: Да, я слышал много хорошего про него. Но он довольно сложный для изучения.
SPEAKER B: Да, но это того стоит. Безопасность памяти стоит того.
"""


VOICE_ALIAS_MAP = {
    "speaker_A": "./voices/speaker_A.wav",
    "speaker_B": "./voices/speaker_C.wav",
    "speaker_C": "./voices/speaker_C.wav",
    "speaker_D": "./voices/speaker_A.wav",
}


# <sad>sad emotion.</sad>	
# <surprised>emotion.</surprised>	
# <angry>emotion.</angry>	
# <fearful>emotion.</fearful>	
# <fast>emotion.</fast>	
# <slow>emotion.</slow>	
# <peppa>emotion.</peppa>	
# <robot>emotion.</robot>

INSTRUCT_EXAMPLES = [
    " <slow>Это пример синтеза с контролем стиля и эмоций. Очень медленно и спокойно. </slow>",
]


def decode_and_save_audio(base64_data, filename):
    """Декодирует base64 аудио и сохраняет в файл"""
    data = base64.b64decode(base64_data)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(data)
    return filepath


def test_status():
    """Проверка статуса сервера"""
    print("\n=== Тест /status ===")
    try:
        r = requests.get(f"{API_URL}/status")
        r.raise_for_status()
        status = r.json()
        print(f"✓ Сервер работает")
        print(f"  Модель загружена: {status['model_loaded']}")
        if status['model_loaded']:
            print(f"  Модель: {status['model_info']}")
            print(f"  Устройство: {status['device']}")
            print(f"  Частота дискретизации: {status['sample_rate']}")
        return status['model_loaded']
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def generate_speaker_audio(speaker_name):
    text = random.choice(SPEAKER_EXAMPLES)
    print(f"\n=== Генерация голоса для {speaker_name} ===")
    print(f"Текст: {text[:80]}...")
    
    try:
        tmp_path = f"{VOICE_ALIAS_MAP[speaker_name]}"
        print(f"tmp_path: {tmp_path}")
        
        with open(tmp_path, "rb") as f:
            files = {"audio_file": f}
            data = {
                "text": text,
                "speed": 1.0,
                # "seed": random.randint(1, 1000000),
                "auto_transcribe": True,
                "ref_text": ""
            }
            
            r = requests.post(
                f"{API_URL}/zero_shot",
                files=files,
                data=data
            )
        
        # os.unlink(tmp_path) - удаление
        
        r.raise_for_status()
        result = r.json()
        
        filename = f"{speaker_name}_voice.wav"
        decode_and_save_audio(result['audio_base64'], filename)
        print(f"✓ Голос сгенерирован: {filename}")
        print(f"  Длительность: {result['duration']:.2f} сек")
        
        return os.path.join(OUTPUT_DIR, filename)
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return None


def test_instruct2(voice_file):
    """Тест инструкционного синтеза"""
    print("\n=== Тест /instruct2 ===")
    
    text = random.choice(INSTRUCT_EXAMPLES)
    

    instruct = "You are a helpful assistant.<|endofprompt|>"

    print(f"Текст: {text}")
    print(f"Инструкция: {instruct}")
    
    try:
        with open(voice_file, "rb") as f:
            files = {"audio_file": f}
            data = {
                "text": text,
                "instruct_text": instruct,
                # "speed": 1.0,
                # "seed": random.randint(1, 1000000),
                # "text_frontend": True
            }
            
            r = requests.post(
                f"{API_URL}/instruct2",
                files=files,
                data=data
            )
        
        r.raise_for_status()
        result = r.json()
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"test_instruct2_{timestamp}.wav"

        decode_and_save_audio(result['audio_base64'], filename)
        print(f"✓ Инструкционный синтез завершен: {filename}")
        print(f"  Длительность: {result['duration']:.2f} сек")
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def test_cross_lingual(voice_file):
    """Тест многоязычного синтеза"""
    print("\n=== Тест /cross_lingual ===")
    
    text = "Hello! This is cross-lingual speech synthesis. The same voice but in English language."
    print(f"Текст (на английском): {text}")
    
    try:
        with open(voice_file, "rb") as f:
            files = {"audio_file": f}
            data = {
                "text": text,
                "target_language": "en",
                "speed": 1.0,
                "seed": random.randint(1, 1000000),
                "text_frontend": True
            }
            
            r = requests.post(
                f"{API_URL}/cross_lingual",
                files=files,
                data=data
            )
        
        r.raise_for_status()
        result = r.json()
        
        filename = f"test_cross_lingual.wav"
        decode_and_save_audio(result['audio_base64'], filename)
        print(f"✓ Многоязычный синтез завершен: {filename}")
        print(f"  Длительность: {result['duration']:.2f} сек")
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def test_dialog(speaker_files):
    """Тест многоголосового диалога"""
    print("\n=== Тест /dialog ===")
    
    # Текст примера (убедитесь, что он определен в коде)
    DIALOG_EXAMPLE = """
SPEAKER A: Привет! Как дела?
SPEAKER B: Отлично, спасибо! А у тебя?
"""
    
    # 1. Парсим текст в список словарей
    lines = []
    for line in DIALOG_EXAMPLE.strip().splitlines():
        if ": " in line:
            speaker_part, text_part = line.split(": ", 1)
            sid = speaker_part.replace("SPEAKER ", "").strip()
            lines.append({"speaker_id": sid, "text": text_part.strip()})
    
    if not lines:
        print("✗ Ошибка: Не удалось распарсить диалог.")
        return False

    print(f"Реплик в диалоге: {len(lines)}")
    
    # 2. Подготавливаем файлы и данные
    files = {}
    opened_files = []
    try:
        # Важно: ключи в files должны совпадать с именами аргументов на сервере
        # speaker_a, speaker_b, speaker_c, speaker_d
        for sid, path in speaker_files.items():
            if path and os.path.exists(path):
                f = open(path, "rb")
                # Приводим "A" -> "speaker_a"
                files[f"speaker_{sid.lower()}"] = f
                opened_files.append(f)
        
        # Данные отправляем как простые поля формы
        # lines обязательно упаковываем в JSON-строку
        data = {
            "lines": json.dumps(lines, ensure_ascii=False),
            "speed": 1.0,
            "seed": random.randint(1, 1000000)
        }

        # 3. Выполняем запрос
        print("Отправка запроса на сервер...")
        r = requests.post(
            f"{API_URL}/dialog",
            files=files,
            data=data
        )
        
        # Сразу закрываем файлы
        for f in opened_files:
            f.close()
            
        # Проверяем ответ
        if r.status_code != 200:
            print(f"✗ Ошибка сервера ({r.status_code}): {r.text}")
            return False
            
        result = r.json()
        
        # 4. Сохраняем результат
        if 'audio_base64' in result:
            filename = "test_dialog.wav"
            decode_and_save_audio(result['audio_base64'], filename)
            print(f"✓ Диалог успешно сгенерирован: {filename}")
            print(f"  Реплик обработано: {result.get('lines_processed')}")
            print(f"  Длительность: {result.get('duration', 0):.2f} сек")
            return True
        else:
            print(f"✗ Ошибка: В ответе нет аудио. Ответ: {result}")
            return False

    except Exception as e:
        # Закрываем файлы в случае падения
        for f in opened_files:
            f.close()
        print(f"✗ Ошибка при выполнении запроса: {e}")
        return False


def test_save_and_clone_speaker(voice_file):
    """Тест сохранения и клонирования спикера"""
    print("\n=== Тест сохранения пресета спикера ===")
    
    speaker_name = f"test_speaker_{random.randint(1000, 9999)}"
    
    try:
        with open(voice_file, "rb") as f:
            files = {"audio_file": f}
            data = {"speaker_name": speaker_name, "ref_text": ""}
            
            r = requests.post(
                f"{API_URL}/save_speaker",
                files=files,
                data=data
            )
        
        r.raise_for_status()
        result = r.json()
        print(f"✓ Спикер сохранен: {speaker_name}")
        print(f"  Путь: {result['saved_path']}")
        
        print("\n=== Тест клонирования из пресета ===")
        text = "Это голос сохраненного спикера. Теперь его можно использовать без повторной загрузки аудио."
        
        r = requests.post(
            f"{API_URL}/speaker_clone",
            data={
                "text": text,
                "speaker_name": speaker_name,
                "speed": 1.0,
                "seed": random.randint(1, 1000000),
                "text_frontend": True
            }
        )
        
        r.raise_for_status()
        result = r.json()
        
        filename = f"test_speaker_clone.wav"
        decode_and_save_audio(result['audio_base64'], filename)
        print(f"✓ Клонирование из пресета завершено: {filename}")
        print(f"  Длительность: {result['duration']:.2f} сек")
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def test_list_speakers():
    """Тест списка сохраненных спикеров"""
    print("\n=== Тест /list_speakers ===")
    try:
        r = requests.get(f"{API_URL}/list_speakers")
        r.raise_for_status()
        result = r.json()
        print(f"✓ Сохраненных спикеров: {len(result['speakers'])}")
        for spk in result['speakers']:
            print(f"  - {spk}")
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        return False


def test_sft_dialog():
    print("\n=== Тест /dialog-sft (Встроенные голоса) ===")
    
    # Подготавливаем данные диалога
    # speaker_id должен соответствовать тем, что прописаны в коде сервера
    lines = [
        {"speaker_id": "A", "text": "Добрый день! Как мне пройти в библиотеку?"},
        {"speaker_id": "B", "text": "Здравствуйте. Вам нужно повернуть направо, а затем идти прямо."},
        {"speaker_id": "A", "text": "Большое спасибо за помощь!"}
    ]
    
    data = {
        "lines": json.dumps(lines, ensure_ascii=False),
        "speed": 1.0,
        "seed": random.randint(1, 1000000)
    }

    try:
        # Для SFT используем обычный POST запрос. 
        # Т.к. файлов нет, requests отправит это как application/x-www-form-urlencoded,
        # что идеально подходит для FastAPI Form(...)
        response = requests.post(f"{API_URL}/dialog-sft", data=data)
        
        if response.status_code != 200:
            print(f"✗ Ошибка: {response.status_code}")
            print(f"Детали: {response.text}")
            return

        result = response.json()
        
        # Сохраняем полученный результат
        if "audio_base64" in result:
            output_file = "test_sft_result.wav"
            audio_data = base64.b64decode(result["audio_base64"])
            with open(output_file, "wb") as f:
                f.write(audio_data)
            
            print(f"✓ Успех! Файл сохранен как: {output_file}")
            print(f"  Параметры: speed={data['speed']}, seed={data['seed']}")
        else:
            print(f"✗ Ошибка: Сервер не вернул аудио. Ответ: {result}")

    except Exception as e:
        print(f"✗ Критическая ошибка при запросе: {e}")

if __name__ == "__main__":
    test_sft_dialog()






def test_dialog_zero_shot(speaker_files):
    """
    Тест функции /dialog-zero-shot.
    speaker_files: словарь вида {"A": "path/to/voice_a.wav", "B": "path/to/voice_b.wav"}
    """
    print("\n=== Тест /dialog-zero-shot ===")
    
    # 1. Формируем диалог
    lines = [
        {"speaker_id": "A", "text": "<slow>Привет! Послушай, как медленно я могу говорить.</slow>"},
        {"speaker_id": "B", "text": "Ого! А я могу ответить очень быстро и весело!"},
        {"speaker_id": "A", "text": "Это потрясающе, технологии CosyVoice v3 удивляют."}
    ]
    
    # 2. Подготавливаем файлы (открываем их в бинарном режиме)
    files = {}
    opened_handlers = []
    
    for sid, path in speaker_files.items():
        if path and os.path.exists(path):
            f = open(path, "rb")
            # Ключи должны быть speaker_a, speaker_b и т.д.
            files[f"speaker_{sid.lower()}"] = f
            opened_handlers.append(f)
        else:
            print(f"⚠ Предупреждение: файл для {sid} не найден по пути {path}")

    # 3. Данные запроса
    data = {
        "lines": json.dumps(lines, ensure_ascii=False),
        "speed": 1.0,
        "seed": random.randint(1, 1000000)
    }

    try:
        # 4. Отправляем multipart/form-data запрос
        print("Генерация аудио...")
        response = requests.post(
            f"{API_URL}/dialog-zero-shot",
            files=files,
            data=data
        )
        
        # Закрываем файлы сразу после отправки
        for f in opened_handlers:
            f.close()

        if response.status_code != 200:
            print(f"✗ Ошибка {response.status_code}: {response.text}")
            return False

        result = response.json()
        
        # 5. Сохранение результата
        if "audio_base64" in result:
            filename = "result_zero_shot.wav"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(result["audio_base64"]))
            print(f"✓ Готово! Файл сохранен: {filename}")
            return True
        else:
            print(f"✗ Ошибка: Сервер не вернул аудио токен.")
            return False

    except Exception as e:
        print(f"✗ Критическая ошибка: {e}")
        return False

# Пример запуска:
# test_dialog_zero_shot({"A": "voices/man.wav", "B": "voices/woman.wav"})




def main():
    print("=" * 60)
    print(" CosyVoice3 API Test Script")
    print("=" * 60)
    
    # Создаем папку для вывода
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Результаты будут сохранены в папку: {os.path.abspath(OUTPUT_DIR)}")
    
    # Проверяем статус сервера
    if not test_status():
        print("\n❌ Сервер не запущен. Запустите сначала api_server.py")
        return
    
    print("\n" + "=" * 60)
    print(" Генерация тестовых голосов для 4 спикеров")
    print("=" * 60)
    
    speaker_files = {}
    for sid in ["A", "B", "C", "D"]:
        path = generate_speaker_audio(f"speaker_{sid}")
        speaker_files[sid] = path
        print(" Тестирование всех функций API")
    
    # Тестируем все функции API
    print("\n" + "=" * 60)
    print(" Тестирование всех функций API")
    print("=" * 60)
    
    # Тестируем на примере первого спикера
    test_voice = speaker_files.get("A")
    if test_voice and os.path.exists(test_voice):
        test_instruct2(test_voice)
        test_cross_lingual(test_voice)
        test_save_and_clone_speaker(test_voice)
        print(" Тестирование всех функций API")
    
    # Тестируем диалог
    test_dialog(speaker_files)
    
    # Список спикеров
    test_list_speakers()
    
    print("\n" + "=" * 60)
    print(" Все тесты завершены!")
    print("=" * 60)
    print(f"Все сгенерированные аудио файлы сохранены в папку: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()