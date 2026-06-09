# OCR Система для распознавания текста на городских вывесках

## Описание

Система распознавания текста на фотографиях городской среды с сравнением 5 архитектур нейронных сетей.

## Архитектуры

1. **EasyOCR** - CNN+RNN+Attention+BiLSTM
2. **CRNN** - CNN+RNN+CTC
3. **Transformer** - ViT+Attention (TrOCR)
4. **Гибридная** - CNN+Transformer+CTC
5. **Ансамбль** - Комитет из 5 моделей

## Установка и запуск

```bash
# Клонирование репозитория
git clone https://github.com/ваш-username/ocr-city-signs-recognition.git
cd ocr-city-signs-recognition

# Установка зависимостей
pip install -r requirements.txt

# Запуск
python app.py
```
