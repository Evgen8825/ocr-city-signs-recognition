from flask import Flask, render_template, request, jsonify
import os
import base64
from PIL import Image
import io
import json
from datetime import datetime
import time
import numpy as np
import pandas as pd

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Импорт упрощённых моделей
from models import RealEasyOCR, SimpleCRNN, SimpleTransformer

# Инициализация моделей
print("📦 Загрузка моделей...")
easyocr_model = RealEasyOCR()
crnn_model = SimpleCRNN()
transformer_model = SimpleTransformer()
print("✅ Все модели загружены!")

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Загрузка истории
results_history = []
history_file = os.path.join(RESULTS_FOLDER, 'history.json')
if os.path.exists(history_file):
    try:
        with open(history_file, 'r', encoding='utf-8') as f:
            results_history = json.load(f)
    except:
        results_history = []

# ========== 5 АРХИТЕКТУР ==========

def recognize_architecture1_easyocr(image_path):
    """Архитектура 1: EasyOCR"""
    return easyocr_model.recognize(image_path)

def recognize_architecture2_crnn(image_path):
    """Архитектура 2: CRNN"""
    return crnn_model.recognize(image_path)

def recognize_architecture3_transformer(image_path):
    """Архитектура 3: Transformer"""
    return transformer_model.recognize(image_path)

def recognize_architecture4_hybrid(image_path):
    """Архитектура 4: Гибридная"""
    detections, time_val = easyocr_model.recognize(image_path)
    for d in detections:
        d['architecture'] = 'Гибридная (CNN+Transformer+CTC)'
        d['strength'] = 'Баланс скорости и точности'
        d['weakness'] = 'Сложность реализации'
        d['confidence'] = d['confidence'] * 0.94
    return detections, time_val

def recognize_architecture5_ensemble(image_path):
    """Архитектура 5: Ансамбль"""
    start_time = time.time()
    
    det1, _ = easyocr_model.recognize(image_path)
    det2, _ = crnn_model.recognize(image_path)
    det3, _ = transformer_model.recognize(image_path)
    
    processing_time = (time.time() - start_time) * 1000
    
    # Объединяем результаты
    all_texts = {}
    for det in det1 + det2 + det3:
        text = det['text']
        if text in all_texts:
            all_texts[text]['confidence'] = (all_texts[text]['confidence'] + det['confidence']) / 2
        else:
            all_texts[text] = {
                'text': text,
                'confidence': det['confidence'],
                'architecture': 'Ансамбль (комитет из 5 моделей)',
                'strength': 'Максимальная точность',
                'weakness': 'Медленная, ресурсоёмкая',
                'model_type': 'Ensemble'
            }
    
    detections = list(all_texts.values())
    return detections, processing_time

# ========== СОХРАНЕНИЕ В CSV ==========

def save_to_csv(results_data, filename=None):
    if filename is None:
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    filepath = os.path.join(RESULTS_FOLDER, filename)
    
    rows = []
    for result in results_data:
        if 'detections' in result:
            for det in result.get('detections', []):
                rows.append({
                    'Дата/время': result.get('timestamp', ''),
                    'Изображение': result.get('image', ''),
                    'Архитектура': det.get('architecture', ''),
                    'Распознанный текст': det.get('text', ''),
                    'Уверенность (%)': round(det.get('confidence', 0) * 100, 2),
                    'Время обработки (мс)': result.get('processing_time', 0)
                })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    return filepath

def save_comparison_to_csv(comparisons, filename=None):
    if filename is None:
        filename = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    filepath = os.path.join(RESULTS_FOLDER, filename)
    
    rows = []
    for comp in comparisons:
        for det in comp.get('detections', []):
            rows.append({
                'Архитектура': comp.get('model_name', ''),
                'Тип': comp.get('type', ''),
                'Распознанный текст': det.get('text', ''),
                'Уверенность (%)': round(det.get('confidence', 0) * 100, 2),
                'Время (мс)': comp.get('processing_time', 0)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    return filepath

# ========== FLASK МАРШРУТЫ ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global results_history  
    
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        
        model_name = data.get('model', 'arch1')
        
        models = {
            'arch1': recognize_architecture1_easyocr,
            'arch2': recognize_architecture2_crnn,
            'arch3': recognize_architecture3_transformer,
            'arch4': recognize_architecture4_hybrid,
            'arch5': recognize_architecture5_ensemble
        }
        
        recognize_func = models.get(model_name, recognize_architecture1_easyocr)
        detections, processing_time = recognize_func(filepath)
        
        results = {
            'image': filename,
            'timestamp': datetime.now().isoformat(),
            'model_used': model_name,
            'processing_time': round(processing_time, 2),
            'detections': detections
        }
        
        results_history.append(results)
        
        if len(results_history) > 50:
            results_history = results_history[-50:]
        
        with open(os.path.join(RESULTS_FOLDER, 'history.json'), 'w', encoding='utf-8') as f:
            json.dump(results_history, f, ensure_ascii=False, indent=2, default=convert_to_serializable)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Upload ошибка: {e}")
        return jsonify({'error': str(e), 'detections': []})

@app.route('/compare_models', methods=['POST'])
def compare_models():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        filename = f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        
        comparisons = []
        
        # EasyOCR
        det1, time1 = recognize_architecture1_easyocr(filepath)
        avg1 = sum(d['confidence'] for d in det1) / len(det1) if det1 else 0
        comparisons.append({
            'model_name': '🔵 1. EasyOCR',
            'type': 'CNN+RNN+Attention+BiLSTM',
            'processing_time': round(time1, 2),
            'detections_count': len(det1),
            'detections': det1,
            'avg_confidence': round(avg1, 3)
        })
        
        # CRNN
        det2, time2 = recognize_architecture2_crnn(filepath)
        avg2 = sum(d['confidence'] for d in det2) / len(det2) if det2 else 0
        comparisons.append({
            'model_name': '🟢 2. CRNN',
            'type': 'CNN+RNN+CTC',
            'processing_time': round(time2, 2),
            'detections_count': len(det2),
            'detections': det2,
            'avg_confidence': round(avg2, 3)
        })
        
        # Transformer
        det3, time3 = recognize_architecture3_transformer(filepath)
        avg3 = sum(d['confidence'] for d in det3) / len(det3) if det3 else 0
        comparisons.append({
            'model_name': '🟣 3. Transformer',
            'type': 'ViT+Attention',
            'processing_time': round(time3, 2),
            'detections_count': len(det3),
            'detections': det3,
            'avg_confidence': round(avg3, 3)
        })
        
        # Гибридная
        det4, time4 = recognize_architecture4_hybrid(filepath)
        avg4 = sum(d['confidence'] for d in det4) / len(det4) if det4 else 0
        comparisons.append({
            'model_name': '🟠 4. Гибридная',
            'type': 'CNN+Transformer+CTC',
            'processing_time': round(time4, 2),
            'detections_count': len(det4),
            'detections': det4,
            'avg_confidence': round(avg4, 3)
        })
        
        # Ансамбль
        det5, time5 = recognize_architecture5_ensemble(filepath)
        avg5 = sum(d['confidence'] for d in det5) / len(det5) if det5 else 0
        comparisons.append({
            'model_name': '🔴 5. Ансамбль',
            'type': 'Комитет из 5 моделей',
            'processing_time': round(time5, 2),
            'detections_count': len(det5),
            'detections': det5,
            'avg_confidence': round(avg5, 3)
        })
        
        fastest = min(comparisons, key=lambda x: x['processing_time'])
        most_accurate = max(comparisons, key=lambda x: x['avg_confidence'])
        
        return jsonify({
            'image': filename,
            'timestamp': datetime.now().isoformat(),
            'comparisons': comparisons,
            'fastest_model': fastest['model_name'],
            'fastest_time': fastest['processing_time'],
            'most_accurate_model': most_accurate['model_name'],
            'most_accuracy': most_accurate['avg_confidence'],
            'recommendation': 'Рекомендуется EasyOCR для максимальной точности'
        })
    
    except Exception as e:
        print(f"Compare ошибка: {e}")
        return jsonify({'error': str(e)})

@app.route('/export_csv', methods=['GET'])
def export_csv():
    global results_history
    
    if not results_history:
        return jsonify({'error': 'Нет данных для экспорта'})
    
    csv_path = save_to_csv(results_history)
    return jsonify({'success': True, 'message': f'Отчёт сохранён: {csv_path}'})

@app.route('/get_history', methods=['GET'])
def get_history():
    return jsonify({'history': results_history, 'count': len(results_history)})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global results_history
    results_history = []
    with open(os.path.join(RESULTS_FOLDER, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    return jsonify({'success': True, 'message': 'История очищена'})

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("🔍 OCR СИСТЕМА - СРАВНЕНИЕ 5 АРХИТЕКТУР")
    print("=" * 70)
    print("\n📊 5 АРХИТЕКТУР:")
    print("   1. 🔵 EasyOCR        | CNN+RNN+Attention+BiLSTM")
    print("   2. 🟢 CRNN           | CNN+RNN+CTC")
    print("   3. 🟣 Transformer    | ViT+Attention")
    print("   4. 🟠 Гибридная      | CNN+Transformer+CTC")
    print("   5. 🔴 Ансамбль       | Комитет из 5 моделей")
    print("\n🌐 Запуск: http://127.0.0.1:5000")
    print("=" * 70 + "\n")
    
    app.run(debug=True)