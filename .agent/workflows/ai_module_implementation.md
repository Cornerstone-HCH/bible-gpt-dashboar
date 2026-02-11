# AI 모듈 통합 구현 계획 (v2.0 - 정본 9장 기준)

> Phase 3.2: 이미지 → 자동 상징 추출 파이프라인  
> **기준 문서**: old_AI_setting.md (정본 9장 AI 로직)

---

## 📋 **전체 아키텍처 (정본 기준)**

```
이미지 입력 + EXIF + 컨텍스트
    ↓
┌─────────────────────────────────────┐
│  전처리 & 민감 라우팅                │
│  - EXIF 정규화                       │
│  - 리사이즈 + 레터박스               │
│  - safety_flags 생성                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Perception (지각 레이어)            │
├─────────────────────────────────────┤
│  ① 객체 인식 (YOLO)                 │
│  ② 장면 분류 (CLIP)                 │
│  ③ 감정 인식 (MobileNetV2) ⭐ 추가   │
│  ④ OCR (EasyOCR)                    │
│  ⑤ 랜드마크 인식 (선택)              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  P1: 상징 매핑 (46개 상징 체계) ⭐   │
│  - 5개 코어 군 구조                  │
│  - 객체/장면/감정 → 상징 변환        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  P2: 주제 매핑 (120×24 매트릭스)    │
│  - 상징 → 24개 주제                 │
│  - S1-S4 점수 계산 ⭐                │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  P3: 컨텍스트 보정                   │
│  - 시간/장소/시각특징 보정           │
│  - 신학적 안전 규칙 적용             │
└─────────────────────────────────────┘
    ↓
벡터 검색 → 재랭킹 → Top-1/2 구절
```

---

## 🔧 **모듈 0: 전처리 & 민감 라우팅** ⭐ 신규 추가

### **목적**
- Perception 실행 전 입력 표준화
- 민감 상황 조기 탐지 및 안전 플로우 라우팅

### **기술 스택**
```python
# requirements.txt 추가
pillow>=10.0.0
exifread>=3.0.0
```

### **구현 코드**
```python
# modules/preprocessor.py
from PIL import Image
import exifread
from datetime import datetime

class ImagePreprocessor:
    def __init__(self, target_size=2048):
        """
        이미지 전처리 및 EXIF 정규화
        
        Args:
            target_size: 긴 변 기준 리사이즈 크기
        """
        self.target_size = target_size
    
    def preprocess(self, image_path):
        """
        전처리 파이프라인
        
        Returns:
            Dict: {
                'image': PIL.Image,
                'exif_time': datetime,
                'orientation': int,
                'metadata': dict
            }
        """
        # 1. EXIF 읽기
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
        
        # 2. 이미지 로드 및 회전 보정
        image = Image.open(image_path)
        orientation = self._get_orientation(tags)
        image = self._apply_orientation(image, orientation)
        
        # 3. 리사이즈 + 레터박스
        image = self._resize_letterbox(image)
        
        # 4. 촬영 시각 추출
        exif_time = self._extract_datetime(tags)
        
        return {
            'image': image,
            'exif_time': exif_time,
            'orientation': orientation,
            'metadata': {
                'exif_present': len(tags) > 0,
                'time_source': 'exif' if exif_time else 'upload'
            }
        }
    
    def _resize_letterbox(self, image):
        """긴 변 기준 리사이즈 + 레터박스 패딩"""
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 리사이즈
        image = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 레터박스 (정사각형으로)
        new_image = Image.new('RGB', (self.target_size, self.target_size), (0, 0, 0))
        paste_x = (self.target_size - new_w) // 2
        paste_y = (self.target_size - new_h) // 2
        new_image.paste(image, (paste_x, paste_y))
        
        return new_image
    
    def _get_orientation(self, tags):
        """EXIF Orientation 추출"""
        if 'Image Orientation' in tags:
            return int(str(tags['Image Orientation']))
        return 1
    
    def _apply_orientation(self, image, orientation):
        """Orientation 기반 회전 보정"""
        if orientation == 3:
            return image.rotate(180, expand=True)
        elif orientation == 6:
            return image.rotate(270, expand=True)
        elif orientation == 8:
            return image.rotate(90, expand=True)
        return image
    
    def _extract_datetime(self, tags):
        """촬영 시각 추출"""
        if 'EXIF DateTimeOriginal' in tags:
            dt_str = str(tags['EXIF DateTimeOriginal'])
            return datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
        return None


# modules/safety_router.py
class SafetyRouter:
    def __init__(self):
        """
        민감 라우팅 모듈
        Perception 이전에 민감 신호 탐지
        """
        self.sensitive_keywords = {
            'medical': ['병원', 'ER', '응급실', '병실', 'ICU'],
            'funeral': ['장례', '조문', '묘지', '관', '헌화'],
            'child': ['유아', '어린이', '유치원', '초등학교'],
            'politics': ['시위', '집회', '정당', '선거']
        }
    
    def route(self, image, ocr_text=''):
        """
        민감 라우팅 실행
        
        Returns:
            List[str]: safety_flags
        """
        flags = []
        
        # OCR 기반 민감 키워드 탐지
        for category, keywords in self.sensitive_keywords.items():
            if any(kw in ocr_text for kw in keywords):
                flags.append(f'{category}_detected')
        
        # 이미지 기반 탐지는 Perception 이후 보완
        
        return flags
```

### **데이터 준비**
- 없음 (전처리 로직만 필요)

---

## 🔧 **모듈 1: 객체 인식 (YOLO/COCO)**

### **목적**
- 이미지 내 객체 감지 (사람, 침대, 십자가, 책 등)
- detection_labels_map.csv와 매핑

### **기술 스택**
```python
# requirements.txt 추가
ultralytics>=8.0.0  # YOLOv8
opencv-python>=4.8.0
torch>=2.0.0
```

### **구현 코드 (예상)**
```python
# modules/object_detection.py
from ultralytics import YOLO
import cv2
import pandas as pd

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        YOLOv8 객체 인식 모듈
        
        Args:
            model_path: YOLO 모델 경로 (yolov8n.pt = 경량 모델)
            confidence_threshold: 신뢰도 임계값 (0.5 = 50%)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # detection_labels_map.csv 로드
        self.label_map = pd.read_csv('data/detection_labels_map.csv')
        
    def detect(self, image_path):
        """
        이미지에서 객체 감지
        
        Returns:
            List[Dict]: [
                {
                    'coco_class': 'person',
                    'bible_symbol': '사람',
                    'confidence': 0.92,
                    'bbox': [x1, y1, x2, y2]
                },
                ...
            ]
        """
        # YOLO 추론
        results = self.model(image_path)
        
        detected_objects = []
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                # COCO 클래스 → 성경 상징 매핑
                coco_class = result.names[int(box.cls[0])]
                bible_symbol = self._map_to_bible_symbol(coco_class)
                
                if bible_symbol:
                    detected_objects.append({
                        'coco_class': coco_class,
                        'bible_symbol': bible_symbol,
                        'confidence': conf,
                        'bbox': box.xyxy[0].tolist()
                    })
        
        return detected_objects
    
    def _map_to_bible_symbol(self, coco_class):
        """COCO 클래스 → 성경 상징 매핑"""
        mapping = self.label_map[self.label_map['coco_class'] == coco_class]
        if not mapping.empty:
            return mapping.iloc[0]['bible_symbol']
        return None
```

### **데이터 준비**
```csv
# data/detection_labels_map.csv (생성 필요)
coco_class,bible_symbol,priority
person,사람,1.0
bed,침대,1.0
book,책,0.9
cross,십자가,1.0
cup,잔,0.8
...
```

---

## 🎨 **모듈 2: CLIP 장면/감정 분석**

### **목적**
- 장면 분류 (실내/실외, 밤/낮)
- 감정/분위기 분석 (평화로운, 슬픈, 기쁜 등)
- 상징 후보 생성

### **기술 스택**
```python
# requirements.txt 추가
transformers>=4.30.0
torch>=2.0.0
pillow>=10.0.0
```

### **구현 코드 (예상)**
```python
# modules/scene_analyzer.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class SceneAnalyzer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        CLIP 기반 장면 분석 모듈
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # 분석 카테고리
        self.scene_categories = {
            'location': ['실내', '실외', '교회', '병원', '자연', '도시'],
            'time': ['아침', '낮', '저녁', '밤', '자정'],
            'mood': ['평화로운', '슬픈', '기쁜', '경건한', '고요한', '어두운']
        }
    
    def analyze(self, image_path):
        """
        장면 분석
        
        Returns:
            Dict: {
                'location': ('병원', 0.85),
                'time': ('밤', 0.78),
                'mood': ('고요한', 0.72),
                'suggested_symbols': ['침대', '어둠', '평강']
            }
        """
        image = Image.open(image_path)
        
        results = {}
        for category, labels in self.scene_categories.items():
            # CLIP으로 분류
            inputs = self.processor(
                text=labels,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
            
            # 가장 높은 확률의 레이블 선택
            max_idx = probs.argmax().item()
            results[category] = (labels[max_idx], float(probs[0][max_idx]))
        
        # 상징 제안
        results['suggested_symbols'] = self._suggest_symbols(results)
        
        return results
    
    def _suggest_symbols(self, scene_results):
        """장면 분석 결과 → 상징 제안"""
        symbols = []
        
        # 규칙 기반 매핑
        location, _ = scene_results['location']
        time, _ = scene_results['time']
        mood, _ = scene_results['mood']
        
        if location == '병원':
            symbols.extend(['침대', '치유'])
        if time in ['밤', '자정']:
            symbols.extend(['어둠', '별'])
        if mood in ['평화로운', '고요한']:
            symbols.extend(['평강', '안식'])
        
        return list(set(symbols))  # 중복 제거
```

---

## 📝 **모듈 3: EasyOCR 텍스트 추출**

### **목적**
- 이미지 내 한글/영어 텍스트 추출
- 성경 구절 감지 (선택적)

### **기술 스택**
```python
# requirements.txt 추가
easyocr>=1.7.0
```

### **구현 코드 (예상)**
```python
# modules/text_extractor.py
import easyocr

class TextExtractor:
    def __init__(self, languages=['ko', 'en']):
        """
        EasyOCR 텍스트 추출 모듈
        """
        self.reader = easyocr.Reader(languages)
    
    def extract(self, image_path):
        """
        텍스트 추출
        
        Returns:
            List[Dict]: [
                {
                    'text': '요한복음 3:16',
                    'confidence': 0.95,
                    'bbox': [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                },
                ...
            ]
        """
        results = self.reader.readtext(image_path)
        
        extracted_texts = []
        for bbox, text, conf in results:
            extracted_texts.append({
                'text': text,
                'confidence': conf,
                'bbox': bbox
            })
        
        return extracted_texts
    
    def detect_bible_verse(self, texts):
        """성경 구절 패턴 감지"""
        import re
        
        bible_pattern = r'([가-힣]+)\s*(\d+):(\d+)'
        
        for item in texts:
            match = re.search(bible_pattern, item['text'])
            if match:
                return {
                    'book': match.group(1),
                    'chapter': int(match.group(2)),
                    'verse': int(match.group(3)),
                    'confidence': item['confidence']
                }
        
        return None
```

---

## 🔗 **모듈 4: 멀티모달 통합**

### **목적**
- 3개 모듈의 결과를 통합
- 가중치 적용하여 최종 상징 리스트 생성

### **구현 코드 (예상)**
```python
# modules/multimodal_integrator.py
from typing import List, Dict
from collections import Counter

class MultimodalIntegrator:
    def __init__(self, weights=None):
        """
        멀티모달 신호 통합
        
        Args:
            weights: {'object': 0.5, 'scene': 0.3, 'text': 0.2}
        """
        self.weights = weights or {
            'object': 0.5,  # 객체 인식 가중치
            'scene': 0.3,   # 장면 분석 가중치
            'text': 0.2     # 텍스트 추출 가중치
        }
    
    def integrate(self, object_results, scene_results, text_results):
        """
        통합 및 최종 상징 리스트 생성
        
        Returns:
            List[str]: ['침대', '어둠', '평강', '치유', '기도']
        """
        symbol_scores = {}
        
        # 1. 객체 인식 결과
        for obj in object_results:
            symbol = obj['bible_symbol']
            score = obj['confidence'] * self.weights['object']
            symbol_scores[symbol] = symbol_scores.get(symbol, 0) + score
        
        # 2. 장면 분석 결과
        for symbol in scene_results.get('suggested_symbols', []):
            score = 0.7 * self.weights['scene']  # 장면 분석은 고정 신뢰도
            symbol_scores[symbol] = symbol_scores.get(symbol, 0) + score
        
        # 3. 텍스트 추출 결과 (선택적)
        # 텍스트에서 추출된 키워드를 상징으로 변환
        
        # 점수순 정렬
        sorted_symbols = sorted(
            symbol_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Top 5 반환
        return [symbol for symbol, score in sorted_symbols[:5]]
```

---

## 🚀 **메인 파이프라인**

```python
# modules/image_pipeline.py
from .object_detection import ObjectDetector
from .scene_analyzer import SceneAnalyzer
from .text_extractor import TextExtractor
from .multimodal_integrator import MultimodalIntegrator

class ImagePipeline:
    def __init__(self):
        self.object_detector = ObjectDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.text_extractor = TextExtractor()
        self.integrator = MultimodalIntegrator()
    
    def process(self, image_path):
        """
        이미지 → 상징 리스트 추출
        
        Returns:
            List[str]: ['침대', '어둠', '평강', '치유', '기도']
        """
        # 병렬 처리 (선택적)
        object_results = self.object_detector.detect(image_path)
        scene_results = self.scene_analyzer.analyze(image_path)
        text_results = self.text_extractor.extract(image_path)
        
        # 통합
        symbols = self.integrator.integrate(
            object_results,
            scene_results,
            text_results
        )
        
        return symbols
```

---

## 📦 **디렉토리 구조**

```
ANTIGRAVITY/
├── modules/
│   ├── __init__.py
│   ├── object_detection.py      # YOLO/COCO
│   ├── scene_analyzer.py        # CLIP
│   ├── text_extractor.py        # EasyOCR
│   ├── multimodal_integrator.py # 통합
│   └── image_pipeline.py        # 메인 파이프라인
├── models/                      # 다운로드된 모델 저장
│   ├── yolov8n.pt
│   └── clip-vit-base-patch32/
├── data/
│   ├── detection_labels_map.csv # COCO → 성경 상징 매핑
│   └── ...
└── bible_gpt_dashboard.py       # 대시보드 통합
```

---

## ⏱️ **구현 일정**

### **Day 1 (4-5시간)**
- [ ] 모듈 디렉토리 구조 생성
- [ ] ObjectDetector 구현 (YOLO)
- [ ] detection_labels_map.csv 생성
- [ ] 테스트 이미지로 검증

### **Day 2 (3-4시간)**
- [ ] SceneAnalyzer 구현 (CLIP)
- [ ] TextExtractor 구현 (EasyOCR)
- [ ] MultimodalIntegrator 구현
- [ ] ImagePipeline 통합

### **Day 3 (2-3시간)**
- [ ] 대시보드 통합
- [ ] 성능 최적화
- [ ] End-to-End 테스트

---

## 🎯 **성능 목표**

- **정확도**: Top-5 상징 중 1개 이상 관련성 있음 (80% 이상)
- **속도**: p95 < 800ms (정본 MD 요구사항)
- **경량화**: 모바일 디바이스에서도 실행 가능 (ONNX 변환)

---

## 📝 **다음 단계**

1. **requirements.txt 업데이트**
2. **modules/ 디렉토리 생성**
3. **ObjectDetector부터 순차 구현**
4. **각 모듈별 단위 테스트**
5. **대시보드 통합**
