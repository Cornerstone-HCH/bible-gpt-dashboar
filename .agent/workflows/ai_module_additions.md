# AI 모듈 추가 구현 (감정 인식 + 46개 상징 체계)

> 정본 9장 기준 누락 모듈 보완

---

## 🎭 **모듈 2.5: 감정 인식 (MobileNetV2)** ⭐ 신규 추가

### **목적**
- 얼굴 표정에서 7개 기본 감정 추출
- EmotionVector 생성
- S2 (감정 앵커) 점수 계산에 사용

### **기술 스택**
```python
# requirements.txt 추가
tensorflow>=2.13.0  # 또는 tensorflow-lite
opencv-python>=4.8.0
numpy>=1.24.0
```

### **구현 코드**
```python
# modules/emotion_detector.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionDetector:
    def __init__(self, model_path='models/emotion_mobilenetv2.h5'):
        """
        MobileNetV2 기반 감정 인식 모듈
        
        7개 기본 감정:
        - 행복 (happy)
        - 놀람 (surprise)
        - 슬픔 (sad)
        - 분노 (angry)
        - 혐오 (disgust)
        - 두려움 (fear)
        - 중립 (neutral)
        """
        self.model = load_model(model_path)
        self.emotions = ['happy', 'surprise', 'sad', 'angry', 'disgust', 'fear', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect(self, image_path):
        """
        이미지에서 감정 인식
        
        Returns:
            Dict: {
                'emotion_probs': {
                    'happy': 0.65,
                    'neutral': 0.20,
                    ...
                },
                'primary_label': 'happy',
                'intensity': 0.65,
                'source': 'face_model',
                'faces_detected': 1
            }
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(64, 64)
        )
        
        if len(faces) == 0:
            return {
                'emotion_probs': {e: 0.0 for e in self.emotions},
                'primary_label': 'neutral',
                'intensity': 0.0,
                'source': 'face_model',
                'faces_detected': 0
            }
        
        # 첫 번째 얼굴만 사용 (가장 큰 얼굴)
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # 얼굴 크롭 및 전처리
        face_crop = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_crop, (64, 64))
        face_normalized = face_resized / 255.0
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = np.expand_dims(face_input, axis=-1)  # (1, 64, 64, 1)
        
        # 감정 예측
        predictions = self.model.predict(face_input)[0]
        
        # 결과 구성
        emotion_probs = {
            emotion: float(prob) 
            for emotion, prob in zip(self.emotions, predictions)
        }
        
        primary_idx = np.argmax(predictions)
        primary_label = self.emotions[primary_idx]
        intensity = float(predictions[primary_idx])
        
        return {
            'emotion_probs': emotion_probs,
            'primary_label': primary_label,
            'intensity': intensity,
            'source': 'face_model',
            'faces_detected': len(faces)
        }
    
    def to_emotion_vector(self, detection_result):
        """
        EmotionVector 엔티티 생성
        
        Returns:
            Dict: {
                'emotion_probs': {...},
                'primary_label': 'happy',
                'intensity': 0.65,
                'source': 'face_model'
            }
        """
        return {
            'emotion_probs': detection_result['emotion_probs'],
            'primary_label': detection_result['primary_label'],
            'intensity': detection_result['intensity'],
            'source': detection_result['source']
        }
```

### **모델 학습 (선택)**
```python
# scripts/train_emotion_model.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

def build_emotion_model():
    """
    MobileNetV2 기반 감정 인식 모델 구축
    """
    base_model = MobileNetV2(
        input_shape=(64, 64, 1),
        include_top=False,
        weights=None  # 처음부터 학습
    )
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(7, activation='softmax')(x)  # 7개 감정
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# 학습 데이터: FER2013 또는 AffectNet 사용
# model = build_emotion_model()
# model.fit(train_data, epochs=50, validation_data=val_data)
# model.save('models/emotion_mobilenetv2.h5')
```

---

## 🎯 **모듈 5: P1 상징 매핑 (46개 상징 체계)** ⭐ 신규 추가

### **목적**
- Perception 결과 → 46개 성경적 상징으로 변환
- 5개 코어 군 구조 적용
- P2 주제 매핑의 입력 생성

### **46개 상징 체계 (정본 9장 기준)**

#### **코어 군 1: 자연 (10개)**
1. 산 - 피난/견고/시련
2. 바다 - 혼돈/깊음/광활
3. 강·샘 - 생명/공급
4. 광야 - 시험/의존
5. 숲·나무 - 생장/의지
6. 꽃·풀 - 덧없음/은혜
7. 해·빛 - 창조/인도/영광
8. 달·별 - 시간/질서
9. 비·눈 - 은혜/정결
10. 폭풍·바람 - 위기/전환

#### **코어 군 2: 생명 (9개)**
11. 씨앗·열매 - 시작/결과
12. 포도나무·가지 - 연합/의존
13. 양 - 보호/목자
14. 사자 - 권세/위엄
15. 비둘기 - 평화/성령
16. 물고기 - 공급/증언
17. 어린아이 - 보호/양육
18. 노인 - 지혜/기억
19. 인간 표정 - 정서 단서

#### **코어 군 3: 인공물 (10개)**
20. 집·가정 - 쉼/돌봄
21. 도시·거리 - 세상/소명
22. 길·다리 - 인도/결단
23. 학교 - 배움/훈련
24. 병원 - 치유/위로
25. 시장·상점 - 생계/공정
26. 법정·재판 - 정의/책임
27. 공장·도구·차량 - 노동/생산
28. 교회·예배당 - 예배/연합
29. 무덤·기념 - 유한성/소망

#### **코어 군 4: 인간 활동 (10개)**
30. 노동·수고 - 책임/청지기
31. 농사·수확 - 인내/열매
32. 공부·독서 - 배움/분별
33. 음악·예술 - 찬양/표현
34. 스포츠·경기 - 절제/경주
35. 결혼·잔치 - 언약/기쁨
36. 육아·돌봄 - 보호/양육
37. 여행·순례 - 보냄/인도
38. 전쟁·무기 - 갈등/분별
39. 정치·회의 - 공공 판단

#### **코어 군 5: 공간·장면 (7개)**
40. 실내·집중 - 묵상/기도
41. 실외·광장 - 개방/선포
42. 군중 - 긍휼/연대
43. 고립·홀로 - 성찰/임재
44. 골목·어둑함 - 경계/분별
45. 병실·대기실 - 연약함/위로
46. 성지·랜드마크 - 기억/예배

### **구현 코드**
```python
# modules/symbol_mapper.py
import pandas as pd

class SymbolMapper:
    def __init__(self, symbol_def_path='data/symbol_definitions.csv'):
        """
        P1 상징 매핑 모듈
        46개 상징 체계 적용
        """
        self.symbol_defs = pd.read_csv(symbol_def_path)
        self.load_mapping_rules()
    
    def load_mapping_rules(self):
        """
        객체/장면/감정 → 상징 매핑 규칙 로드
        """
        self.object_to_symbol = {
            # COCO 객체 → 상징
            'person': ['사람', '인간 표정'],
            'bed': ['침대', '병실·대기실'],
            'mountain': ['산'],
            'sea': ['바다'],
            'church': ['교회·예배당'],
            'hospital': ['병원', '병실·대기실'],
            'book': ['공부·독서'],
            'car': ['공장·도구·차량'],
            # ... (120개 피사체 매핑)
        }
        
        self.scene_to_symbol = {
            # 장면 → 상징
            '실내': ['실내·집중'],
            '실외': ['실외·광장'],
            '교회': ['교회·예배당'],
            '병원': ['병원', '병실·대기실'],
            '밤': ['달·별', '골목·어둑함'],
            # ...
        }
        
        self.emotion_to_symbol = {
            # 감정 → 상징
            'sad': ['눈물', '고립·홀로'],
            'happy': ['결혼·잔치', '음악·예술'],
            'fear': ['폭풍·바람', '골목·어둑함'],
            'neutral': ['실내·집중'],
            # ...
        }
    
    def map(self, perception_result):
        """
        Perception 결과 → 상징 리스트
        
        Args:
            perception_result: {
                'objects': [...],  # ObjectDetector 결과
                'scene': {...},    # SceneAnalyzer 결과
                'emotion': {...}   # EmotionDetector 결과
            }
        
        Returns:
            List[Dict]: [
                {
                    'symbol': '침대',
                    'confidence': 0.85,
                    'source': 'object',
                    'core_group': '인공물'
                },
                ...
            ]
        """
        symbols = []
        
        # 1. 객체 → 상징
        for obj in perception_result.get('objects', []):
            coco_class = obj['coco_class']
            if coco_class in self.object_to_symbol:
                for symbol in self.object_to_symbol[coco_class]:
                    symbols.append({
                        'symbol': symbol,
                        'confidence': obj['confidence'],
                        'source': 'object',
                        'core_group': self._get_core_group(symbol)
                    })
        
        # 2. 장면 → 상징
        scene = perception_result.get('scene', {})
        location = scene.get('location', ('', 0))[0]
        if location in self.scene_to_symbol:
            for symbol in self.scene_to_symbol[location]:
                symbols.append({
                    'symbol': symbol,
                    'confidence': scene['location'][1],
                    'source': 'scene',
                    'core_group': self._get_core_group(symbol)
                })
        
        # 3. 감정 → 상징
        emotion = perception_result.get('emotion', {})
        primary_emotion = emotion.get('primary_label', 'neutral')
        if primary_emotion in self.emotion_to_symbol:
            for symbol in self.emotion_to_symbol[primary_emotion]:
                symbols.append({
                    'symbol': symbol,
                    'confidence': emotion.get('intensity', 0.5),
                    'source': 'emotion',
                    'core_group': self._get_core_group(symbol)
                })
        
        # 중복 제거 및 점수 합산
        symbol_scores = {}
        for s in symbols:
            key = s['symbol']
            if key not in symbol_scores:
                symbol_scores[key] = s
            else:
                # 점수 합산
                symbol_scores[key]['confidence'] += s['confidence']
        
        # 정렬 및 반환
        sorted_symbols = sorted(
            symbol_scores.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return sorted_symbols[:10]  # Top 10
    
    def _get_core_group(self, symbol):
        """상징 → 코어 군 매핑"""
        symbol_info = self.symbol_defs[self.symbol_defs['symbol'] == symbol]
        if not symbol_info.empty:
            return symbol_info.iloc[0]['core_group']
        return 'unknown'
```

### **데이터 준비**
```csv
# data/symbol_definitions.csv
symbol_id,symbol,core_group,meaning,primary_themes
1,산,자연,피난/견고/시련,"보호·인도,인내·시험"
2,바다,자연,혼돈/깊음/광활,"창조·섭리,소망·부활"
3,강·샘,자연,생명/공급,"치유·회복,은혜"
...
20,집·가정,인공물,쉼/돌봄,"가정·양육,공동체·연합"
...
40,실내·집중,공간·장면,묵상/기도,"기도,말씀·진리"
...
```

---

## 📊 **S1-S4 점수 계산 로직** ⭐ 신규 추가

### **목적**
- 정본 9장의 S1-S4 점수 체계 구현
- P2 주제 매핑에 사용

### **구현 코드**
```python
# modules/score_calculator.py
class ScoreCalculator:
    def __init__(self, weights=None):
        """
        S1-S4 점수 계산 모듈
        
        Args:
            weights: {
                'w1': 0.4,  # S1 (객체/장면/상징)
                'w2': 0.2,  # S2 (감정)
                'w3': 0.2,  # S3 (행위)
                'w4': 0.2   # S4 (교리)
            }
        """
        self.weights = weights or {
            'w1': 0.4,
            'w2': 0.2,
            'w3': 0.2,
            'w4': 0.2
        }
    
    def calculate_s1(self, symbols):
        """
        S1: Object / Scene Anchors
        상징 기반 점수
        """
        if not symbols:
            return 0.0
        
        # 상위 3개 상징의 평균 신뢰도
        top3 = symbols[:3]
        return sum(s['confidence'] for s in top3) / len(top3)
    
    def calculate_s2(self, emotion_vector):
        """
        S2: Emotion Anchors
        감정 기반 점수
        """
        if not emotion_vector:
            return 0.0
        
        return emotion_vector.get('intensity', 0.0)
    
    def calculate_s3(self, symbols, scene):
        """
        S3: Action Anchors
        행위/동작 기반 점수
        
        (현재는 간단히 구현, 추후 행위 인식 모델 추가 가능)
        """
        # 행위 관련 상징 체크
        action_symbols = [
            '노동·수고', '농사·수확', '공부·독서',
            '음악·예술', '스포츠·경기', '여행·순례'
        ]
        
        action_score = 0.0
        for s in symbols:
            if s['symbol'] in action_symbols:
                action_score += s['confidence']
        
        return min(action_score, 1.0)
    
    def calculate_s4(self, symbols, theological_rules):
        """
        S4: Theology Anchors
        교리/신학 기반 점수
        """
        # 신학적으로 중요한 상징 체크
        theological_symbols = [
            '교회·예배당', '성지·랜드마크',
            '공부·독서', '기도'
        ]
        
        theo_score = 0.0
        for s in symbols:
            if s['symbol'] in theological_symbols:
                theo_score += s['confidence']
        
        return min(theo_score, 1.0)
    
    def calculate_total(self, s1, s2, s3, s4):
        """
        score_total = (w1 × S1) + (w2 × S2) + (w3 × S3) + (w4 × S4)
        """
        return (
            self.weights['w1'] * s1 +
            self.weights['w2'] * s2 +
            self.weights['w3'] * s3 +
            self.weights['w4'] * s4
        )
    
    def calculate_priority(self, score_total, bonus_anchor=0.1, normalize_divisor=1.0):
        """
        priority_score = (score_total + bonus) / normalize_divisor
        """
        bonus = bonus_anchor if score_total > 0 else 0
        return (score_total + bonus) / normalize_divisor
```

---

## 🔄 **통합 파이프라인 업데이트**

```python
# modules/image_pipeline_v2.py
from .preprocessor import ImagePreprocessor
from .safety_router import SafetyRouter
from .object_detection import ObjectDetector
from .scene_analyzer import SceneAnalyzer
from .emotion_detector import EmotionDetector
from .text_extractor import TextExtractor
from .symbol_mapper import SymbolMapper
from .score_calculator import ScoreCalculator

class ImagePipelineV2:
    def __init__(self):
        # 전처리
        self.preprocessor = ImagePreprocessor()
        self.safety_router = SafetyRouter()
        
        # Perception
        self.object_detector = ObjectDetector()
        self.scene_analyzer = SceneAnalyzer()
        self.emotion_detector = EmotionDetector()
        self.text_extractor = TextExtractor()
        
        # P1 상징 매핑
        self.symbol_mapper = SymbolMapper()
        
        # 점수 계산
        self.score_calculator = ScoreCalculator()
    
    def process(self, image_path):
        """
        전체 파이프라인 실행
        
        Returns:
            Dict: {
                'symbols': [...],
                'scores': {
                    's1': 0.75,
                    's2': 0.60,
                    's3': 0.40,
                    's4': 0.50,
                    'total': 0.58,
                    'priority': 0.68
                },
                'safety_flags': [...],
                'emotion_vector': {...}
            }
        """
        # 0. 전처리
        preprocessed = self.preprocessor.preprocess(image_path)
        
        # 1. Perception
        objects = self.object_detector.detect(image_path)
        scene = self.scene_analyzer.analyze(image_path)
        emotion = self.emotion_detector.detect(image_path)
        texts = self.text_extractor.extract(image_path)
        
        # 2. 민감 라우팅
        ocr_text = ' '.join([t['text'] for t in texts])
        safety_flags = self.safety_router.route(preprocessed['image'], ocr_text)
        
        # 3. P1 상징 매핑
        perception_result = {
            'objects': objects,
            'scene': scene,
            'emotion': emotion
        }
        symbols = self.symbol_mapper.map(perception_result)
        
        # 4. S1-S4 점수 계산
        s1 = self.score_calculator.calculate_s1(symbols)
        s2 = self.score_calculator.calculate_s2(emotion)
        s3 = self.score_calculator.calculate_s3(symbols, scene)
        s4 = self.score_calculator.calculate_s4(symbols, {})
        
        score_total = self.score_calculator.calculate_total(s1, s2, s3, s4)
        priority_score = self.score_calculator.calculate_priority(score_total)
        
        return {
            'symbols': symbols,
            'scores': {
                's1': s1,
                's2': s2,
                's3': s3,
                's4': s4,
                'total': score_total,
                'priority': priority_score
            },
            'safety_flags': safety_flags,
            'emotion_vector': emotion
        }
```

---

## 📝 **다음 단계**

1. **감정 인식 모델 학습** (FER2013 데이터셋)
2. **symbol_definitions.csv 완성** (46개 상징 전체)
3. **S3 행위 인식** 로직 개선
4. **S4 교리 앵커** 정의 및 매칭
5. **통합 테스트**
