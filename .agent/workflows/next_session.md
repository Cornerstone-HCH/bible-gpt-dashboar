# 다음 세션 작업 계획

> 생성일: 2026-02-11 22:58  
> Phase 3.3: AI 모듈 통합 (ObjectDetector, SceneAnalyzer, EmotionDetector)

---

## 🎯 **다음 세션 목표**

**Phase 3.3 AI 모듈 통합 시작**
- ObjectDetector (YOLO) 구현
- SceneAnalyzer (CLIP) 구현
- EmotionDetector (MobileNetV2) 구현

**예상 소요 시간**: 6-9시간 (2-3일)

---

## ✅ **현재까지 완료된 것**

### **Phase 1-2: 데이터 품질 (100%)**
- ✅ 기본 검증 완료
- ✅ 신학적 안전 규칙 데이터화
- ✅ SIM-01 시나리오 테스트 통과

### **Phase 3.0-3.2: 데이터 및 기본 모듈 (40%)**
- ✅ symbol_definitions.csv (46개 상징)
- ✅ detection_labels_map.csv (COCO 매핑)
- ✅ emotion_scripture_map.csv (감정 매핑)
- ✅ SymbolMapper 모듈 (P1 상징 매핑)
- ✅ ThemeMapper 모듈 (P2 주제 매핑)
- ✅ 웹 테스트 대시보드 (http://localhost:8508)

---

## 🚀 **다음 세션 작업 순서**

### **1단계: ObjectDetector 구현 (2-3시간)**

#### **준비 사항:**
```bash
# YOLOv8 설치
pip install ultralytics

# 모델 다운로드 (자동)
# yolov8n.pt (경량 모델, ~6MB)
```

#### **구현 파일:**
- `modules/object_detector.py`

#### **핵심 기능:**
```python
class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.label_map = pd.read_csv('data/detection_labels_map.csv')
    
    def detect(self, image_path):
        # YOLO 추론
        # COCO → 성경 상징 매핑
        # 신뢰도 필터링 (>0.5)
        return detected_objects
```

#### **테스트:**
- 테스트 이미지 5장 준비
- SIM-01 시나리오 재현 (침대 감지)

---

### **2단계: SceneAnalyzer 구현 (2-3시간)**

#### **준비 사항:**
```bash
# CLIP 설치
pip install transformers torch pillow

# 모델 다운로드 (자동)
# openai/clip-vit-base-patch32 (~600MB)
```

#### **구현 파일:**
- `modules/scene_analyzer.py`

#### **핵심 기능:**
```python
class SceneAnalyzer:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def analyze(self, image_path):
        # 장소 분류 (실내/실외/교회/병원 등)
        # 시간대 추정 (아침/낮/저녁/밤)
        # 분위기 분석 (평화로운/슬픈/기쁜 등)
        return scene_results
```

#### **테스트:**
- 다양한 장면 이미지로 테스트
- 정확도 확인 (>70%)

---

### **3단계: EmotionDetector 구현 (2-3시간)**

#### **준비 사항:**
```bash
# TensorFlow 또는 PyTorch 설치
pip install tensorflow opencv-python

# 모델 준비 (2가지 옵션)
# 옵션 1: 사전 학습 모델 다운로드 (FER2013)
# 옵션 2: 간단한 모델 직접 학습
```

#### **구현 파일:**
- `modules/emotion_detector.py`

#### **핵심 기능:**
```python
class EmotionDetector:
    def __init__(self, model_path='models/emotion_mobilenetv2.h5'):
        self.model = load_model(model_path)
        self.emotions = ['happy', 'sad', 'angry', 'fear', 'neutral', 'surprise', 'disgust']
    
    def detect(self, image_path):
        # 얼굴 감지 (OpenCV)
        # 감정 분류 (MobileNetV2)
        # EmotionVector 생성
        return emotion_result
```

#### **테스트:**
- 다양한 표정 이미지로 테스트
- 7개 감정 분류 확인

---

## 📦 **필요한 패키지 설치**

```bash
# requirements.txt 업데이트
pip install ultralytics>=8.0.0
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install tensorflow>=2.13.0
pip install opencv-python>=4.8.0
pip install easyocr>=1.7.0
```

---

## 🧪 **테스트 계획**

### **단위 테스트**
각 모듈별로:
1. 정상 입력 테스트
2. 엣지 케이스 테스트 (얼굴 없음, 어두운 이미지 등)
3. 성능 측정 (추론 시간)

### **통합 테스트**
- SIM-01 시나리오 End-to-End
- 이미지 → 상징 → 주제 → 구절 전체 파이프라인

---

## 📊 **성공 기준**

### **ObjectDetector**
- ✅ COCO 80개 클래스 감지
- ✅ 신뢰도 >0.5 필터링
- ✅ 성경 상징 매핑 정확도 >80%

### **SceneAnalyzer**
- ✅ 장소 분류 정확도 >70%
- ✅ 시간대 추정 정확도 >60%
- ✅ 분위기 분석 정확도 >60%

### **EmotionDetector**
- ✅ 얼굴 감지율 >90%
- ✅ 감정 분류 정확도 >70%
- ✅ 7개 감정 모두 지원

---

## 🔧 **예상 문제 및 해결책**

### **문제 1: 모델 다운로드 느림**
**해결책:**
- 미리 다운로드 스크립트 실행
- 캐시 디렉토리 확인

### **문제 2: GPU 메모리 부족**
**해결책:**
- CPU 모드로 전환
- 배치 크기 줄이기
- 경량 모델 사용 (yolov8n, clip-vit-base)

### **문제 3: 한글 인코딩 오류**
**해결책:**
- UTF-8 인코딩 명시
- Windows 콘솔 설정 확인

---

## 📝 **다음 세션 시작 시 할 일**

### **1. 체크리스트 확인**
```
"체크리스트 보여줘"
또는
".agent/workflows/checklist.md 열어줘"
```

### **2. 이 문서 확인**
```
"다음 세션 계획 보여줘"
또는
".agent/workflows/next_session.md 열어줘"
```

### **3. 작업 시작**
```
"ObjectDetector 구현 시작"
```

---

## 🎯 **최종 목표 (이번 주)**

```
Phase 3.3 완료 → Phase 3.4 시작
- ObjectDetector ✅
- SceneAnalyzer ✅
- EmotionDetector ✅
- OCR (EasyOCR) ✅
- 통합 파이프라인 ✅
```

**예상 완료일**: 2026-02-13 (2일 후)

---

## 📌 **중요 링크**

- **체크리스트**: `.agent/workflows/checklist.md`
- **AI 구현 계획**: `.agent/workflows/ai_module_implementation.md`
- **AI 추가 구현**: `.agent/workflows/ai_module_additions.md`
- **GitHub**: https://github.com/Cornerstone-HCH/bible-gpt-dashboar.git

---

## 💬 **다음 세션 시작 멘트**

```
안녕하세요! 다음 세션을 시작합니다.

지난 세션 요약:
- Phase 1-2 완료 (100%)
- Phase 3.0-3.2 완료 (40%)
- P1/P2 모듈 구현 완료
- 웹 테스트 대시보드 완료

오늘 목표:
- Phase 3.3 AI 모듈 통합
- ObjectDetector, SceneAnalyzer, EmotionDetector 구현

준비되셨나요? 시작하겠습니다!
```
