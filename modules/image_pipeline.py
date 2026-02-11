"""
메인 AI 이미지 파이프라인 (ImagePipeline)

이미지를 입력받아 전처리 및 AI 모듈(Perception)을 거쳐 
최종 성경적 상징을 도출하는 전체 프로세스를 관리합니다.
"""

from .object_detector import ObjectDetector
from .scene_analyzer import SceneAnalyzer
from .emotion_detector import EmotionDetector
from .text_extractor import TextExtractor
from .multimodal_integrator import MultimodalIntegrator
from typing import List, Dict

class ImagePipeline:
    """
    BibleGPT AI 핵심 파이프라인 클래스
    """
    
    def __init__(self, 
                 yolo_model: str = 'yolov8n.pt',
                 clip_model: str = 'openai/clip-vit-base-patch32'):
        """
        AI 모듈 전체 초기화
        """
        print("AI 파이프라인 초기화 중...")
        self.object_detector = ObjectDetector(model_path=yolo_model)
        self.scene_analyzer = SceneAnalyzer(model_name=clip_model)
        self.emotion_detector = EmotionDetector()
        self.text_extractor = TextExtractor()
        self.integrator = MultimodalIntegrator()
        print("AI 파이프라인 초기화 완료!")

    def process(self, image_path: str, force_ocr: bool = False) -> List[Dict]:
        """
        이미지 -> 최종 상징 추출 (Smart OCR 로직 포함)
        """
        # 1. 빠른 모듈 실행 (객체, 장면, 감정)
        obj_results = self.object_detector.detect(image_path)
        scene_results = self.scene_analyzer.analyze(image_path)
        emotion_results = self.emotion_detector.detect(image_path)
        
        # 2. [Smart OCR] 내부 감지 로직에 의한 선택적 실행
        trigger_ocr = force_ocr
        if not trigger_ocr:
            # 텍스트가 포함될 확률이 높은 객체들
            text_bearing_objects = {'book', 'stop sign', 'traffic light', 'laptop', 'cell phone', 'tv', 'refrigerator'}
            detected_classes = {obj['coco_class'] for obj in obj_results}
            
            # 조건 1: 특정 객체 감지
            if detected_classes & text_bearing_objects:
                trigger_ocr = True
            # 조건 2: 도시 거리 등 텍스트가 많은 장면
            elif scene_results.get('location', {}).get('label') == 'city street':
                trigger_ocr = True
            # 조건 3: 신뢰도 높은 객체 분포 등 (필요 시 확장 가능)

        ocr_results = []
        if trigger_ocr:
            print(f"[Smart OCR] 텍스트 가능성 감지됨 -> OCR 가동")
            ocr_results = self.text_extractor.extract(image_path)
        else:
            print(f"[Smart OCR] 텍스트 가능성 낮음 -> OCR 건너跳")
        
        # 3. 멀티모달 신호 통합
        final_symbols = self.integrator.integrate(
            object_results=obj_results,
            scene_results=scene_results,
            emotion_results=emotion_results,
            ocr_results=ocr_results
        )
        
        return final_symbols

if __name__ == '__main__':
    # 메인 파이프라인 로드 테스트
    pipeline = ImagePipeline()
    print("ImagePipeline 로드 성공")
