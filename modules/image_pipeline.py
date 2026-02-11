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

    def process(self, image_path: str) -> List[Dict]:
        """
        이미지 -> 최종 상징 추출 (End-to-End)
        
        Args:
            image_path: 분석할 이미지 파일 경로
            
        Returns:
            List[Dict]: 최종 상징 리스트 (Top 10)
        """
        # 1. 각 개별 모듈 실행 (Perception 지각)
        obj_results = self.object_detector.detect(image_path)
        scene_results = self.scene_analyzer.analyze(image_path)
        emotion_results = self.emotion_detector.detect(image_path)
        ocr_results = self.text_extractor.extract(image_path)
        
        # 2. 멀티모달 신호 통합 (Integration 통합)
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
