"""
객체 인식 모듈 (YOLOv8)

이미지 내 객체를 감지하고 성경적 상징으로 매핑합니다.
"""

import os
import pandas as pd
from typing import List, Dict, Optional
from ultralytics import YOLO

class ObjectDetector:
    """
    YOLOv8 기반 객체 인식 및 상징 매핑 클래스
    """
    
    def __init__(self, 
                 model_path: str = 'yolov8n.pt', 
                 label_map_path: str = 'data/detection_labels_map.csv',
                 confidence_threshold: float = 0.5):
        """
        초기화
        
        Args:
            model_path: YOLOv8 모델 경로 (기본값: yolov8n.pt)
            label_map_path: COCO -> 상징 매핑 파일 경로
            confidence_threshold: 감지 신뢰도 임계값
        """
        # 모델 로드 (없으면 자동 다운로드됨)
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            self.model = None
            
        self.label_map_path = label_map_path
        self.confidence_threshold = confidence_threshold
        
        # 매핑 데이터 로드
        self._load_label_map()
        
    def _load_label_map(self):
        """매핑 CSV 로드"""
        if os.path.exists(self.label_map_path):
            self.label_map_df = pd.read_csv(self.label_map_path)
            # COCO 클래스명을 키로 하는 딕셔너리로 변환
            self.label_map = self.label_map_df.set_index('model_label').to_dict('index')
        else:
            print(f"매핑 파일을 찾을 수 없습니다: {self.label_map_path}")
            self.label_map = {}
            
    def detect(self, image_path: str) -> List[Dict]:
        """
        이미지에서 객체 감지 및 상징 매핑
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            List[Dict]: 감지된 객체 리스트
            [
                {
                    'coco_class': 'bed',
                    'bible_symbol': '병실·대기실',
                    'confidence': 0.92,
                    'bbox': [x1, y1, x2, y2],
                    'priority': 1.0
                },
                ...
            ]
        """
        if not self.model:
            return []
            
        # YOLO 추론
        results = self.model(image_path, conf=self.confidence_threshold)
        
        detected_objects = []
        
        for result in results:
            for box in result.boxes:
                # 클래스 ID 및 신뢰도
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coco_class = result.names[cls_id]
                
                # 상징 매핑 정보 확인
                if coco_class in self.label_map:
                    mapping = self.label_map[coco_class]
                    bible_symbol = mapping.get('bible_symbol')
                    
                    # 우선순위 확인 및 타입 변환 (str -> float 방지)
                    raw_priority = mapping.get('priority', 0.5)
                    try:
                        priority = float(raw_priority)
                    except (ValueError, TypeError):
                        priority = 0.5
                    
                    # 상징이 있는 경우에만 추가 (priority 0 제외)
                    if pd.notna(bible_symbol) and bible_symbol and priority > 0:
                        detected_objects.append({
                            'coco_class': coco_class,
                            'bible_symbol': bible_symbol,
                            'confidence': conf,
                            'bbox': box.xyxy[0].tolist(),
                            'priority': priority
                        })
                        
        return detected_objects

if __name__ == '__main__':
    # 간단한 테스트
    detector = ObjectDetector()
    print("ObjectDetector 초기화 완료")
    
    # 이미지가 있는 경우 테스트 (실제 배포시는 주석 처리)
    # image_path = 'sample_image.jpg'
    # if os.path.exists(image_path):
    #     results = detector.detect(image_path)
    #     for i, obj in enumerate(results, 1):
    #         print(f"{i}. {obj['coco_class']} -> {obj['bible_symbol']} (Conf: {obj['confidence']:.2f})")
