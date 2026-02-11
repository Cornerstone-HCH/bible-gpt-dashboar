"""
감정 인식 모듈 (MobileNetV2 기반)

얼굴 및 이미지 분위기를 통해 7개 기본 감정을 분류합니다.
(온디바이스/모바일 최적화: MobileNetV2 아키텍처 사용)
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, List

class EmotionDetector:
    """
    MobileNetV2 기반 감정 인식 클래스
    """
    
    def __init__(self, model_path: str = None):
        """
        초기화
        
        Args:
            model_path: 학습된 모델 가중치 경로 (없으면 기본 아키텍처만 로드)
        """
        # 온디바이스 최적화 아키텍처: MobileNetV2
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        
        # 감정 분류를 위해 마지막 레이어 수정 (7개 감정)
        # 0: neutral, 1: happy, 2: sad, 3: surprise, 4: fear, 5: angry, 6: disgust
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, 7)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"이미 학습된 가중치 로드 실패: {e}")
        
        # 감정 레이블
        self.emotions = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'angry', 'disgust']
        
        # 전처리
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def detect(self, image_path: str) -> Dict:
        """
        이미지에서 감정 인식 수행
        
        Returns:
            Dict: {
                'primary_label': 'happy',
                'intensity': 0.85,
                'emotion_probs': {'happy': 0.85, 'neutral': 0.1, ...}
            }
        """
        try:
            image = Image.open(image_path).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"이미지 처리 실패: {e}")
            return {}
            
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # 결과 정리
            max_idx = torch.argmax(probs).item()
            primary_label = self.emotions[max_idx]
            intensity = float(probs[max_idx])
            
            emotion_probs = {self.emotions[i]: float(probs[i]) for i in range(len(self.emotions))}
            
        return {
            'primary_label': primary_label,
            'intensity': intensity,
            'emotion_probs': emotion_probs
        }

if __name__ == '__main__':
    # 간단한 테스트
    detector = EmotionDetector()
    print("EmotionDetector (MobileNetV2) 초기화 완료")
