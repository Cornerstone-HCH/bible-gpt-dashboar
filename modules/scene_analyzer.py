"""
장면 인식 모듈 (CLIP)

이미지의 장소, 시간대, 분위기를 분석하여 성경적 상징을 추천합니다.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from typing import List, Dict, Tuple

class SceneAnalyzer:
    """
    CLIP 기반 장면 및 분위기 분석 클래스
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        초기화
        
        Args:
            model_name: HuggingFace CLIP 모델명
        """
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
        except Exception as e:
            print(f"CLIP 모델 로드 실패: {e}")
            self.model = None
            
        # 분석 카테고리 정의 (정본 기준)
        self.categories = {
            'location': [
                'indoor space', 'outdoor plaza', 'church sanctuary', 
                'hospital room', 'nature forest', 'city street', 
                'wilderness desert', 'mountain top', 'sea side', 'snowy field'
            ],
            'time': [
                'dawn morning', 'bright daytime', 'sunset evening', 
                'dark night', 'midnight'
            ],
            'mood': [
                'peaceful and calm', 'sad and sorrowful', 'joyful and happy', 
                'reverent and holy', 'dark and scary', 'lonely and isolated'
            ],
            'weather': [
                'sunny day', 'rainy weather', 'snowy winter', 'cloudy sky', 
                'stormy weather', 'foggy morning'
            ]
        }
        
        # 카테고리 영문 -> 한글 상징 매핑 (120개 표준 라벨 기준)
        self.label_to_symbols = {
            'indoor space': ['집'],
            'outdoor plaza': ['공원'],
            'church sanctuary': ['교회'],
            'hospital room': ['병원'],
            'nature forest': ['숲', '나무'],
            'city street': ['표지판', '신호등'],
            'wilderness desert': ['사막'],
            'mountain top': ['산'],
            'sea side': ['바다'],
            'snowy field': ['눈', '서리'],
            'dawn morning': ['해', '빛'],
            'bright daytime': ['해', '빛'],
            'sunset evening': ['달', '별'],
            'dark night': ['달', '별', '어둠'],
            'midnight': ['달', '별', '어둠'],
            'peaceful and calm': ['양'], # 평강의 상징
            'sad and sorrowful': ['눈물'],
            'joyful and happy': ['잔'],
            'reverent and holy': ['교회'],
            'dark and scary': ['어둠'],
            'lonely and isolated': ['들꽃'], # 사막 대신 좀 더 중립적인 광야/들꽃 느낌으로 변경
            'sunny day': ['해', '빛'],
            'rainy weather': ['비'],
            'snowy winter': ['눈', '서리'],
            'cloudy sky': ['구름'],
            'stormy weather': ['번개', '물'],
            'foggy morning': ['안개']
        }

    def analyze(self, image_path: str) -> Dict:
        """
        장면 분석 수행
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Dict: 분석 결과 (장소, 시간, 분위기 및 추천 상징)
        """
        if not self.model:
            return {}
            
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지 로드 실패: {e}")
            return {}
            
        results = {}
        suggested_symbols = []
        
        with torch.no_grad():
            for category, labels in self.categories.items():
                # CLIP 추론
                inputs = self.processor(
                    text=labels, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                # 최고 점수 결과 추출
                max_idx = probs.argmax().item()
                conf = float(probs[0][max_idx])
                label = labels[max_idx]
                
                results[category] = {
                    'label': label,
                    'confidence': conf
                }
                
                # 상징 추천 추가 (여러 심볼 지원)
                if label in self.label_to_symbols:
                    for symbol in self.label_to_symbols[label]:
                        suggested_symbols.append({
                            'symbol': symbol,
                            'confidence': conf,
                            'source': f'scene_{category}'
                        })
                    
        results['suggested_symbols'] = suggested_symbols
        return results

if __name__ == '__main__':
    # 간단한 테스트
    analyzer = SceneAnalyzer()
    print("SceneAnalyzer 초기화 완료")
