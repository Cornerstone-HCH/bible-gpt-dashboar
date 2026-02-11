"""
OCR 텍스트 추출 모듈 (EasyOCR)

이미지 내 한글/영어 텍스트를 추출하여 성경적 상징으로 변환합니다.
(오픈소스 지향: EasyOCR 활용)
"""

import easyocr
import pandas as pd
from typing import List, Dict

class TextExtractor:
    """
    EasyOCR 기반 텍스트 추출 클래스
    """
    
    def __init__(self, languages: List[str] = ['ko', 'en']):
        """
        초기화
        
        Args:
            languages: 추출할 언어 리스트
        """
        # 온디바이스 최적화: GPU 사용 가능 시 자동 활용
        try:
            self.reader = easyocr.Reader(languages)
        except Exception as e:
            print(f"EasyOCR Reader 초기화 실패: {e}")
            self.reader = None
            
        # 텍스트 -> 상징 매핑 (간이 사전, 향후 상징 정의 파일과 연동 가능)
        self.text_to_symbol_map = {
            '교회': '교회·예배당',
            '하나님': '교회·예배당',
            '예수': '교회·예배당',
            '사랑': '사랑(아가페)',
            '병원': '병실·대기실',
            '치료': '치유·회복',
            '공부': '공부·독서',
            '성경': '공부·독서',
            '예배': '교회·예배당'
        }

    def extract(self, image_path: str) -> List[Dict]:
        """
        텍스트 추출 및 상징 변환
        
        Returns:
            List[Dict]: [
                {
                    'text': '교회',
                    'confidence': 0.98,
                    'bible_symbol': '교회·예배당'
                }, ...
            ]
        """
        if not self.reader:
            return []
            
        try:
            results = self.reader.readtext(image_path)
        except Exception as e:
            print(f"OCR 실행 실패: {e}")
            return []
            
        extracted_data = []
        for (bbox, text, conf) in results:
            # 텍스트가 상징 사전에 있는지 확인
            bible_symbol = None
            for key, symbol in self.text_to_symbol_map.items():
                if key in text:
                    bible_symbol = symbol
                    break
            
            extracted_data.append({
                'text': text,
                'confidence': float(conf),
                'bible_symbol': bible_symbol
            })
            
        return extracted_data

if __name__ == '__main__':
    # 간단한 테스트
    extractor = TextExtractor()
    print("TextExtractor (EasyOCR) 초기화 완료")
