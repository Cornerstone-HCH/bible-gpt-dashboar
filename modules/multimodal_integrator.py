"""
멀티모달 신호 통합 모듈 (MultimodalIntegrator)

객체, 장면, 감정, OCR 등 다양한 AI 모듈의 결과를 하나로 통합하여 
최종 상징 리스트(Top 10)를 생성합니다.
"""

from typing import List, Dict, Any

class MultimodalIntegrator:
    """
    다양한 Perception 신호를 통합하여 상징 점수를 산출하는 클래스
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        초기화
        
        Args:
            weights: 각 모듈별 신뢰도 가중치
        """
        # 정본 MD 요구사항 및 일반적 중요도에 따른 기본 가중치
        self.weights = weights or {
            'object': 0.50,    # 객체가 가장 확실한 상징 (침대, 책 등)
            'scene': 0.25,     # 장면 분위기 (병원, 교회 등)
            'emotion': 0.15,   # 감정 (슬픔, 기쁨 등)
            'ocr': 0.10        # 발견된 텍스트
        }

    def integrate(self, 
                  object_results: List[Dict], 
                  scene_results: Dict, 
                  emotion_results: Dict, 
                  ocr_results: List[Dict]) -> List[Dict]:
        """
        모든 AI 신호를 통합하여 상징 리스트 생성
        
        Returns:
            List[Dict]: 통합된 상징 리스트 (점수순 정렬)
        """
        symbol_scores = {}
        
        # 1. 객체 인식 통합
        for obj in object_results:
            symbol = obj['bible_symbol']
            # 점수 = 신뢰도 * 우선순위(상징중요도) * 모듈가중치
            score = obj['confidence'] * obj['priority'] * self.weights['object']
            self._update_symbol_score(symbol_scores, symbol, score, 'object')
            
        # 2. 장면/분위기 인식 통합
        for suggestion in scene_results.get('suggested_symbols', []):
            symbol = suggestion['symbol']
            score = suggestion['confidence'] * self.weights['scene']
            self._update_symbol_score(symbol_scores, symbol, score, suggestion['source'])
            
        # 3. 감정 인식 통합
        if emotion_results:
            primary_label = emotion_results['primary_label']
            intensity = emotion_results['intensity']
            # 감정 -> 상징 매핑 (임시: SceneAnalyzer와 유사한 방식)
            # 7개 기본 감정에 따른 상징 연결 (정본 매핑 필요)
            emotion_to_symbol = {
                'happy': '결혼·잔치',
                'sad': '고립·홀로',
                'surprise': '해·빛',
                'fear': '폭풍·바람',
                'angry': '전쟁·무기',
                'disgust': '광야·거친 돌',
                'neutral': '실내·집중'
            }
            if primary_label in emotion_to_symbol:
                symbol = emotion_to_symbol[primary_label]
                score = intensity * self.weights['emotion']
                self._update_symbol_score(symbol_scores, symbol, score, 'emotion')
                
        # 4. OCR 텍스트 통합
        for ocr in ocr_results:
            if ocr.get('bible_symbol'):
                symbol = ocr['bible_symbol']
                score = ocr['confidence'] * self.weights['ocr']
                self._update_symbol_score(symbol_scores, symbol, score, 'ocr')

        # 리스트 변환 및 정렬
        final_symbols = []
        for symbol, data in symbol_scores.items():
            final_symbols.append({
                'symbol': symbol,
                'confidence': min(1.0, data['score']), # 1.0 캡
                'sources': list(data['sources'])
            })
            
        # 신뢰도 높은 순으로 정렬 (Top 10)
        final_symbols.sort(key=lambda x: x['confidence'], reverse=True)
        
        return final_symbols[:10]

    def _update_symbol_score(self, scores: Dict, symbol: str, score: float, source: str):
        """심볼 점수 합산 및 소스 기록"""
        if symbol not in scores:
            scores[symbol] = {'score': 0.0, 'sources': set()}
        
        # 단순 합산 대신 감쇠 합산(Asymptotic sum)이나 Max를 고려할 수 있으나, 
        # 현재는 멀티모달 신뢰 결합을 위해 단순 가중치 합산 후 1.0 캡 적용
        scores[symbol]['score'] += score
        scores[symbol]['sources'].add(source)

if __name__ == '__main__':
    integrator = MultimodalIntegrator()
    print("MultimodalIntegrator 초기화 완료")
