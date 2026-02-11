"""
P2 주제 매핑 모듈

46개 상징 → 24개 신학적 주제 변환
"""

import pandas as pd
from typing import List, Dict
from pathlib import Path


class ThemeMapper:
    """
    P2 주제 매핑 모듈
    
    46개 상징 → 24개 신학적 주제로 변환
    topic_weights_v1.csv 기반 가중치 적용
    """
    
    def __init__(self, topic_weights_path: str = 'data/topic_weights_v1.csv'):
        """
        초기화
        
        Args:
            topic_weights_path: 주제 가중치 파일 경로
        """
        self.topic_weights_df = pd.read_csv(topic_weights_path)
        
        # 24개 주제 목록 (컬럼명에서 추출)
        self.themes = [
            '창조·섭리', '임재·영광', '거룩·성화', '은혜',
            '믿음·신뢰', '회개', '구원·의(칭의)', '소망·부활',
            '말씀·진리', '기도', '예배·찬양', '순종·겸손',
            '사랑(아가페)', '인내·시험', '지혜·분별', '성령·열매',
            '평강·샬롬', '공동체·연합', '정의·공의', '화해·용서(관계)',
            '소명·일·청지기', '가정·양육', '치유·회복', '보호·인도'
        ]
        
        # 상징(라벨) → 주제 가중치 매핑 생성
        self._build_symbol_to_theme_weights()
    
    def _build_symbol_to_theme_weights(self):
        """
        상징 → 주제 가중치 매핑 딕셔너리 생성
        """
        self.symbol_to_themes = {}
        
        for _, row in self.topic_weights_df.iterrows():
            symbol = row['라벨']
            theme_weights = {}
            
            for theme in self.themes:
                weight = row.get(theme, 0.0)
                try:
                    weight = float(weight) if pd.notna(weight) else 0.0
                except (ValueError, TypeError):
                    weight = 0.0
                
                if weight > 0:
                    theme_weights[theme] = weight
            
            if theme_weights:
                self.symbol_to_themes[symbol] = theme_weights
    
    def map(self, symbols: List[Dict]) -> Dict[str, float]:
        """
        상징 리스트 → 주제 점수 변환
        
        Args:
            symbols: SymbolMapper의 출력
                [
                    {
                        'symbol': '병실·대기실',
                        'confidence': 0.765,
                        'source': 'scene_location',
                        'core_group': '공간장면'
                    },
                    ...
                ]
        
        Returns:
            Dict[str, float]: {
                '평강·샬롬': 0.85,
                '치유·회복': 0.72,
                ...
            }
        """
        theme_scores = {theme: 0.0 for theme in self.themes}
        
        # 각 상징에 대해 주제 가중치 적용
        for symbol_info in symbols:
            symbol = symbol_info['symbol']
            symbol_confidence = symbol_info['confidence']
            
            # 상징 → 피사체 라벨 매핑 (간단히 직접 매핑)
            # 실제로는 symbol_definitions.csv와 topic_weights_v1.csv를 연결해야 함
            # 여기서는 간단히 상징 이름으로 직접 매칭 시도
            
            # 상징 이름에서 키워드 추출하여 매칭
            matched_labels = self._match_symbol_to_labels(symbol)
            
            for label in matched_labels:
                if label in self.symbol_to_themes:
                    theme_weights = self.symbol_to_themes[label]
                    
                    for theme, weight in theme_weights.items():
                        # 상징 신뢰도 × 주제 가중치
                        theme_scores[theme] += symbol_confidence * weight
        
        # 정규화 (0-1 범위)
        max_score = max(theme_scores.values()) if theme_scores.values() else 1.0
        if max_score > 0:
            theme_scores = {
                theme: score / max_score 
                for theme, score in theme_scores.items()
            }
        
        # 점수가 0보다 큰 주제만 반환
        return {
            theme: score 
            for theme, score in theme_scores.items() 
            if score > 0
        }
    
    def _match_symbol_to_labels(self, symbol: str) -> List[str]:
        """
        상징 이름 → topic_weights_v1.csv의 라벨 매칭
        
        Args:
            symbol: 상징 이름 (예: '병실·대기실', '침대')
        
        Returns:
            매칭된 라벨 리스트
        """
        matched_labels = []
        
        # 직접 매칭 시도
        if symbol in self.symbol_to_themes:
            matched_labels.append(symbol)
        
        # 키워드 기반 매칭
        keyword_mapping = {
            '병실': ['병원', '침상'],
            '대기실': ['병원'],
            '침대': ['침상'],
            '병원': ['병원'],
            '달': ['하늘'],
            '별': ['하늘'],
            '골목': ['길'],
            '어둑함': [],
            '고립': [],
            '홀로': [],
            '교회': ['성전'],
            '예배당': ['성전'],
            '책': ['성경책', '책(두루마리)'],
            '공부': ['책(두루마리)'],
            '독서': ['책(두루마리)'],
            '결혼': ['부부'],
            '잔치': ['식탁'],
            '음악': ['악기'],
            '예술': ['악기'],
            '스포츠': [],
            '경기': [],
            '육아': ['어린이'],
            '돌봄': ['어린이', '병자'],
            '여행': ['배', '길'],
            '순례': ['길'],
            '전쟁': ['칼'],
            '무기': ['칼'],
            '정치': [],
            '회의': [],
            '실내': ['집'],
            '집중': [],
            '실외': ['광장'],
            '광장': ['광장'],
            '군중': ['군중'],
            '성지': ['성전'],
            '랜드마크': ['성전'],
            '산': ['산'],
            '바다': ['바다'],
            '강': ['강'],
            '샘': ['강'],
            '광야': [],
            '숲': ['숲'],
            '나무': ['나무'],
            '꽃': ['들꽃'],
            '풀': ['들꽃'],
            '해': ['하늘'],
            '빛': ['등불'],
            '비': ['비'],
            '눈': ['비'],
            '폭풍': ['바람'],
            '바람': ['바람'],
            '씨앗': ['씨앗'],
            '열매': ['열매'],
            '포도나무': ['포도나무'],
            '가지': ['가지'],
            '양': ['양'],
            '사자': [],
            '비둘기': ['비둘기'],
            '물고기': ['물고기'],
            '어린아이': ['어린이'],
            '노인': ['노인'],
            '인간': ['친구'],
            '표정': [],
            '집': ['집'],
            '가정': ['집'],
            '도시': ['광장'],
            '거리': ['광장'],
            '길': ['길'],
            '다리': ['다리'],
            '학교': ['학교'],
            '시장': ['시장'],
            '상점': ['시장'],
            '법정': [],
            '재판': [],
            '공장': [],
            '도구': [],
            '차량': ['수레/마차'],
            '노동': ['쟁기'],
            '수고': ['쟁기'],
            '농사': ['쟁기', '밀'],
            '수확': ['낫'],
        }
        
        # 키워드 매칭
        for keyword, labels in keyword_mapping.items():
            if keyword in symbol:
                matched_labels.extend(labels)
        
        # 중복 제거
        return list(set(matched_labels))
    
    def get_top_themes(self, theme_scores: Dict[str, float], top_n: int = 5) -> List[Dict]:
        """
        상위 N개 주제 반환
        
        Args:
            theme_scores: map() 결과
            top_n: 반환할 주제 개수
        
        Returns:
            List[Dict]: [
                {'theme': '평강·샬롬', 'score': 0.85},
                ...
            ]
        """
        sorted_themes = sorted(
            theme_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'theme': theme, 'score': score}
            for theme, score in sorted_themes[:top_n]
        ]


if __name__ == '__main__':
    # 테스트 코드
    from symbol_mapper import SymbolMapper
    
    # SymbolMapper 초기화
    symbol_mapper = SymbolMapper()
    theme_mapper = ThemeMapper()
    
    # 테스트 입력 (SIM-01)
    test_perception = {
        'objects': [
            {'coco_class': 'bed', 'confidence': 0.92},
            {'coco_class': 'person', 'confidence': 0.85}
        ],
        'scene': {
            'location': ('병원', 0.85),
            'time': ('밤', 0.78),
            'mood': ('고요한', 0.72)
        },
        'emotion': {
            'primary_label': 'sad',
            'intensity': 0.65
        }
    }
    
    # P1: 상징 매핑
    symbols = symbol_mapper.map(test_perception)
    
    print("=== P1 상징 매핑 결과 ===")
    for i, s in enumerate(symbols[:5], 1):
        print(f"{i}. {s['symbol']} (신뢰도: {s['confidence']:.3f})")
    print()
    
    # P2: 주제 매핑
    theme_scores = theme_mapper.map(symbols)
    top_themes = theme_mapper.get_top_themes(theme_scores, top_n=5)
    
    print("=== P2 주제 매핑 결과 ===")
    print(f"총 {len(theme_scores)}개 주제 추출\n")
    
    for i, t in enumerate(top_themes, 1):
        print(f"{i}. {t['theme']}")
        print(f"   점수: {t['score']:.3f}")
        print()
