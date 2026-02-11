"""
P1 상징 매핑 모듈

정본 9장 기준 46개 상징 체계 적용
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


class SymbolMapper:
    """
    P1 상징 매핑 모듈
    
    Perception 결과 (객체/장면/감정) → 46개 성경적 상징으로 변환
    """
    
    def __init__(self, 
                 symbol_def_path: str = 'data/symbol_definitions.csv',
                 detection_map_path: str = 'data/detection_labels_map.csv'):
        """
        초기화
        
        Args:
            symbol_def_path: 46개 상징 정의 파일 경로
            detection_map_path: COCO → 성경 상징 매핑 파일 경로
        """
        self.symbol_defs = pd.read_csv(symbol_def_path)
        self.detection_map = pd.read_csv(detection_map_path)
        
        # 매핑 규칙 로드
        self._load_mapping_rules()
    
    def _load_mapping_rules(self):
        """
        객체/장면/감정 → 상징 매핑 규칙 로드
        """
        # COCO 객체 → 상징 매핑 (detection_labels_map.csv 기반)
        self.object_to_symbol = {}
        for _, row in self.detection_map.iterrows():
            coco_class = row['model_label']
            bible_symbol = row.get('bible_symbol', '')
            priority = row.get('priority', 0.0)
            
            # priority를 float로 변환
            try:
                priority = float(priority) if pd.notna(priority) else 0.0
            except (ValueError, TypeError):
                priority = 0.0
            
            if pd.notna(bible_symbol) and bible_symbol and priority > 0:
                self.object_to_symbol[coco_class] = {
                    'symbol': bible_symbol,
                    'priority': priority
                }
        
        # 장면 → 상징 매핑 (하드코딩, 추후 CSV로 분리 가능)
        self.scene_to_symbol = {
            '실내': [{'symbol': '실내·집중', 'confidence': 0.7}],
            '실외': [{'symbol': '실외·광장', 'confidence': 0.7}],
            '교회': [{'symbol': '교회·예배당', 'confidence': 0.9}],
            '병원': [{'symbol': '병원', 'confidence': 0.9}, 
                    {'symbol': '병실·대기실', 'confidence': 0.8}],
            '자연': [{'symbol': '숲·나무', 'confidence': 0.7}],
            '도시': [{'symbol': '도시·거리', 'confidence': 0.8}],
            '밤': [{'symbol': '달·별', 'confidence': 0.8}, 
                  {'symbol': '골목·어둑함', 'confidence': 0.6}],
            '낮': [{'symbol': '해·빛', 'confidence': 0.8}],
            '아침': [{'symbol': '해·빛', 'confidence': 0.7}],
            '저녁': [{'symbol': '달·별', 'confidence': 0.6}],
        }
        
        # 감정 → 상징 매핑
        self.emotion_to_symbol = {
            'sad': [{'symbol': '고립·홀로', 'confidence': 0.7}],
            'happy': [{'symbol': '결혼·잔치', 'confidence': 0.6}, 
                     {'symbol': '음악·예술', 'confidence': 0.5}],
            'fear': [{'symbol': '폭풍·바람', 'confidence': 0.7}, 
                    {'symbol': '골목·어둑함', 'confidence': 0.6}],
            'angry': [{'symbol': '전쟁·무기', 'confidence': 0.5}],
            'neutral': [{'symbol': '실내·집중', 'confidence': 0.5}],
            'surprise': [{'symbol': '해·빛', 'confidence': 0.4}],
            'disgust': [],
        }
    
    def map(self, perception_result: Dict) -> List[Dict]:
        """
        Perception 결과 → 상징 리스트
        
        Args:
            perception_result: {
                'objects': [
                    {'coco_class': 'bed', 'confidence': 0.92, 'bbox': [...]},
                    ...
                ],
                'scene': {
                    'location': ('병원', 0.85),
                    'time': ('밤', 0.78),
                    'mood': ('고요한', 0.72)
                },
                'emotion': {
                    'primary_label': 'sad',
                    'intensity': 0.65,
                    'emotion_probs': {...}
                }
            }
        
        Returns:
            List[Dict]: [
                {
                    'symbol': '침대',
                    'confidence': 0.85,
                    'source': 'object',
                    'core_group': '공간장면'
                },
                ...
            ]
        """
        symbols = []
        
        # 1. 객체 → 상징
        objects = perception_result.get('objects', [])
        for obj in objects:
            coco_class = obj.get('coco_class', '')
            obj_conf = obj.get('confidence', 0.0)
            
            if coco_class in self.object_to_symbol:
                mapping = self.object_to_symbol[coco_class]
                symbol = mapping['symbol']
                priority = mapping['priority']
                
                symbols.append({
                    'symbol': symbol,
                    'confidence': obj_conf * priority,
                    'source': 'object',
                    'core_group': self._get_core_group(symbol),
                    'raw_confidence': obj_conf,
                    'priority': priority
                })
        
        # 2. 장면 → 상징
        scene = perception_result.get('scene', {})
        
        # 2-1. 장소
        location = scene.get('location', ('', 0))
        if isinstance(location, tuple) and len(location) == 2:
            loc_name, loc_conf = location
            if loc_name in self.scene_to_symbol:
                for mapping in self.scene_to_symbol[loc_name]:
                    symbols.append({
                        'symbol': mapping['symbol'],
                        'confidence': loc_conf * mapping['confidence'],
                        'source': 'scene_location',
                        'core_group': self._get_core_group(mapping['symbol']),
                        'raw_confidence': loc_conf
                    })
        
        # 2-2. 시간대
        time = scene.get('time', ('', 0))
        if isinstance(time, tuple) and len(time) == 2:
            time_name, time_conf = time
            if time_name in self.scene_to_symbol:
                for mapping in self.scene_to_symbol[time_name]:
                    symbols.append({
                        'symbol': mapping['symbol'],
                        'confidence': time_conf * mapping['confidence'],
                        'source': 'scene_time',
                        'core_group': self._get_core_group(mapping['symbol']),
                        'raw_confidence': time_conf
                    })
        
        # 3. 감정 → 상징
        emotion = perception_result.get('emotion', {})
        primary_emotion = emotion.get('primary_label', 'neutral')
        emotion_intensity = emotion.get('intensity', 0.0)
        
        if primary_emotion in self.emotion_to_symbol:
            for mapping in self.emotion_to_symbol[primary_emotion]:
                symbols.append({
                    'symbol': mapping['symbol'],
                    'confidence': emotion_intensity * mapping['confidence'],
                    'source': 'emotion',
                    'core_group': self._get_core_group(mapping['symbol']),
                    'raw_confidence': emotion_intensity
                })
        
        # 중복 제거 및 점수 합산
        symbol_scores = {}
        for s in symbols:
            key = s['symbol']
            if key not in symbol_scores:
                symbol_scores[key] = s
            else:
                # 같은 상징이 여러 소스에서 나온 경우 점수 합산
                symbol_scores[key]['confidence'] += s['confidence']
                # 소스 병합
                if isinstance(symbol_scores[key]['source'], str):
                    symbol_scores[key]['source'] = [symbol_scores[key]['source']]
                if isinstance(s['source'], str):
                    symbol_scores[key]['source'].append(s['source'])
                else:
                    symbol_scores[key]['source'].extend(s['source'])
        
        # 정렬 및 반환 (신뢰도 높은 순)
        sorted_symbols = sorted(
            symbol_scores.values(),
            key=lambda x: x['confidence'],
            reverse=True
        )
        
        return sorted_symbols[:10]  # Top 10
    
    def _get_core_group(self, symbol: str) -> str:
        """
        상징 → 코어 군 매핑
        
        Args:
            symbol: 상징 이름
        
        Returns:
            코어 군 이름 (자연, 생명, 인공물, 인간활동, 공간장면)
        """
        symbol_info = self.symbol_defs[self.symbol_defs['symbol'] == symbol]
        if not symbol_info.empty:
            return symbol_info.iloc[0]['core_group']
        return 'unknown'
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """
        상징 상세 정보 조회
        
        Args:
            symbol: 상징 이름
        
        Returns:
            상징 정보 딕셔너리 또는 None
        """
        symbol_info = self.symbol_defs[self.symbol_defs['symbol'] == symbol]
        if not symbol_info.empty:
            row = symbol_info.iloc[0]
            return {
                'symbol_id': row['symbol_id'],
                'symbol': row['symbol'],
                'core_group': row['core_group'],
                'meaning': row['meaning'],
                'primary_themes': row['primary_themes'],
                'detection_hints': row['detection_hints']
            }
        return None


if __name__ == '__main__':
    # 테스트 코드
    mapper = SymbolMapper()
    
    # 테스트 입력 (SIM-01 시나리오: 병상, 밤, 슬픔)
    test_perception = {
        'objects': [
            {'coco_class': 'bed', 'confidence': 0.92, 'bbox': [100, 100, 300, 300]},
            {'coco_class': 'person', 'confidence': 0.85, 'bbox': [150, 150, 250, 350]}
        ],
        'scene': {
            'location': ('병원', 0.85),
            'time': ('밤', 0.78),
            'mood': ('고요한', 0.72)
        },
        'emotion': {
            'primary_label': 'sad',
            'intensity': 0.65,
            'emotion_probs': {
                'sad': 0.65,
                'neutral': 0.20,
                'fear': 0.10,
                'happy': 0.05
            }
        }
    }
    
    # 상징 매핑 실행
    symbols = mapper.map(test_perception)
    
    print("=== P1 상징 매핑 결과 ===")
    print(f"총 {len(symbols)}개 상징 추출\n")
    
    for i, s in enumerate(symbols, 1):
        print(f"{i}. {s['symbol']}")
        print(f"   신뢰도: {s['confidence']:.3f}")
        print(f"   소스: {s['source']}")
        print(f"   코어 군: {s['core_group']}")
        
        # 상징 상세 정보
        info = mapper.get_symbol_info(s['symbol'])
        if info:
            print(f"   의미: {info['meaning']}")
            print(f"   주요 주제: {info['primary_themes']}")
        print()
