# 바이블GPT 대시보드 데이터 로더

import pandas as pd
from typing import Dict, List, Tuple
import config

def load_bible_objects() -> pd.DataFrame:
    """
    120개 피사체(상징) 정의 로드
    
    Returns:
        DataFrame with columns: id, 가족군, 라벨, 별칭(동의어 예시), 비고
    """
    try:
        df = pd.read_csv(config.BIBLE_OBJECTS_FILE, encoding='utf-8')
        print(f"[OK] 피사체 정의 로드 완료: {len(df)}개")
        return df
    except Exception as e:
        print(f"[ERROR] 피사체 정의 로드 실패: {e}")
        return pd.DataFrame()

def load_detection_labels() -> pd.DataFrame:
    """
    COCO 모델 라벨 매핑 로드
    
    Returns:
        DataFrame with columns: model_label, object_tag, synonyms_ko, synonyms_en, source, source_id, allow_flag, notes
    """
    try:
        df = pd.read_csv(config.DETECTION_LABELS_FILE, encoding='utf-8')
        print(f"[OK] 탐지 라벨 매핑 로드 완료: {len(df)}개")
        return df
    except Exception as e:
        print(f"[ERROR] 탐지 라벨 매핑 로드 실패: {e}")
        return pd.DataFrame()

def load_topic_weights() -> pd.DataFrame:
    """
    120개 피사체 × 24개 주제 가중치 매트릭스 로드
    
    Returns:
        DataFrame with columns: id, 대분류, 라벨, 별칭, [24개 주제 컬럼들]
    """
    try:
        df = pd.read_csv(config.TOPIC_WEIGHTS_FILE, encoding='utf-8')
        print(f"[OK] 주제 가중치 매트릭스 로드 완료: {len(df)}개 피사체 x {len(config.TOPICS_KO)}개 주제")
        return df
    except Exception as e:
        print(f"[ERROR] 주제 가중치 매트릭스 로드 실패: {e}")
        return pd.DataFrame()

def load_topic_symbol_mapping() -> Dict[str, List[str]]:
    """
    주제별 관련 상징 매핑 로드
    
    Returns:
        Dict[주제명, List[상징들]]
    """
    try:
        mapping_file = config.DATA_DIR / "topic_symbol_mapping.csv"
        df = pd.read_csv(mapping_file, encoding='utf-8')
        
        mapping = {}
        for _, row in df.iterrows():
            topic = row['topic_name']
            symbols_str = row['symbols']
            # 쉼표로 구분된 상징들을 리스트로 변환
            symbols = [s.strip() for s in symbols_str.split(',')]
            mapping[topic] = symbols
        
        print(f"[OK] 주제별 상징 매핑 로드 완료: {len(mapping)}개 주제")
        return mapping
    except Exception as e:
        print(f"[WARNING] 주제별 상징 매핑 로드 실패: {e}")
        return {}

def load_verse_bank() -> pd.DataFrame:
    """
    24개 주제별 성경 구절 데이터베이스 로드
    
    Returns:
        DataFrame with columns: topic_id, topic_name, ref, book_ko, chapter, verse, match_count, text_ko
    """
    try:
        df = pd.read_csv(config.VERSE_BANK_FILE, encoding='utf-8')
        print(f"[OK] 성경 구절 데이터베이스 로드 완료: {len(df)}개 구절")
        return df
    except Exception as e:
        print(f"[ERROR] 성경 구절 데이터베이스 로드 실패: {e}")
        return pd.DataFrame()

def get_symbol_categories(bible_objects_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    피사체를 가족군별로 그룹화
    
    Args:
        bible_objects_df: 피사체 정의 DataFrame
        
    Returns:
        Dict[가족군, List[라벨]]
    """
    if bible_objects_df.empty:
        return {}
    
    categories = {}
    for _, row in bible_objects_df.iterrows():
        category = row['가족군']
        label = row['라벨']
        
        if category not in categories:
            categories[category] = []
        categories[category].append(label)
    
    return categories

def get_topic_weights_for_symbol(topic_weights_df: pd.DataFrame, symbol: str) -> Dict[str, float]:
    """
    특정 피사체에 대한 주제별 가중치 반환
    
    Args:
        topic_weights_df: 주제 가중치 DataFrame
        symbol: 피사체 라벨
        
    Returns:
        Dict[주제명, 가중치]
    """
    if topic_weights_df.empty:
        return {}
    
    # 해당 피사체 행 찾기
    symbol_row = topic_weights_df[topic_weights_df['라벨'] == symbol]
    
    if symbol_row.empty:
        return {}
    
    # 주제 컬럼들만 추출 (id, 대분류, 라벨, 별칭 제외)
    weights = {}
    for topic in config.TOPICS_KO:
        if topic in symbol_row.columns:
            weight = symbol_row[topic].values[0]
            if pd.notna(weight) and weight > 0:
                weights[topic] = float(weight)
    
    return weights

def convert_verse_to_dict(verse_row: pd.Series, topic_symbol_mapping: Dict[str, List[str]] = None) -> Dict:
    """
    verse_bank DataFrame의 한 행을 기존 VERSES_DB 형식의 딕셔너리로 변환
    
    Args:
        verse_row: verse_bank DataFrame의 한 행
        topic_symbol_mapping: 주제별 상징 매핑 딕셔너리
        
    Returns:
        기존 VERSES_DB 형식의 딕셔너리
    """
    topic_name = verse_row['topic_name']
    
    # 주제에 맞는 상징들 가져오기
    symbols = []
    if topic_symbol_mapping and topic_name in topic_symbol_mapping:
        symbols = topic_symbol_mapping[topic_name]
    
    return {
        "id": int(verse_row['topic_id']) * 1000 + verse_row.name,  # 고유 ID 생성
        "book": verse_row['book_ko'],
        "chapter": int(verse_row['chapter']),
        "verse": int(verse_row['verse']),
        "text": verse_row['text_ko'],
        "themes": [topic_name],  # 주제는 리스트로
        "symbols": symbols,  # 주제별 상징 매핑
        "context_fit": {}  # 나중에 매핑 필요
    }

def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    모든 데이터 파일을 한 번에 로드
    
    Returns:
        (bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping)
    """
    print("=" * 60)
    print("데이터 로딩 시작...")
    print("=" * 60)
    
    bible_objects_df = load_bible_objects()
    detection_labels_df = load_detection_labels()
    topic_weights_df = load_topic_weights()
    verse_bank_df = load_verse_bank()
    topic_symbol_mapping = load_topic_symbol_mapping()
    
    print("=" * 60)
    print("데이터 로딩 완료!")
    print("=" * 60)
    
    return bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping

if __name__ == "__main__":
    # 테스트 실행
    bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df = load_all_data()
    
    print("\n[피사체 카테고리]")
    categories = get_symbol_categories(bible_objects_df)
    for cat, symbols in categories.items():
        print(f"  {cat}: {len(symbols)}개")
    
    print("\n[샘플 피사체의 주제 가중치]")
    if not topic_weights_df.empty:
        sample_weights = get_topic_weights_for_symbol(topic_weights_df, "산")
        for topic, weight in list(sample_weights.items())[:5]:
            print(f"  {topic}: {weight}")
    
    print("\n[샘플 성경 구절]")
    if not verse_bank_df.empty:
        sample_verse = convert_verse_to_dict(verse_bank_df.iloc[0])
        print(f"  {sample_verse['book']} {sample_verse['chapter']}:{sample_verse['verse']}")
        print(f"  {sample_verse['text'][:50]}...")
