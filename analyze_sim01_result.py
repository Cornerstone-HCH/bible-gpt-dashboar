"""
SIM-01 시나리오에서 '회개' 주제가 나온 이유 분석
"""
import data_loader
import pandas as pd

# 데이터 로드
bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping = data_loader.load_all_data()

print("=== SIM-01 입력 분석 ===")
input_symbols = ["침대", "어둠"]
print(f"입력 상징: {input_symbols}")

# 1. 각 상징이 어떤 주제와 연결되는지 확인
print("\n=== 상징별 주제 매핑 (topic_symbol_mapping.csv) ===")
for topic, symbols in topic_symbol_mapping.items():
    for input_sym in input_symbols:
        if input_sym in symbols:
            print(f"'{input_sym}' → 주제: {topic}")
            print(f"  해당 주제의 모든 상징: {symbols}")

# 2. topic_weights_v1.csv에서 가중치 확인
print("\n=== 상징별 주제 가중치 (topic_weights_v1.csv) ===")
for symbol in input_symbols:
    weights = data_loader.get_topic_weights_for_symbol(topic_weights_df, symbol)
    if weights:
        print(f"\n'{symbol}'의 주제 가중치 (상위 5개):")
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        for topic, weight in sorted_weights:
            print(f"  {topic}: {weight}")
    else:
        print(f"\n'{symbol}': 가중치 없음")

# 3. '회개' 주제의 구절들 확인
print("\n=== '회개' 주제 구절 분석 ===")
repentance_verses = verse_bank_df[verse_bank_df['topic_name'] == '회개']
print(f"총 {len(repentance_verses)}개 구절")

# 회개 주제에 연결된 상징들
repentance_symbols = topic_symbol_mapping.get('회개', [])
print(f"\n'회개' 주제와 연결된 상징들:")
print(repentance_symbols)

# 입력 상징과의 교집합
overlap = set(input_symbols) & set(repentance_symbols)
print(f"\n입력 상징과 '회개' 주제 상징의 교집합: {overlap}")

# 4. 실제 추천된 구절들 확인
print("\n=== 추천된 '회개' 구절들 ===")
recommended_refs = [
    ("고린도후서", 12, 21),
    ("고린도후서", 7, 10),
    ("고린도후서", 7, 9)
]

for book, chapter, verse in recommended_refs:
    matching = verse_bank_df[
        (verse_bank_df['book_ko'] == book) & 
        (verse_bank_df['chapter'] == chapter) & 
        (verse_bank_df['verse'] == verse)
    ]
    if not matching.empty:
        row = matching.iloc[0]
        print(f"\n{book} {chapter}:{verse}")
        print(f"  본문: {row['text_ko'][:50]}...")
        print(f"  주제: {row['topic_name']}")
