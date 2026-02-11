import data_loader

# 데이터 로드
bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping = data_loader.load_all_data()

# VERSES_DB 생성
verses_db = []
for idx, row in verse_bank_df.iterrows():
    verse_dict = data_loader.convert_verse_to_dict(row, topic_symbol_mapping)
    verses_db.append(verse_dict)

print("\n=== SIM-01 시나리오 테스트 ===")
input_symbols = ["침대"]

print(f"\n입력 상징: {input_symbols}")

# 침대가 포함된 주제 찾기
print("\n'침대'가 포함된 주제:")
for topic, symbols in topic_symbol_mapping.items():
    if "침대" in symbols:
        print(f"  - {topic}: {symbols}")

# 침대가 symbols에 포함된 구절 찾기
print("\n'침대'가 symbols에 포함된 구절:")
matching_verses = [v for v in verses_db if "침대" in v['symbols']]
print(f"  총 {len(matching_verses)}개 구절")
if matching_verses:
    for v in matching_verses[:3]:
        print(f"  - {v['book']} {v['chapter']}:{v['verse']} (주제: {v['themes']})")
        print(f"    상징: {v['symbols'][:5]}...")

# S1 점수 계산 테스트
print("\n=== S1 점수 계산 테스트 ===")
def calculate_s1_image_relevance(verse, symbols):
    verse_symbols = set(verse.get("symbols", []))
    selected_symbols = set(symbols)
    
    if not verse_symbols or not selected_symbols:
        return 0.0
    
    intersection = len(verse_symbols & selected_symbols)
    union = len(verse_symbols | selected_symbols)
    
    return intersection / union if union > 0 else 0.0

# 매칭되는 구절의 S1 점수
if matching_verses:
    for v in matching_verses[:3]:
        s1 = calculate_s1_image_relevance(v, input_symbols)
        print(f"{v['book']} {v['chapter']}:{v['verse']} - S1: {s1:.3f}")
