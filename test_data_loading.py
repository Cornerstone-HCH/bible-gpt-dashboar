import data_loader

# 데이터 로드
bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping = data_loader.load_all_data()

print("\n=== 주제별 상징 매핑 샘플 ===")
for topic, symbols in list(topic_symbol_mapping.items())[:5]:
    print(f"{topic}: {symbols[:5]}...")  # 처음 5개만

print("\n=== 첫 번째 구절 변환 테스트 ===")
verse = data_loader.convert_verse_to_dict(verse_bank_df.iloc[0], topic_symbol_mapping)
print(f"주제: {verse['themes']}")
print(f"상징 개수: {len(verse['symbols'])}")
print(f"상징 샘플: {verse['symbols'][:5]}")

print("\n=== VERSES_DB 형식으로 변환 (처음 3개) ===")
verses_db = []
for idx, row in verse_bank_df.head(3).iterrows():
    verse_dict = data_loader.convert_verse_to_dict(row, topic_symbol_mapping)
    verses_db.append(verse_dict)
    print(f"\n구절 {idx+1}:")
    print(f"  주제: {verse_dict['themes']}")
    print(f"  상징 개수: {len(verse_dict['symbols'])}")
    print(f"  상징: {verse_dict['symbols'][:3]}...")
