import data_loader
import config

# 데이터 로드
bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping = data_loader.load_all_data()

# VERSES_DB 생성
verses_db = []
for idx, row in verse_bank_df.iterrows():
    verse_dict = data_loader.convert_verse_to_dict(row, topic_symbol_mapping)
    verses_db.append(verse_dict)

print(f"\n=== 전체 구절 수: {len(verses_db)} ===")

# SIM-01 입력
input_symbols = ["침대"]
time_bucket = "자정"
place = "병원"

print(f"\n입력: {input_symbols}, {time_bucket}, {place}")

# 각 주제별 구절 수 확인
from collections import Counter
theme_counts = Counter()
for v in verses_db:
    for theme in v['themes']:
        theme_counts[theme] += 1

print("\n=== 주제별 구절 수 ===")
for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"{theme}: {count}개")

# "침대"가 포함된 구절 찾기
print("\n=== '침대'가 symbols에 포함된 구절 ===")
matching_verses = [v for v in verses_db if "침대" in v.get('symbols', [])]
print(f"총 {len(matching_verses)}개")

if matching_verses:
    print("\n샘플:")
    for v in matching_verses[:5]:
        print(f"  {v['book']} {v['chapter']}:{v['verse']}")
        print(f"    주제: {v['themes']}")
        print(f"    상징 샘플: {v['symbols'][:5]}")
else:
    print("\n⚠️ 문제: '침대'가 포함된 구절이 하나도 없습니다!")
    print("\n첫 10개 구절의 symbols 확인:")
    for i, v in enumerate(verses_db[:10]):
        print(f"\n구절 {i+1}: {v['book']} {v['chapter']}:{v['verse']}")
        print(f"  주제: {v['themes']}")
        print(f"  symbols 타입: {type(v['symbols'])}")
        print(f"  symbols 길이: {len(v['symbols'])}")
        print(f"  symbols 샘플: {v['symbols'][:3] if v['symbols'] else '비어있음'}")
