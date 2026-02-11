"""
SIM-01 시나리오 자동 테스트 (수정 후 검증)
"""
import sys
sys.path.insert(0, 'e:/ODINA/ANTIGRAVITY')

from bible_gpt_dashboard import (
    recommend_verses,
    load_data
)

print("=" * 80)
print("SIM-01 시나리오 자동 테스트 (수정 후)")
print("=" * 80)

# 데이터 로드
print("\n[1] 데이터 로드 중...")
bible_objects_df, topic_weights_df, SYMBOLS, VERSES_DB = load_data()
print(f"  [OK] {len(VERSES_DB)}개 구절 로드")

# SIM-01 입력
print("\n[2] SIM-01 시나리오 설정")
symbols = ["침대", "어둠"]
time_bucket = "자정"
place = "병원"
weights = {'s1': 0.4, 's2': 0.2, 's3': 0.2, 's4': 0.2}

print(f"  상징: {symbols}")
print(f"  시간대: {time_bucket}")
print(f"  장소: {place}")
print(f"  가중치: S1={weights['s1']}, S2={weights['s2']}, S3={weights['s3']}, S4={weights['s4']}")

# 추천 실행
print("\n[3] 말씀 추천 실행 중...")
results = recommend_verses(symbols, time_bucket, place, weights)

# 결과 출력
print("\n" + "=" * 80)
print("추천 결과 (Top 3)")
print("=" * 80)

for i, result in enumerate(results, 1):
    verse = result['verse']
    scores = result['scores']
    
    print(f"\n[{i}위] {verse['book']} {verse['chapter']}:{verse['verse']}")
    print(f"   주제: {', '.join(verse['themes'])}")
    print(f"   본문: {verse['text'][:60]}...")
    print(f"   점수:")
    print(f"     - S1 (이미지): {scores['s1']:.4f}")
    print(f"     - S2 (컨텍스트): {scores['s2']:.4f}")
    print(f"     - S3 (신학): {scores['s3']:.4f}")
    print(f"     - S4 (페널티): {scores['s4']:.4f}")
    print(f"     - 총점: {scores['total']:.4f}")

# 검증
print("\n" + "=" * 80)
print("검증 결과")
print("=" * 80)

# 주제 분포 확인
themes_in_results = []
for result in results:
    themes_in_results.extend(result['verse']['themes'])

from collections import Counter
theme_counts = Counter(themes_in_results)

print("\n추천된 구절의 주제 분포:")
for theme, count in theme_counts.most_common():
    print(f"  - {theme}: {count}개")

# 기대 주제 체크
expected_themes = ["평강·샬롬", "치유·회복", "기도"]
unwanted_themes = ["회개", "심판·경고", "정의·공의"]

print("\n[기대 주제 포함 여부]")
for theme in expected_themes:
    if theme in themes_in_results:
        print(f"  [O] {theme}: 포함됨")
    else:
        print(f"  [X] {theme}: 없음")

print("\n[지양 주제 포함 여부]")
for theme in unwanted_themes:
    if theme in themes_in_results:
        print(f"  [!] {theme}: 포함됨 (문제!)")
    else:
        print(f"  [O] {theme}: 없음 (정상)")

# S1 점수 체크
print("\n[S1 점수 체크]")
for i, result in enumerate(results, 1):
    s1 = result['scores']['s1']
    if s1 > 0:
        print(f"  [O] {i}위: S1 = {s1:.4f} (정상)")
    else:
        print(f"  [X] {i}위: S1 = {s1:.4f} (문제!)")

print("\n" + "=" * 80)
print("테스트 완료")
print("=" * 80)
