"""
Phase 2.3: 데이터 수정 및 검증
신학적 안전 규칙에 맞게 데이터 수정
"""
import pandas as pd
import json
from datetime import datetime

print("=" * 80)
print("Phase 2.3: 데이터 수정 및 검증")
print("=" * 80)

# 1. 신학적 안전 규칙 로드
print("\n[1] 신학적 안전 규칙 로드...")
with open('theological_safety_rules.json', 'r', encoding='utf-8') as f:
    rules = json.load(f)

symbol_remapping = rules['symbol_remapping_suggestions']
print(f"  [OK] {len(symbol_remapping)}개 상징 재매핑 제안 로드")

# 2. topic_symbol_mapping.csv 수정
print("\n" + "=" * 80)
print("[2] topic_symbol_mapping.csv 수정")
print("=" * 80)

df = pd.read_csv('data/topic_symbol_mapping.csv', encoding='utf-8')

# 현재 매핑을 딕셔너리로 변환
topic_symbols = {}
for _, row in df.iterrows():
    topic = row['topic_name']
    symbols_str = row['symbols']
    symbols = [s.strip() for s in symbols_str.split(',')]
    topic_symbols[topic] = symbols

print("\n수정 전 상태:")
for symbol, suggestion in symbol_remapping.items():
    if suggestion['priority'] == 'high':
        current_topics = [t for t, syms in topic_symbols.items() if symbol in syms]
        print(f"  '{symbol}': {current_topics}")

# 수정 작업
modifications = []

# 1) '어둠' 재매핑: 회개, 정의·공의 → 평강·샬롬
if '어둠' in topic_symbols.get('회개', []):
    topic_symbols['회개'].remove('어둠')
    modifications.append("'어둠'을 '회개'에서 제거")

if '어둠' in topic_symbols.get('정의·공의', []):
    topic_symbols['정의·공의'].remove('어둠')
    modifications.append("'어둠'을 '정의·공의'에서 제거")

if '어둠' not in topic_symbols.get('평강·샬롬', []):
    topic_symbols['평강·샬롬'].append('어둠')
    modifications.append("'어둠'을 '평강·샬롬'에 추가")

# 2) '밤' 재매핑: 회개 → 평강·샬롬, 기도
if '밤' in topic_symbols.get('회개', []):
    topic_symbols['회개'].remove('밤')
    modifications.append("'밤'을 '회개'에서 제거")

if '밤' not in topic_symbols.get('평강·샬롬', []):
    topic_symbols['평강·샬롬'].append('밤')
    modifications.append("'밤'을 '평강·샬롬'에 추가")

if '밤' not in topic_symbols.get('기도', []):
    topic_symbols['기도'].append('밤')
    modifications.append("'밤'을 '기도'에 추가")

# 3) '눈물' 재매핑: 회개 → 치유·회복, 평강·샬롬
if '눈물' in topic_symbols.get('회개', []):
    topic_symbols['회개'].remove('눈물')
    modifications.append("'눈물'을 '회개'에서 제거")

if '눈물' not in topic_symbols.get('치유·회복', []):
    topic_symbols['치유·회복'].append('눈물')
    modifications.append("'눈물'을 '치유·회복'에 추가")

if '눈물' not in topic_symbols.get('평강·샬롬', []):
    topic_symbols['평강·샬롬'].append('눈물')
    modifications.append("'눈물'을 '평강·샬롬'에 추가")

print(f"\n총 {len(modifications)}개 수정 사항:")
for mod in modifications:
    print(f"  - {mod}")

# 수정된 데이터를 DataFrame으로 변환
modified_rows = []
for topic, symbols in topic_symbols.items():
    modified_rows.append({
        'topic_name': topic,
        'symbols': ', '.join(symbols)
    })

modified_df = pd.DataFrame(modified_rows)

# 백업 저장
backup_filename = f'data/topic_symbol_mapping_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(backup_filename, index=False, encoding='utf-8')
print(f"\n[OK] 백업 저장: {backup_filename}")

# 수정된 파일 저장
modified_df.to_csv('data/topic_symbol_mapping.csv', index=False, encoding='utf-8')
print("[OK] 수정된 파일 저장: data/topic_symbol_mapping.csv")

print("\n수정 후 상태:")
for symbol, suggestion in symbol_remapping.items():
    if suggestion['priority'] == 'high':
        current_topics = [t for t, syms in topic_symbols.items() if symbol in syms]
        print(f"  '{symbol}': {current_topics}")

# 3. 검증: SIM-01 재테스트
print("\n" + "=" * 80)
print("[3] SIM-01 시나리오 재테스트")
print("=" * 80)

import data_loader
import importlib
importlib.reload(data_loader)

bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping = data_loader.load_all_data()

# VERSES_DB 생성
verses_db = []
for idx, row in verse_bank_df.iterrows():
    verse_dict = data_loader.convert_verse_to_dict(row, topic_symbol_mapping)
    verses_db.append(verse_dict)

print(f"\n총 {len(verses_db)}개 구절 로드")

# '침대'가 포함된 구절 확인
input_symbols = ["침대", "어둠"]
print(f"\n입력 상징: {input_symbols}")

for symbol in input_symbols:
    matching_verses = [v for v in verses_db if symbol in v['symbols']]
    print(f"\n'{symbol}' 포함 구절: {len(matching_verses)}개")
    
    if matching_verses:
        # 주제별 분포
        topic_dist = {}
        for v in matching_verses:
            for theme in v['themes']:
                topic_dist[theme] = topic_dist.get(theme, 0) + 1
        
        print(f"  주제별 분포:")
        for topic, count in sorted(topic_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {topic}: {count}개")

# 4. 수정 요약 보고서 저장
print("\n" + "=" * 80)
print("[4] 수정 요약 보고서 저장")
print("=" * 80)

modification_report = {
    'timestamp': datetime.now().isoformat(),
    'phase': 'Phase 2.3 - 데이터 수정 및 검증',
    'modifications': modifications,
    'verification': {
        'total_verses': len(verses_db),
        'symbol_coverage': {
            symbol: len([v for v in verses_db if symbol in v['symbols']])
            for symbol in input_symbols
        }
    }
}

with open('data_modification_report.json', 'w', encoding='utf-8') as f:
    json.dump(modification_report, f, ensure_ascii=False, indent=2)

print("[OK] 수정 보고서 저장: data_modification_report.json")

print("\n" + "=" * 80)
print("Phase 2.3 완료")
print("=" * 80)
print("\n다음 단계: 대시보드에서 SIM-01 재실행하여 결과 확인")
print("예상 결과: '회개' 대신 '평강·샬롬', '보호·인도', '치유·회복' 주제 구절 추천")
