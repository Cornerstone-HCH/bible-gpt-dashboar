"""
Phase 2.1: 데이터 감사 및 문서화
전체 데이터 현황을 파악하고 문제점을 체계적으로 정리
"""
import pandas as pd
import json
from collections import defaultdict
import config

print("=" * 80)
print("바이블GPT 데이터 감사 보고서")
print("=" * 80)

# 1. 데이터 파일 로드
print("\n[1] 데이터 파일 로드 중...")
bible_objects_df = pd.read_csv('data/Bible_Object_Definition_120_v1.csv', encoding='utf-8')
topic_weights_df = pd.read_csv('data/topic_weights_v1.csv', encoding='utf-8')
topic_symbol_mapping_df = pd.read_csv('data/topic_symbol_mapping.csv', encoding='utf-8')
verse_bank_df = pd.read_csv('data/verse_bank_24topics_ko_seed_v1.csv', encoding='utf-8')

print(f"  [OK] Bible_Object_Definition: {len(bible_objects_df)}개 피사체")
print(f"  [OK] topic_weights: {len(topic_weights_df)}개 행")
print(f"  [OK] topic_symbol_mapping: {len(topic_symbol_mapping_df)}개 주제")
print(f"  [OK] verse_bank: {len(verse_bank_df)}개 구절")

# 2. topic_symbol_mapping.csv 분석
print("\n" + "=" * 80)
print("[2] topic_symbol_mapping.csv 분석")
print("=" * 80)

topic_symbol_dict = {}
for _, row in topic_symbol_mapping_df.iterrows():
    topic = row['topic_name']
    symbols_str = row['symbols']
    symbols = [s.strip() for s in symbols_str.split(',')]
    topic_symbol_dict[topic] = symbols

print(f"\n총 {len(topic_symbol_dict)}개 주제")
print("\n주제별 상징 개수:")
for topic, symbols in sorted(topic_symbol_dict.items(), key=lambda x: len(x[1]), reverse=True):
    print(f"  {topic}: {len(symbols)}개 상징")

# 민감 상징 체크
sensitive_symbols = ['침대', '어둠', '밤', '병상', '눈물', '고통', '죽음', '무덤', '관', '피']
print(f"\n민감 상징({', '.join(sensitive_symbols)}) 매핑 현황:")
for symbol in sensitive_symbols:
    mapped_topics = [topic for topic, syms in topic_symbol_dict.items() if symbol in syms]
    if mapped_topics:
        print(f"  '{symbol}' → {mapped_topics}")
    else:
        print(f"  '{symbol}' → (매핑 없음)")

# 3. topic_weights_v1.csv 완성도 분석
print("\n" + "=" * 80)
print("[3] topic_weights_v1.csv 완성도 분석")
print("=" * 80)

# 주제 컬럼 추출 (id, 대분류, 라벨, 별칭 제외)
topic_columns = [col for col in topic_weights_df.columns if col not in ['id', 'Unnamed: 0', '대분류', '라벨', '별칭']]
print(f"\n주제 컬럼 수: {len(topic_columns)}개")

# 각 피사체별 가중치 완성도 체크
missing_weights = []
for idx, row in topic_weights_df.iterrows():
    label = row['라벨']
    weights = row[topic_columns]
    null_count = weights.isna().sum()
    zero_count = (weights == 0).sum()
    
    if null_count > len(topic_columns) * 0.5:  # 50% 이상 누락
        missing_weights.append({
            'label': label,
            'null_count': null_count,
            'zero_count': zero_count,
            'total': len(topic_columns)
        })

print(f"\n가중치 50% 이상 누락된 피사체: {len(missing_weights)}개")
if missing_weights:
    print("\n상위 10개:")
    for item in missing_weights[:10]:
        print(f"  {item['label']}: {item['null_count']}/{item['total']} 누락, {item['zero_count']}/{item['total']} 제로")

# 민감 상징의 가중치 체크
print(f"\n민감 상징의 가중치 현황:")
for symbol in sensitive_symbols:
    symbol_row = topic_weights_df[topic_weights_df['라벨'] == symbol]
    if not symbol_row.empty:
        weights = symbol_row.iloc[0][topic_columns]
        null_count = weights.isna().sum()
        nonzero_count = (weights > 0).sum()
        print(f"  '{symbol}': {nonzero_count}개 주제에 가중치 있음 (누락: {null_count}개)")
    else:
        print(f"  '{symbol}': topic_weights_v1.csv에 없음")

# 4. verse_bank 주제별 분포 분석
print("\n" + "=" * 80)
print("[4] verse_bank 주제별 구절 분포")
print("=" * 80)

topic_verse_counts = verse_bank_df['topic_name'].value_counts()
print(f"\n총 {len(topic_verse_counts)}개 주제에 구절 분포")
print("\n주제별 구절 수 (상위 10개):")
for topic, count in topic_verse_counts.head(10).items():
    print(f"  {topic}: {count}개")

print("\n주제별 구절 수 (하위 10개):")
for topic, count in topic_verse_counts.tail(10).items():
    print(f"  {topic}: {count}개")

# 민감 주제 체크
sensitive_topics = ['회개', '심판·경고', '죄·타락', '고난·시련']
print(f"\n민감 주제 구절 수:")
for topic in sensitive_topics:
    if topic in topic_verse_counts:
        count = topic_verse_counts[topic]
        percentage = (count / len(verse_bank_df)) * 100
        print(f"  {topic}: {count}개 ({percentage:.1f}%)")
    else:
        print(f"  {topic}: 0개")

# 5. 정본 MD 원칙 위반 가능성 체크
print("\n" + "=" * 80)
print("[5] 정본 MD 원칙 위반 가능성 체크")
print("=" * 80)

# 정본 MD 2.3.5: 고난·죽음·장례·우울·자살 → 위로·평강 우선
# 정본 MD 1.9: 심판/경고 단독 지양
violations = []

# 체크 1: '어둠'이 '회개'와 매핑되어 있는지
if '회개' in topic_symbol_dict and '어둠' in topic_symbol_dict['회개']:
    violations.append({
        'type': '부적절한 상징-주제 매핑',
        'detail': "'어둠' → '회개' 매핑은 정본 MD 2.3.5 원칙 위반 가능성",
        'recommendation': "'어둠' → '평강·샬롬' 또는 '보호·인도'로 재매핑"
    })

# 체크 2: '침대'가 '회개'와 매핑되어 있는지
if '회개' in topic_symbol_dict and '침대' in topic_symbol_dict['회개']:
    violations.append({
        'type': '부적절한 상징-주제 매핑',
        'detail': "'침대' → '회개' 매핑은 병상 상황에서 부적절",
        'recommendation': "'침대' → '치유·회복' 또는 '평강·샬롬'으로 재매핑"
    })

# 체크 3: 민감 상징이 심판/경고와 매핑되어 있는지
if '심판·경고' in topic_symbol_dict:
    for symbol in sensitive_symbols:
        if symbol in topic_symbol_dict['심판·경고']:
            violations.append({
                'type': '부적절한 상징-주제 매핑',
                'detail': f"'{symbol}' → '심판·경고' 매핑은 정본 MD 1.9 지양점 위반",
                'recommendation': f"'{symbol}'를 심판·경고에서 제거"
            })

print(f"\n발견된 잠재적 위반 사항: {len(violations)}건")
for i, v in enumerate(violations, 1):
    print(f"\n{i}. {v['type']}")
    print(f"   문제: {v['detail']}")
    print(f"   권장: {v['recommendation']}")

# 6. 보고서 저장
print("\n" + "=" * 80)
print("[6] 보고서 저장")
print("=" * 80)

report = {
    'summary': {
        'total_symbols': len(bible_objects_df),
        'total_topics': len(topic_symbol_dict),
        'total_verses': len(verse_bank_df),
        'missing_weights_count': len(missing_weights),
        'violations_count': len(violations)
    },
    'topic_symbol_mapping': {
        topic: symbols for topic, symbols in topic_symbol_dict.items()
    },
    'topic_verse_distribution': topic_verse_counts.to_dict(),
    'missing_weights': missing_weights,
    'violations': violations
}

with open('data_audit_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n[OK] 보고서 저장 완료: data_audit_report.json")

print("\n" + "=" * 80)
print("데이터 감사 완료")
print("=" * 80)
print("\n다음 단계: Phase 2.2 - 신학적 안전 규칙 데이터화")
