"""
주제 이름 매핑 문제 해결 스크립트
topic_symbol_mapping.csv의 주제명을 config.py의 TOPICS_KO와 일치시킴
"""
import pandas as pd
import config

# 주제명 매핑 테이블 (mapping.csv -> config.py)
TOPIC_NAME_MAPPING = {
    "평안·안식": "평강·샬롬",
    "회복·치유": "치유·회복",
    "사랑·친교": "사랑(아가페)",
    "용서·화해": "화해·용서(관계)",
    "공동체·교회": "공동체·연합",
    "인내·견딤": "인내·시험",
    "죄·타락": "회개",  # 가장 가까운 주제로 매핑
    "심판·경고": "정의·공의",  # 가장 가까운 주제로 매핑
    "고난·시련": "인내·시험",  # 가장 가까운 주제로 매핑
    "기쁨·감사": "예배·찬양",  # 가장 가까운 주제로 매핑
    "선교·증거": "소명·일·청지기",  # 가장 가까운 주제로 매핑
    "종말·재림": "소망·부활",  # 가장 가까운 주제로 매핑
}

# topic_symbol_mapping.csv 로드
mapping_df = pd.read_csv('data/topic_symbol_mapping.csv', encoding='utf-8')

print("=== 주제명 변환 전 ===")
print(mapping_df['topic_name'].tolist())

# 주제명 변환
mapping_df['topic_name'] = mapping_df['topic_name'].replace(TOPIC_NAME_MAPPING)

print("\n=== 주제명 변환 후 ===")
print(mapping_df['topic_name'].tolist())

# Config와 비교
config_topics = set(config.TOPICS_KO)
mapping_topics = set(mapping_df['topic_name'].tolist())

print(f"\n=== 검증 ===")
print(f"Config 주제 수: {len(config_topics)}")
print(f"Mapping 주제 수: {len(mapping_topics)}")
print(f"일치하지 않는 주제: {mapping_topics - config_topics}")

# 백업 저장
mapping_df.to_csv('data/topic_symbol_mapping_backup.csv', index=False, encoding='utf-8')
print("\n백업 저장: topic_symbol_mapping_backup.csv")

# 수정된 파일 저장
mapping_df.to_csv('data/topic_symbol_mapping.csv', index=False, encoding='utf-8')
print("수정 완료: topic_symbol_mapping.csv")
