
import pandas as pd
import config
import json

mapping_df = pd.read_csv('data/topic_symbol_mapping.csv', encoding='utf-8')
mapping_topics = mapping_df['topic_name'].tolist()
config_topics = config.TOPICS_KO

print(f"Config count: {len(config_topics)}")
print(f"Mapping count: {len(mapping_topics)}")

# 주제들을 정렬해서 비교 (유사한 것끼리 매칭해보기 위해)
config_sorted = sorted(config_topics)
mapping_sorted = sorted(mapping_topics)

with open('topic_comparison.json', 'w', encoding='utf-8') as f:
    json.dump({
        "config": config_sorted,
        "mapping": mapping_sorted
    }, f, ensure_ascii=False, indent=2)

print("\ntopic_comparison.json에 저장되었습니다.")
