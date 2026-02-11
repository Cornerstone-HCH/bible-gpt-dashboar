
import pandas as pd
import config

print("=== 주제 일치 여부 확인 ===")
mapping_df = pd.read_csv('data/topic_symbol_mapping.csv', encoding='utf-8')
mapping_topics = mapping_df['topic_name'].tolist()

verse_bank_df = pd.read_csv('data/verse_bank_24topics_ko_seed_v1.csv', encoding='utf-8')
verse_topics = verse_bank_df['topic_name'].unique().tolist()

print(f"\nConfig TOPICS_KO ({len(config.TOPICS_KO)}개):")
print(config.TOPICS_KO)

print(f"\ntopic_symbol_mapping.csv Topics ({len(mapping_topics)}개):")
print(mapping_topics)

print(f"\nverse_bank_...csv Topics ({len(verse_topics)}개):")
print(verse_topics)

missing_in_mapping = set(config.TOPICS_KO) - set(mapping_topics)
print(f"\nConfig에는 있지만 Mapping에는 없는 주제: {missing_in_mapping}")

extra_in_mapping = set(mapping_topics) - set(config.TOPICS_KO)
print(f"Mapping에는 있지만 Config에는 없는 주제: {extra_in_mapping}")

missing_in_verse = set(config.TOPICS_KO) - set(verse_topics)
print(f"\nConfig에는 있지만 Verse Bank에는 없는 주제: {missing_in_verse}")

extra_in_verse = set(verse_topics) - set(config.TOPICS_KO)
print(f"Verse Bank에는 있지만 Config에는 없는 주제: {extra_in_verse}")
