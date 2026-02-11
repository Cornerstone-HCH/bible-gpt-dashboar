# 바이블GPT 대시보드 설정 파일

import os
from pathlib import Path

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent

# 데이터 파일 경로
DATA_DIR = PROJECT_ROOT / "data"

# CSV 파일 경로
BIBLE_OBJECTS_FILE = DATA_DIR / "Bible_Object_Definition_120_v1.csv"
DETECTION_LABELS_FILE = DATA_DIR / "detection_labels_map.csv"
TOPIC_WEIGHTS_FILE = DATA_DIR / "topic_weights_v1.csv"
VERSE_BANK_FILE = DATA_DIR / "verse_bank_24topics_ko_seed_v1.csv"

# 시스템 상수
NUM_OBJECTS = 120  # 피사체(상징) 개수
NUM_TOPICS = 24    # 신학적 주제 개수

# 주제 리스트 (한국어)
TOPICS_KO = [
    "창조·섭리", "임재·영광", "거룩·성화", "은혜",
    "믿음·신뢰", "회개", "구원·의(칭의)", "소망·부활",
    "말씀·진리", "기도", "예배·찬양", "순종·겸손",
    "사랑(아가페)", "인내·시험", "지혜·분별", "성령·열매",
    "평강·샬롬", "공동체·연합", "정의·공의", "화해·용서(관계)",
    "소명·일·청지기", "가정·양육", "치유·회복", "보호·인도"
]

# 시간대
TIME_BUCKETS = ["새벽", "아침", "낮", "저녁", "밤", "자정"]

# 장소 유형
PLACE_TYPES = ["집", "교회", "병원", "학교", "직장", "거리", "공원", "산", "바다", "기타"]

# 부적절한 입력 패턴 (가이드라인 엔진)
INAPPROPRIATE_PATTERNS = {
    "기복신앙": ["복권", "로또", "대박", "부자", "돈"],
    "저주/혐오": ["저주", "죽어라", "망해라", "증오"],
    "점괘식": ["운세", "점", "미래예측", "당첨"],
    "극단해석": ["무조건", "반드시", "100%"]
}
