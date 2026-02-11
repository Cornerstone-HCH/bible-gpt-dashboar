"""
P1 상징 매핑 테스트 대시보드

SymbolMapper 모듈을 웹에서 테스트
"""

import streamlit as st
import sys
from pathlib import Path

# modules 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

from modules.symbol_mapper import SymbolMapper

# 페이지 설정
st.set_page_config(
    page_title="P1 상징 매핑 테스트",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 P1 상징 매핑 테스트")
st.markdown("---")

# SymbolMapper 초기화
@st.cache_resource
def load_mapper():
    return SymbolMapper()

mapper = load_mapper()

# 사이드바: 시나리오 선택
st.sidebar.header("📋 테스트 시나리오")

scenario = st.sidebar.selectbox(
    "시나리오 선택",
    [
        "SIM-01: 병상, 밤, 슬픔",
        "SIM-02: 교회, 아침, 기쁨",
        "SIM-03: 자연, 낮, 평화",
        "커스텀 입력"
    ]
)

# 시나리오별 기본값
scenarios = {
    "SIM-01: 병상, 밤, 슬픔": {
        'objects': [
            {'coco_class': 'bed', 'confidence': 0.92},
            {'coco_class': 'person', 'confidence': 0.85}
        ],
        'scene': {
            'location': ('병원', 0.85),
            'time': ('밤', 0.78),
            'mood': ('고요한', 0.72)
        },
        'emotion': {
            'primary_label': 'sad',
            'intensity': 0.65
        }
    },
    "SIM-02: 교회, 아침, 기쁨": {
        'objects': [
            {'coco_class': 'book', 'confidence': 0.88},
            {'coco_class': 'person', 'confidence': 0.90}
        ],
        'scene': {
            'location': ('교회', 0.92),
            'time': ('아침', 0.85),
            'mood': ('경건한', 0.80)
        },
        'emotion': {
            'primary_label': 'happy',
            'intensity': 0.75
        }
    },
    "SIM-03: 자연, 낮, 평화": {
        'objects': [
            {'coco_class': 'bird', 'confidence': 0.78},
            {'coco_class': 'potted plant', 'confidence': 0.82}
        ],
        'scene': {
            'location': ('자연', 0.88),
            'time': ('낮', 0.90),
            'mood': ('평화로운', 0.85)
        },
        'emotion': {
            'primary_label': 'neutral',
            'intensity': 0.60}
    }
}

# 메인 영역
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📥 입력 (Perception 결과)")
    
    if scenario == "커스텀 입력":
        st.subheader("1. 객체 인식 (COCO)")
        
        num_objects = st.number_input("객체 개수", 1, 5, 2)
        objects = []
        for i in range(num_objects):
            st.markdown(f"**객체 {i+1}**")
            col_obj1, col_obj2 = st.columns([2, 1])
            with col_obj1:
                coco_class = st.selectbox(
                    f"COCO 클래스 {i+1}",
                    ['bed', 'person', 'book', 'chair', 'laptop', 'bird', 'potted plant'],
                    key=f"coco_{i}"
                )
            with col_obj2:
                confidence = st.slider(f"신뢰도 {i+1}", 0.0, 1.0, 0.85, 0.05, key=f"conf_{i}")
            objects.append({'coco_class': coco_class, 'confidence': confidence})
        
        st.subheader("2. 장면 분석 (CLIP)")
        col_scene1, col_scene2 = st.columns([2, 1])
        with col_scene1:
            location = st.selectbox("장소", ['실내', '실외', '교회', '병원', '자연', '도시'])
        with col_scene2:
            loc_conf = st.slider("장소 신뢰도", 0.0, 1.0, 0.85, 0.05)
        
        col_time1, col_time2 = st.columns([2, 1])
        with col_time1:
            time = st.selectbox("시간", ['아침', '낮', '저녁', '밤'])
        with col_time2:
            time_conf = st.slider("시간 신뢰도", 0.0, 1.0, 0.78, 0.05)
        
        st.subheader("3. 감정 인식")
        col_emo1, col_emo2 = st.columns([2, 1])
        with col_emo1:
            emotion = st.selectbox("감정", ['happy', 'sad', 'angry', 'fear', 'neutral', 'surprise'])
        with col_emo2:
            emo_intensity = st.slider("강도", 0.0, 1.0, 0.65, 0.05)
        
        perception_result = {
            'objects': objects,
            'scene': {
                'location': (location, loc_conf),
                'time': (time, time_conf),
                'mood': ('', 0.0)
            },
            'emotion': {
                'primary_label': emotion,
                'intensity': emo_intensity
            }
        }
    else:
        # 시나리오 사용
        perception_result = scenarios[scenario]
        
        st.subheader("1. 객체 인식 (COCO)")
        for i, obj in enumerate(perception_result['objects'], 1):
            st.markdown(f"- **{obj['coco_class']}** (신뢰도: {obj['confidence']:.2f})")
        
        st.subheader("2. 장면 분석 (CLIP)")
        scene = perception_result['scene']
        st.markdown(f"- **장소**: {scene['location'][0]} ({scene['location'][1]:.2f})")
        st.markdown(f"- **시간**: {scene['time'][0]} ({scene['time'][1]:.2f})")
        
        st.subheader("3. 감정 인식")
        emotion = perception_result['emotion']
        st.markdown(f"- **감정**: {emotion['primary_label']} (강도: {emotion['intensity']:.2f})")

with col2:
    st.header("📤 출력 (상징 매핑 결과)")
    
    if st.button("🚀 상징 매핑 실행", type="primary"):
        with st.spinner("매핑 중..."):
            # 상징 매핑 실행
            symbols = mapper.map(perception_result)
            
            st.success(f"✅ 총 {len(symbols)}개 상징 추출 완료!")
            
            # 결과 표시
            for i, s in enumerate(symbols, 1):
                with st.expander(f"**{i}. {s['symbol']}** (신뢰도: {s['confidence']:.3f})", expanded=(i <= 3)):
                    col_info1, col_info2 = st.columns([1, 1])
                    
                    with col_info1:
                        st.markdown(f"**소스**: {s['source']}")
                        st.markdown(f"**코어 군**: {s['core_group']}")
                        st.markdown(f"**신뢰도**: {s['confidence']:.3f}")
                    
                    with col_info2:
                        # 상징 상세 정보
                        info = mapper.get_symbol_info(s['symbol'])
                        if info:
                            st.markdown(f"**의미**: {info['meaning']}")
                            st.markdown(f"**주요 주제**: {info['primary_themes']}")
                            st.markdown(f"**탐지 힌트**: {info['detection_hints']}")

# 하단: 상징 정의 테이블
st.markdown("---")
st.header("📊 46개 상징 전체 목록")

# 코어 군별 필터
core_group_filter = st.selectbox(
    "코어 군 필터",
    ['전체', '자연', '생명', '인공물', '인간활동', '공간장면']
)

if core_group_filter == '전체':
    filtered_symbols = mapper.symbol_defs
else:
    filtered_symbols = mapper.symbol_defs[mapper.symbol_defs['core_group'] == core_group_filter]

st.dataframe(
    filtered_symbols[['symbol_id', 'symbol', 'core_group', 'meaning', 'primary_themes']],
    use_container_width=True,
    height=400
)

st.markdown("---")
st.markdown("**💡 Tip**: 상단에서 시나리오를 선택하거나 커스텀 입력으로 직접 테스트해보세요!")
