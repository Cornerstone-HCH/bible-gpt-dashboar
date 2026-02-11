# 바이블GPT AI 추천 로직 검증 대시보드
# 기획서: 0211_바이블GPT통합본_v1.4_14.md

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import random

# 데이터 로더 및 설정 임포트
import config
import data_loader
import importlib
importlib.reload(data_loader)  # 개발 중 모듈 변경 반영을 위해 강제 리로드

# 페이지 설정
st.set_page_config(
    page_title="바이블GPT AI 추천 로직 검증 대시보드",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# 데이터 로딩 (실제 CSV 파일 기반)
# ============================================================================

# @st.cache_data  # 캐시 임시 비활성화
def load_data():
    """모든 데이터를 로드하고 캐싱"""
    bible_objects_df, detection_labels_df, topic_weights_df, verse_bank_df, topic_symbol_mapping = data_loader.load_all_data()
    
    # 피사체 카테고리 생성
    symbol_categories = data_loader.get_symbol_categories(bible_objects_df)
    
    # verse_bank를 기존 VERSES_DB 형식으로 변환 (주제별 상징 매핑 포함)
    verses_db = []
    for idx, row in verse_bank_df.iterrows():
        verse_dict = data_loader.convert_verse_to_dict(row, topic_symbol_mapping)
        verses_db.append(verse_dict)
    
    # 디버그: 침대가 포함된 구절 수 확인
    bed_count = sum(1 for v in verses_db if "침대" in v.get('symbols', []))
    print(f"\n[DEBUG] 생성된 VERSES_DB: 총 {len(verses_db)}개 구절")
    print(f"[DEBUG] '침대'가 포함된 구절: {bed_count}개")
    if verses_db:
        print(f"[DEBUG] 첫 번째 구절 symbols: {verses_db[0].get('symbols', [])[:5]}")
    
    return bible_objects_df, topic_weights_df, symbol_categories, verses_db

# 데이터 로드
bible_objects_df, topic_weights_df, SYMBOLS, VERSES_DB = load_data()

# 24개 신학적 주제 (config에서 가져옴)
THEMES = config.TOPICS_KO

# 시간대
TIME_BUCKETS = config.TIME_BUCKETS

# 장소 유형
PLACE_TYPES = config.PLACE_TYPES

# 부적절한 입력 패턴 (가이드라인 엔진)
INAPPROPRIATE_PATTERNS = config.INAPPROPRIATE_PATTERNS


# ============================================================================
# 핵심 로직 함수
# ============================================================================

def p1_symbol_mapping(selected_symbols: List[str]) -> Dict[str, float]:
    """
    P1: 이미지 상징 매핑
    선택된 상징들을 가중치와 함께 반환
    """
    symbol_scores = {}
    for symbol in selected_symbols:
        # 간단한 가중치 할당 (실제로는 매핑 매트릭스 사용)
        symbol_scores[symbol] = random.uniform(0.7, 1.0)
    return symbol_scores

def p2_theme_mapping(symbol_scores: Dict[str, float]) -> Dict[str, float]:
    """
    P2: 신학적 주제 매핑
    상징 점수를 24개 주제로 변환 (실제 가중치 매트릭스 사용)
    """
    theme_scores = {}
    
    # 각 피사체에 대해 가중치 매트릭스에서 주제 가중치를 가져옴
    for symbol, symbol_score in symbol_scores.items():
        weights = data_loader.get_topic_weights_for_symbol(topic_weights_df, symbol)
        
        for theme, weight in weights.items():
            if theme not in theme_scores:
                theme_scores[theme] = 0
            # 피사체 점수 × 주제 가중치
            theme_scores[theme] += symbol_score * weight
    
    # 정규화
    if theme_scores:
        max_score = max(theme_scores.values())
        if max_score > 0:
            theme_scores = {k: v/max_score for k, v in theme_scores.items()}
    
    return theme_scores


def p3_context_adjustment(theme_scores: Dict[str, float], 
                          time_bucket: str, 
                          place: str) -> Dict[str, float]:
    """
    P3: 컨텍스트 보정
    시간대와 장소에 따라 주제 점수 조정
    """
    adjusted_scores = theme_scores.copy()
    
    # 시간대별 보정
    time_adjustments = {
        "새벽": ["인도/소망", "예배/감사", "말씀/진리"],
        "밤": ["보호/피난처", "평강·샬롬"],
        "자정": ["보호/피난처", "신뢰/의지"]
    }
    
    if time_bucket in time_adjustments:
        for theme in time_adjustments[time_bucket]:
            if theme in adjusted_scores:
                # 새벽: x1.3, 밤/자정: 심판/경고는 S4에서 처리
                adjusted_scores[theme] *= 1.3
    
    # 자정/밤 심판/경고 억제 (기획서 8.3)
    if time_bucket in ["밤", "자정"] and "심판/경고" in adjusted_scores:
        adjusted_scores["심판/경고"] *= 0.2

    # 장소별 보정
    place_adjustments = {
        "병원": {
            "themes": ["평강·샬롬", "치유·회복"],
            "multiplier": 1.5
        },
        "교회": {
            "themes": ["예배/감사", "공동체/연합", "말씀/진리"],
            "multiplier": 1.3
        },
        "산": {
            "themes": ["창조/섭리", "영광/찬송"],
            "multiplier": 1.3
        }
    }
    
    if place in place_adjustments:
        adj = place_adjustments[place]
        for theme in adj["themes"]:
            if theme in adjusted_scores:
                adjusted_scores[theme] *= adj["multiplier"]
        
        # 특정 장소 위험 주제 추가 억제 (기획서 8.3)
        if place == "병원":
            for risky in ["심판/경고", "회개"]:
                if risky in adjusted_scores:
                    adjusted_scores[risky] *= 0.3
    
    return adjusted_scores

def calculate_s1_image_relevance(verse: Dict, symbols: List[str]) -> float:
    """S1: 이미지 관련도 점수"""
    verse_symbols = set(verse.get("symbols", []))
    selected_symbols = set(symbols)
    
    if not verse_symbols or not selected_symbols:
        return 0.0
    
    # Jaccard 유사도
    intersection = len(verse_symbols & selected_symbols)
    union = len(verse_symbols | selected_symbols)
    
    return intersection / union if union > 0 else 0.0

def calculate_s2_context_fit(verse: Dict, time_bucket: str, place: str) -> float:
    """S2: 컨텍스트 적합도 점수"""
    context_fit = verse.get("context_fit", {})
    
    # 장소와 시간대 적합도 평균
    place_score = context_fit.get(place, 0.5)
    time_score = context_fit.get(time_bucket, 0.5)
    
    return (place_score + time_score) / 2

def calculate_s3_orthodoxy(verse: Dict) -> float:
    """S3: 신학적 정합도 점수 (예장통합 교리 기준)"""
    # 간소화: 특정 주제가 있으면 높은 점수
    themes = verse.get("themes", [])
    
    # 위험한 주제 (심판/경고 단독 등)
    risky_themes = ["심판/경고"]
    
    # 안전한 주제
    safe_themes = ["평강·샬롬", "사랑(아가페)", "소망·부활", "은혜", "보호·인도", "치유·회복", "믿음·신뢰"]
    
    score = 0.7  # 기본 점수
    
    for theme in themes:
        if theme in safe_themes:
            score += 0.1
        elif theme in risky_themes:
            score -= 0.2
    
    return max(0.0, min(1.0, score))

def calculate_s4_penalty(verse: Dict, symbols: List[str], 
                         time_bucket: str, place: str) -> float:
    """S4: 페널티 점수"""
    penalty = 0.0
    
    # 민감한 조합 체크 (기획서 8.5)
    if "침대" in symbols and any(t in verse.get("themes", []) for t in ["심판/경고", "회개"]):
        penalty += 0.8  # 병상에서 심판/회개는 부적절
    
    # 밤 + 공포 조합
    if time_bucket in ["밤", "자정"] and "심판/경고" in verse.get("themes", []):
        penalty += 0.6
    
    # 눈물/고통 + 회개/정의 (기획서 8.5)
    if any(s in symbols for s in ["눈물", "고통"]) and any(t in verse.get("themes", []) for t in ["회개", "정의·공의"]):
        penalty += 0.5
    
    return min(1.0, penalty)

def check_guideline_filter(symbols: List[str], user_input: str = "") -> Tuple[bool, str]:
    """
    가이드라인 엔진: 부적절한 입력 감지
    Returns: (is_blocked, warning_message)
    """
    user_text = " ".join(symbols) + " " + user_input
    user_text = user_text.lower()
    
    for category, patterns in INAPPROPRIATE_PATTERNS.items():
        for pattern in patterns:
            if pattern in user_text:
                return True, f"⚠️ 부적절한 입력이 감지되었습니다: {category}\n\n이 앱은 성경 말씀을 통한 위로와 묵상을 돕기 위한 도구입니다.\n기복 신앙, 점괘식 해석, 저주 등의 목적으로 사용할 수 없습니다."
    
    return False, ""

def recommend_verses(symbols: List[str], time_bucket: str, place: str, 
                     weights: Dict[str, float]) -> List[Dict]:
    """
    전체 추천 파이프라인
    """
    # P1: 상징 매핑
    symbol_scores = p1_symbol_mapping(symbols)
    
    # P2: 주제 매핑
    theme_scores = p2_theme_mapping(symbol_scores)
    
    # P3: 컨텍스트 보정
    adjusted_themes = p3_context_adjustment(theme_scores, time_bucket, place)
    
    # 각 구절에 대해 점수 계산
    scored_verses = []
    for verse in VERSES_DB:
        s1 = calculate_s1_image_relevance(verse, symbols)
        s2 = calculate_s2_context_fit(verse, time_bucket, place)
        s3 = calculate_s3_orthodoxy(verse)
        s4 = calculate_s4_penalty(verse, symbols, time_bucket, place)
        
        # 주제 매칭 점수
        verse_themes = set(verse.get("themes", []))
        theme_match = sum(adjusted_themes.get(t, 0) for t in verse_themes)
        
        # 최종 점수 계산
        total_score = (
            weights['s1'] * s1 +
            weights['s2'] * s2 +
            weights['s3'] * s3 -
            weights['s4'] * s4 +
            0.2 * theme_match  # 주제 매칭 보너스
        )
        
        scored_verses.append({
            'verse': verse,
            'scores': {
                's1': s1,
                's2': s2,
                's3': s3,
                's4': s4,
                'total': total_score
            }
        })
    
    # 점수순 정렬
    scored_verses.sort(key=lambda x: x['scores']['total'], reverse=True)
    
    return scored_verses[:3]  # Top 3 반환

# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.title("📖 바이블GPT AI 추천 로직 검증 대시보드")
    st.markdown("---")
    
    # 사이드바: 가중치 설정
    st.sidebar.header("⚙️ S1~S4 가중치 설정")
    st.sidebar.markdown("기본값: 40:20:20:20")
    
    w1 = st.sidebar.slider("S1 (이미지 관련도)", 0.0, 1.0, 0.4, 0.05)
    w2 = st.sidebar.slider("S2 (컨텍스트 적합도)", 0.0, 1.0, 0.2, 0.05)
    w3 = st.sidebar.slider("S3 (신학적 정합도)", 0.0, 1.0, 0.2, 0.05)
    w4 = st.sidebar.slider("S4 (페널티)", 0.0, 1.0, 0.2, 0.05)
    
    weights = {'s1': w1, 's2': w2, 's3': w3, 's4': w4}
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**합계**: {sum(weights.values()):.2f}")
    
    # 기본 시나리오 버튼
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 기본 시나리오")
    if st.sidebar.button("SIM-01 로드 (고난 중 밤에 기도)"):
        st.session_state.sim01_loaded = True
    
    # 메인 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📥 입력 설정")
        
        # SIM-01 기본값 설정
        if 'sim01_loaded' in st.session_state and st.session_state.sim01_loaded:
            default_symbols = ["침대", "어둠"]  # 병상 → 침대로 변경 (실제 데이터에 존재)
            default_time = "자정"
            default_place = "병원"
            st.info("✅ SIM-01 시나리오가 로드되었습니다!")
        else:
            default_symbols = []
            default_time = "낮"
            default_place = "기타"
        
        # 이미지 상징 선택
        st.subheader("1️⃣ 이미지 상징 선택")
        all_symbols = []
        for category, items in SYMBOLS.items():
            all_symbols.extend(items)
        
        selected_symbols = st.multiselect(
            "이미지에서 인식된 상징을 선택하세요 (최대 5개)",
            all_symbols,
            default=default_symbols,
            max_selections=5
        )
        
        # 시간대 선택
        st.subheader("2️⃣ 시간대 선택")
        time_bucket = st.selectbox(
            "사진 촬영 시간대",
            TIME_BUCKETS,
            index=TIME_BUCKETS.index(default_time)
        )
        
        # 장소 선택
        st.subheader("3️⃣ 장소 선택")
        place = st.selectbox(
            "촬영 장소",
            PLACE_TYPES,
            index=PLACE_TYPES.index(default_place)
        )
        
        # 추가 입력 (가이드라인 필터 테스트용)
        st.subheader("4️⃣ 추가 입력 (선택)")
        user_input = st.text_input(
            "추가 컨텍스트 입력 (테스트용)",
            placeholder="예: 복권 당첨, 시험 합격 등"
        )
    
    with col2:
        st.header("📊 추천 결과")
        
        if st.button("🔍 말씀 추천 실행", type="primary"):
            if not selected_symbols:
                st.warning("⚠️ 최소 1개 이상의 이미지 상징을 선택해주세요.")
            else:
                # 가이드라인 필터 체크
                is_blocked, warning_msg = check_guideline_filter(selected_symbols, user_input)
                
                if is_blocked:
                    st.error(warning_msg)
                else:
                    # 추천 실행
                    with st.spinner("추천 중..."):
                        results = recommend_verses(selected_symbols, time_bucket, place, weights)
                    
                    st.success("✅ 추천 완료!")
                    
                    # 추천 구절 표시
                    for i, result in enumerate(results, 1):
                        verse = result['verse']
                        scores = result['scores']
                        
                        with st.expander(f"🏆 추천 {i}위 - {verse['book']} {verse['chapter']}:{verse['verse']}", expanded=(i==1)):
                            st.markdown(f"### {verse['text']}")
                            st.markdown(f"**출처**: {verse['book']} {verse['chapter']}장 {verse['verse']}절")
                            st.markdown(f"**주제**: {', '.join(verse['themes'])}")
                            
                            # 점수 시각화
                            st.markdown("#### 📈 점수 상세")
                            
                            score_df = pd.DataFrame({
                                '점수 항목': ['S1 (이미지)', 'S2 (컨텍스트)', 'S3 (신학)', 'S4 (페널티)'],
                                '점수': [scores['s1'], scores['s2'], scores['s3'], -scores['s4']],
                                '가중치': [weights['s1'], weights['s2'], weights['s3'], weights['s4']],
                                '가중 점수': [
                                    scores['s1'] * weights['s1'],
                                    scores['s2'] * weights['s2'],
                                    scores['s3'] * weights['s3'],
                                    -scores['s4'] * weights['s4']
                                ]
                            })
                            
                            # 막대 그래프
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                name='원점수',
                                x=score_df['점수 항목'],
                                y=score_df['점수'],
                                marker_color='lightblue'
                            ))
                            
                            fig.add_trace(go.Bar(
                                name='가중 점수',
                                x=score_df['점수 항목'],
                                y=score_df['가중 점수'],
                                marker_color='darkblue'
                            ))
                            
                            fig.update_layout(
                                title=f"점수 합산 과정 (총점: {scores['total']:.3f})",
                                barmode='group',
                                height=300
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_verse_{verse['id']}")
                            
                            # 점수 테이블
                            st.dataframe(score_df, use_container_width=True)
    
    # 하단: 파이프라인 설명
    st.markdown("---")
    st.header("🔄 AI 추천 파이프라인")
    
    pipeline_cols = st.columns(4)
    
    with pipeline_cols[0]:
        st.markdown("### P1: 상징 매핑")
        st.markdown("이미지 태그 → 120개 상징 라벨")
        st.markdown("**입력**: 이미지 상징")
        st.markdown("**출력**: 상징 점수")
    
    with pipeline_cols[1]:
        st.markdown("### P2: 주제 매핑")
        st.markdown("상징 → 24개 신학적 주제")
        st.markdown("**입력**: 상징 점수")
        st.markdown("**출력**: 주제 점수")
    
    with pipeline_cols[2]:
        st.markdown("### P3: 컨텍스트 보정")
        st.markdown("시간/장소로 주제 조정")
        st.markdown("**입력**: 주제 점수 + 컨텍스트")
        st.markdown("**출력**: 보정된 주제 점수")
    
    with pipeline_cols[3]:
        st.markdown("### Scoring Engine")
        st.markdown("S1~S4 점수 계산 및 합산")
        st.markdown("**입력**: 모든 신호")
        st.markdown("**출력**: 최종 추천 구절")

if __name__ == "__main__":
    main()
