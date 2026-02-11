# 바이블GPT AI 추천 로직 검증 대시보드
# 기획서: 0211_바이블GPT통합본_v1.4_14.md

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import random
from PIL import Image

# 데이터 로더 및 설정 임포트
import config
import data_loader
from modules.image_pipeline import ImagePipeline

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

# @st.cache_data
@st.cache_data
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
# AI 파이프라인 지연 로딩 및 캐싱
# ============================================================================

@st.cache_resource
def get_ai_pipeline():
    """AI 파이프라인 모델들을 한 번만 로드하여 메모리에 고정"""
    return ImagePipeline()

def preprocess_image(image_file):
    """분석 속도 향상을 위해 이미지 리사이징"""
    from PIL import Image
    try:
        img = Image.open(image_file)
        # 최대 640px로 리사이징 (YOLO 최적 크기)
        img.thumbnail((640, 640))
        temp_path = "temp_process.jpg"
        img.save(temp_path, "JPEG", quality=85)
        return temp_path
    except Exception as e:
        print(f"[ERROR] Image preprocessing failed: {e}")
        # 리사이징 실패 시 원본 그대로 저장하여 진행
        with open("temp_process.jpg", "wb") as f:
            f.write(image_file.getbuffer())
        return "temp_process.jpg"

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
    
    # 메인 영역
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📥 입력 설정")
        
        # --- [NEW] 이미지 업로드 섹션 ---
        st.subheader("🖼️ 이미지 분석 (AI 자동 인식)")
        uploaded_file = st.file_uploader("사진 중 하나를 업로드하세요.", type=['jpg', 'jpeg', 'png'])
        
        ai_symbols = []
        ai_analysis = None
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption='업로드된 사진', use_container_width=True)
            
            with st.spinner("AI 엔진이 이미지를 분석 중입니다..."):
                try:
                    # 1. 이미지 최적화 (리사이징)
                    temp_path = preprocess_image(uploaded_file)
                    
                    # 2. 파이프라인 로드 (캐시 사용)
                    pipeline = get_ai_pipeline()
                    
                    # 3. AI 스마트 분석 수행 (내부에서 OCR 필요성 자동 판단)
                    # 상세 분석 리포트를 위해 개별 결과도 필요하지만, 
                    # 여기서는 통합 분석 기능을 우선 활용하거나 기존대로 개별 호출 가능
                    # 사용자 요청에 따라 '내부 로직'으로 숨기기 위해 pipeline.process 추천
                    
                    # 상세 리포트 시각화용 개별 데이터 추출 (Smart OCR 로직 대시보드 반영)
                    obj_res = pipeline.object_detector.detect(temp_path)
                    scene_res = pipeline.scene_analyzer.analyze(temp_path)
                    emo_res = pipeline.emotion_detector.detect(temp_path)
                    
                    # OCR은 ImagePipeline의 로직을 따라 자동 결정
                    ocr_res = []
                    text_bearing_objects = {'book', 'stop sign', 'traffic light', 'laptop', 'cell phone', 'tv'}
                    if ({obj['coco_class'] for obj in obj_res} & text_bearing_objects) or \
                       (scene_res.get('location', {}).get('label') == 'city street'):
                        ocr_res = pipeline.text_extractor.extract(temp_path)
                    
                    final_res = pipeline.integrator.integrate(obj_res, scene_res, emo_res, ocr_res)
                    ai_symbols = [s['symbol'] for s in final_res]
                    
                    # 시각화 데이터 저장
                    ai_analysis = {
                        'objects': obj_res,
                        'scene': scene_res,
                        'emotion': emo_res,
                        'ocr': ocr_res,
                        'final': final_res
                    }
                    
                    st.success("분석 완료!")
                    
                except Exception as e:
                    st.error(f"AI 분석 중 오류 발생: {e}")
        
        # AI 분석 결과 시각화 (Expander)
        if ai_analysis:
            with st.expander("🔍 AI 상세 분석 리포트 보기", expanded=False):
                tabs = st.tabs(["객체(YOLO)", "장면(CLIP)", "감정(MBV2)", "텍스트(OCR)"])
                
                with tabs[0]:
                    if ai_analysis['objects']:
                        for obj in ai_analysis['objects']:
                            st.write(f"- **{obj['coco_class']}** → {obj['bible_symbol']} (신뢰도: {obj['confidence']:.2f})")
                    else:
                        st.write("감지된 객체 없음")
                
                with tabs[1]:
                    s = ai_analysis['scene']
                    st.write(f"- 장소: **{s['location']['label']}** ({s['location']['confidence']:.2f})")
                    st.write(f"- 시간: **{s['time']['label']}** ({s['time']['confidence']:.2f})")
                    st.write(f"- 분위기: **{s['mood']['label']}** ({s['mood']['confidence']:.2f})")
                    if 'weather' in s:
                        st.write(f"- 날씨: **{s['weather']['label']}** ({s['weather']['confidence']:.2f})")
                
                with tabs[2]:
                    e = ai_analysis['emotion']
                    st.write(f"- 주 감정: **{e['primary_label']}** ({e['intensity']:.2f})")
                
                with tabs[3]:
                    if ai_analysis['ocr']:
                        for t in ai_analysis['ocr']:
                            st.write(f"- '{t['text']}' (Conf: {t['confidence']:.2f})")
                    else:
                        st.write("추출된 텍스트 없음")

        st.markdown("---")
        
        # 기본값 설정
        default_symbols = []
        default_time = "낮"
        default_place = "기타"
        
        # 이미지 상징 선택
        st.subheader("1️⃣ 이미지 상징 선택")
        all_symbols = []
        for category, items in SYMBOLS.items():
            all_symbols.extend(items)
        
        selected_symbols = st.multiselect(
            "인식된 상징 (자동 추출되거나 수동으로 선택할 수 있습니다)",
            all_symbols,
            default=[s for s in ai_symbols if s in all_symbols][:5] if ai_symbols else default_symbols,
            max_selections=5
        )
        
        # 시간대 선택 (AI 분석 결과 연동)
        st.subheader("2️⃣ 시간대 선택")
        scene_time = ai_analysis['scene']['time']['label'] if ai_analysis else "낮"
        # 영문 라벨 -> 한글 매핑 (CLIP 라벨 기반)
        time_map = {"dawn morning": "새벽", "bright daytime": "낮", "sunset evening": "저녁", "dark night": "밤", "midnight": "자정"}
        mapped_time = time_map.get(scene_time, "낮")
        
        selected_time = st.selectbox(
            "사진 촬영 시간대",
            TIME_BUCKETS,
            index=TIME_BUCKETS.index(mapped_time) if mapped_time in TIME_BUCKETS else 2
        )
        
        # 장소 선택 (AI 분석 결과 연동)
        st.subheader("3️⃣ 장소 선택")
        scene_loc = ai_analysis['scene']['location']['label'] if ai_analysis else "기타"
        loc_map = {"indoor space": "집", "outdoor plaza": "공원", "church sanctuary": "교회", "hospital room": "병원", "nature forest": "산", "sea side": "바다"}
        mapped_loc = loc_map.get(scene_loc, "기타")
        
        selected_place = st.selectbox(
            "촬영 장소",
            config.PLACE_TYPES,
            index=config.PLACE_TYPES.index(mapped_loc) if mapped_loc in config.PLACE_TYPES else 9
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
                        results = recommend_verses(selected_symbols, selected_time, selected_place, weights)
                    
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
