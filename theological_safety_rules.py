"""
Phase 2.2: 신학적 안전 규칙 데이터화
정본 MD의 신학적 안전 규칙을 구조화된 데이터로 변환
"""

# 정본 MD 기반 신학적 안전 규칙 정의

# ============================================================================
# 1. 금지 조합 규칙 (정본 MD 1.9 지양점 기반)
# ============================================================================

FORBIDDEN_COMBINATIONS = [
    {
        "name": "병상_심판경고_금지",
        "description": "병상/침대 상황에서 심판·경고 구절 금지",
        "condition": {
            "symbols": ["침대", "병상"],
            "topics": ["심판·경고", "정의·공의"]  # 정의·공의도 조심
        },
        "action": "suppress_topics",
        "severity": "high",
        "reference": "정본 MD 1.9 - 재난/사고 공포 지양"
    },
    {
        "name": "어둠_밤_심판_금지",
        "description": "밤/어둠 상황에서 심판·경고 단독 금지",
        "condition": {
            "symbols": ["어둠", "밤", "자정"],
            "topics": ["심판·경고"]
        },
        "action": "suppress_topics",
        "severity": "high",
        "reference": "정본 MD 1.9 - 심판/경고 단독 지양"
    },
    {
        "name": "고난_회개_단독_금지",
        "description": "고난 상황에서 회개 단독 강조 금지",
        "condition": {
            "symbols": ["눈물", "고통", "병상", "침대"],
            "topics": ["회개"]
        },
        "action": "require_balance",  # 회개 + 위로/평강 함께 제시
        "severity": "medium",
        "reference": "정본 MD 2.3.5 - 민감 주제 처리 원칙"
    }
]

# ============================================================================
# 2. 우선 주제 규칙 (정본 MD 2.3.5 민감 주제 처리 원칙)
# ============================================================================

PRIORITY_TOPICS_BY_CONTEXT = {
    "고난_상황": {
        "triggers": {
            "symbols": ["눈물", "고통", "병상", "침대", "어둠"],
            "places": ["병원"],
            "times": ["밤", "자정"]
        },
        "priority_topics": [
            "평강·샬롬",
            "보호·인도",
            "치유·회복",
            "사랑(아가페)",
            "소망·부활"
        ],
        "suppress_topics": [
            "심판·경고",
            "회개"  # 단독 사용 억제
        ],
        "reference": "정본 MD 2.3.5 - 하나님의 동행·위로·눈물·소망 중심"
    },
    "죽음_장례_상황": {
        "triggers": {
            "symbols": ["무덤", "관", "장례", "죽음"],
            "places": ["장례식장", "묘지"]
        },
        "priority_topics": [
            "소망·부활",
            "평강·샬롬",
            "사랑(아가페)",
            "보호·인도"
        ],
        "suppress_topics": [
            "심판·경고"
        ],
        "reference": "정본 MD 2.3.5 - 소망·부활 중심"
    },
    "재난_사고_상황": {
        "triggers": {
            "symbols": ["불", "물", "재난", "사고"],
            "contexts": ["재난", "사고"]
        },
        "priority_topics": [
            "보호·인도",
            "평강·샬롬",
            "치유·회복"
        ],
        "suppress_topics": [
            "심판·경고"
        ],
        "reference": "정본 MD 1.9 - 공포 증폭 아닌 위로·평강"
    }
}

# ============================================================================
# 3. 컨텍스트 보정 규칙 (시간대/장소별)
# ============================================================================

CONTEXT_ADJUSTMENTS = {
    "time_based": {
        "새벽": {
            "boost_topics": ["기도", "소망·부활", "말씀·진리"],
            "boost_factor": 1.3,
            "reference": "정본 MD - 새벽 기도 문화 반영"
        },
        "밤": {
            "boost_topics": ["평강·샬롬", "보호·인도", "기도"],
            "boost_factor": 1.2,
            "suppress_topics": ["심판·경고"],
            "suppress_factor": 0.3,
            "reference": "정본 MD 1.9 - 밤 + 공포 조합 지양"
        },
        "자정": {
            "boost_topics": ["평강·샬롬", "보호·인도", "믿음·신뢰"],
            "boost_factor": 1.3,
            "suppress_topics": ["심판·경고"],
            "suppress_factor": 0.2,
            "reference": "정본 MD - 깊은 밤 위로 중심"
        }
    },
    "place_based": {
        "병원": {
            "boost_topics": ["치유·회복", "평강·샬롬", "보호·인도"],
            "boost_factor": 1.5,
            "suppress_topics": ["심판·경고", "회개"],
            "suppress_factor": 0.3,
            "reference": "정본 MD 2.3.5 - 고난·질병 상황 위로 우선"
        },
        "교회": {
            "boost_topics": ["예배·찬양", "공동체·연합", "말씀·진리"],
            "boost_factor": 1.3,
            "reference": "정본 MD - 교회 공동체 강조"
        },
        "장례식장": {
            "boost_topics": ["소망·부활", "평강·샬롬", "사랑(아가페)"],
            "boost_factor": 1.5,
            "suppress_topics": ["심판·경고"],
            "suppress_factor": 0.1,
            "reference": "정본 MD 2.3.5 - 죽음·장례 소망 중심"
        }
    }
}

# ============================================================================
# 4. 주제별 안전도 등급 (S3 신학적 정합도 개선용)
# ============================================================================

TOPIC_SAFETY_GRADES = {
    # 안전 주제 (단독 사용 가능)
    "safe": {
        "topics": [
            "사랑(아가페)",
            "평강·샬롬",
            "소망·부활",
            "은혜",
            "보호·인도",
            "치유·회복",
            "믿음·신뢰"
        ],
        "base_score": 0.9,
        "description": "단독 사용 시에도 안전한 주제"
    },
    # 중립 주제 (문맥에 따라 사용)
    "neutral": {
        "topics": [
            "창조·섭리",
            "임재·영광",
            "거룩·성화",
            "기도",
            "말씀·진리",
            "예배·찬양",
            "순종·겸손",
            "인내·시험",
            "지혜·분별",
            "성령·열매",
            "공동체·연합",
            "소명·일·청지기",
            "가정·양육"
        ],
        "base_score": 0.7,
        "description": "문맥에 맞게 사용 가능한 주제"
    },
    # 주의 주제 (단독 사용 지양, 복음·소망과 함께)
    "caution": {
        "topics": [
            "회개",
            "정의·공의",
            "화해·용서(관계)"
        ],
        "base_score": 0.6,
        "description": "단독 사용 지양, 복음·소망 문맥 필요",
        "require_balance_with": ["은혜", "사랑(아가페)", "소망·부활"]
    },
    # 위험 주제 (매우 신중하게 사용)
    "risky": {
        "topics": [
            "심판·경고"  # 현재 데이터에는 없지만 향후 추가 시
        ],
        "base_score": 0.4,
        "description": "복음 문맥 없이 단독 사용 금지",
        "require_balance_with": ["구원·의(칭의)", "은혜", "소망·부활"]
    }
}

# ============================================================================
# 5. 상징 재매핑 제안 (정본 MD 원칙 기반)
# ============================================================================

SYMBOL_REMAPPING_SUGGESTIONS = {
    "어둠": {
        "current": ["회개", "정의·공의"],
        "suggested": ["평강·샬롬", "보호·인도", "믿음·신뢰"],
        "reason": "정본 MD 2.3.5 - 어둠은 고난 상황 암시, 위로 우선",
        "priority": "high"
    },
    "침대": {
        "current": ["평강·샬롬", "치유·회복"],
        "suggested": ["평강·샬롬", "치유·회복", "보호·인도"],
        "reason": "현재 매핑 적절, 보호·인도 추가 권장",
        "priority": "medium"
    },
    "밤": {
        "current": ["회개"],
        "suggested": ["평강·샬롬", "보호·인도", "기도"],
        "reason": "정본 MD - 밤은 기도·평강 문맥으로",
        "priority": "high"
    },
    "눈물": {
        "current": ["회개"],
        "suggested": ["치유·회복", "평강·샬롬", "사랑(아가페)"],
        "reason": "정본 MD 2.3.5 - 눈물은 위로·치유 중심",
        "priority": "high"
    }
}

# ============================================================================
# 6. 페널티 규칙 강화 (S4 개선용)
# ============================================================================

PENALTY_RULES = {
    "sensitive_combination": {
        "침대_심판": {
            "condition": {
                "symbols": ["침대", "병상"],
                "topics": ["심판·경고", "회개"]
            },
            "penalty": 0.8,  # 대폭 감점
            "reason": "병상에서 심판·회개는 부적절"
        },
        "밤_공포": {
            "condition": {
                "time": ["밤", "자정"],
                "topics": ["심판·경고"]
            },
            "penalty": 0.6,
            "reason": "밤 + 심판은 공포 증폭"
        },
        "고난_정죄": {
            "condition": {
                "symbols": ["눈물", "고통"],
                "topics": ["회개", "정의·공의"]
            },
            "penalty": 0.5,
            "reason": "고난 중 정죄는 2차 상처"
        }
    }
}

# ============================================================================
# 규칙 저장
# ============================================================================

if __name__ == "__main__":
    import json
    
    theological_rules = {
        "forbidden_combinations": FORBIDDEN_COMBINATIONS,
        "priority_topics_by_context": PRIORITY_TOPICS_BY_CONTEXT,
        "context_adjustments": CONTEXT_ADJUSTMENTS,
        "topic_safety_grades": TOPIC_SAFETY_GRADES,
        "symbol_remapping_suggestions": SYMBOL_REMAPPING_SUGGESTIONS,
        "penalty_rules": PENALTY_RULES
    }
    
    with open('theological_safety_rules.json', 'w', encoding='utf-8') as f:
        json.dump(theological_rules, f, ensure_ascii=False, indent=2)
    
    print("=" * 80)
    print("신학적 안전 규칙 데이터화 완료")
    print("=" * 80)
    print("\n저장 파일: theological_safety_rules.json")
    print("\n주요 규칙:")
    print(f"  - 금지 조합: {len(FORBIDDEN_COMBINATIONS)}개")
    print(f"  - 우선 주제 컨텍스트: {len(PRIORITY_TOPICS_BY_CONTEXT)}개")
    print(f"  - 시간대별 보정: {len(CONTEXT_ADJUSTMENTS['time_based'])}개")
    print(f"  - 장소별 보정: {len(CONTEXT_ADJUSTMENTS['place_based'])}개")
    print(f"  - 주제 안전도 등급: {len(TOPIC_SAFETY_GRADES)}개 등급")
    print(f"  - 상징 재매핑 제안: {len(SYMBOL_REMAPPING_SUGGESTIONS)}개")
    print(f"  - 페널티 규칙: {len(PENALTY_RULES['sensitive_combination'])}개")
    
    print("\n다음 단계: Phase 2.3 - 데이터 수정 및 검증")
