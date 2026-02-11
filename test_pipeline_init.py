"""
ImagePipeline 통합 테스트 스크립트
"""

import os
from modules.image_pipeline import ImagePipeline

def test_pipeline_initialization():
    print("=== ImagePipeline 초기화 테스트 ===")
    try:
        pipeline = ImagePipeline()
        print("✓ 초기화 성공")
        return pipeline
    except Exception as e:
        print(f"✗ 초기화 실패: {e}")
        return None

if __name__ == "__main__":
    pipeline = test_pipeline_initialization()
    if pipeline:
        print("\n=== 모든 AI 모듈 로드 완료 ===")
        print("1. YOLOv8 (객체)")
        print("2. CLIP (장명/분위기)")
        print("3. MobileNetV2 (감정)")
        print("4. EasyOCR (텍스트)")
        print("5. Integrator (통합)")
