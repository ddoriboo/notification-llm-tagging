#!/usr/bin/env python3
"""
Quick Start Script - 가장 간단한 시작 방법
"""

import os
import sys


def main():
    print("""
    🚀 빠른 시작 가이드
    ==================
    
    이 프로젝트를 사용하는 방법:
    
    1️⃣ 기본 분석 실행:
       python main.py analyze --input your_data.csv
    
    2️⃣ LLM 태깅 실행:
       python main.py tag --input your_data.csv --sample-size 50
    
    3️⃣ 예측 모델 구축:
       python main.py predict
    
    4️⃣ 전체 파이프라인 실행:
       python main.py full --input your_data.csv --api-key YOUR_API_KEY
    
    📝 필수 사항:
    - OpenAI API 키 (태깅 기능 사용시)
    - CSV 데이터 파일 (컬럼: 발송 문구, 클릭율 등)
    
    💡 팁: 환경변수로 API 키 설정하기
       export OPENAI_API_KEY="your-api-key-here"
    
    자세한 사용법은 README.md를 참조하세요!
    """)
    
    # 샘플 데이터 존재 확인
    if os.path.exists('202507_.csv'):
        print("\n✅ 샘플 데이터 파일을 찾았습니다: 202507_.csv")
        print("   바로 시작하려면: python main.py analyze")
    else:
        print("\n⚠️  샘플 데이터 파일이 없습니다.")
        print("   CSV 파일을 준비한 후 --input 옵션으로 지정하세요.")


if __name__ == "__main__":
    main()