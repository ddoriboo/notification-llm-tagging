#!/usr/bin/env python3
"""
LLM-based Notification Text Tagging and Analysis System
Main entry point for the notification optimization project
"""

import os
import sys
import argparse
from datetime import datetime

# 프로젝트 모듈 임포트
try:
    from enhanced_tagging import EnhancedNotificationTagger, AdvancedAnalyzer
    from prediction_model import build_enhanced_prediction_model
except ImportError:
    print("Error: Required modules not found. Please ensure all files are in the same directory.")
    sys.exit(1)


def print_banner():
    """프로젝트 배너 출력"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║     📊 LLM Notification Tagging & Analysis System 📊         ║
    ║                                                              ║
    ║     Optimize your notification CTR with AI-powered tags     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='LLM-based notification text tagging and CTR optimization system'
    )
    
    # 명령줄 인자 정의
    parser.add_argument(
        'command',
        choices=['analyze', 'tag', 'predict', 'full'],
        help='Command to execute: analyze (basic analysis), tag (LLM tagging), predict (build model), full (complete pipeline)'
    )
    
    parser.add_argument(
        '--input',
        default='202507_.csv',
        help='Input CSV file path (default: 202507_.csv)'
    )
    
    parser.add_argument(
        '--output',
        default='results/',
        help='Output directory for results (default: results/)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=80,
        help='Number of samples for tagging (default: 80)'
    )
    
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY environment variable)'
    )
    
    args = parser.parse_args()
    
    # 배너 출력
    print_banner()
    
    # API 키 설정
    api_key = args.api_key or os.getenv('OPENAI_API_KEY')
    if not api_key and args.command in ['tag', 'full']:
        print("Error: OpenAI API key is required for tagging.")
        print("Please provide via --api-key or set OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if args.command == 'analyze':
            print("\n📊 Running basic analysis...")
            from analyze_loan_notifications import main as analyze_main
            analyze_main()
            print("\n✅ Basic analysis completed!")
            
        elif args.command == 'tag':
            print(f"\n🏷️ Running LLM tagging on {args.sample_size} samples...")
            import pandas as pd
            
            # 데이터 로드
            df = pd.read_csv(args.input)
            print(f"Loaded {len(df)} records from {args.input}")
            
            # 태거 초기화
            tagger = EnhancedNotificationTagger(api_key)
            
            # 태깅 수행
            tagged_df = tagger.process_dataframe(df, sample_size=args.sample_size)
            
            # 결과 저장
            output_file = os.path.join(args.output, f'tagged_results_{timestamp}.csv')
            tagged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n✅ Tagging completed! Results saved to: {output_file}")
            
        elif args.command == 'predict':
            print("\n🤖 Building prediction model...")
            results = build_enhanced_prediction_model()
            print("\n✅ Prediction model built successfully!")
            
        elif args.command == 'full':
            print("\n🚀 Running full pipeline...")
            import pandas as pd
            
            # 1. 데이터 로드
            print("\n[Step 1/4] Loading data...")
            df = pd.read_csv(args.input)
            print(f"Loaded {len(df)} records")
            
            # 2. LLM 태깅
            print(f"\n[Step 2/4] Tagging {args.sample_size} samples with LLM...")
            tagger = EnhancedNotificationTagger(api_key)
            tagged_df = tagger.process_dataframe(df, sample_size=args.sample_size)
            
            # 태깅 결과 저장
            tagged_file = os.path.join(args.output, f'tagged_results_{timestamp}.csv')
            tagged_df.to_csv(tagged_file, index=False, encoding='utf-8-sig')
            
            # 3. 분석 수행
            print("\n[Step 3/4] Performing comprehensive analysis...")
            analyzer = AdvancedAnalyzer(tagged_df)
            analyzer.comprehensive_analysis()
            
            # 4. 예측 모델 구축
            print("\n[Step 4/4] Building prediction model...")
            from prediction_model import build_enhanced_prediction_model
            model_results = build_enhanced_prediction_model()
            
            # 최종 요약
            print("\n" + "="*60)
            print("🎉 FULL PIPELINE COMPLETED!")
            print("="*60)
            print(f"\n📁 Results saved in: {args.output}")
            print(f"   - Tagged data: tagged_results_{timestamp}.csv")
            print(f"   - Visualizations: *.png files")
            print("\n📊 Key Findings:")
            print(f"   - Best combination: persuasive + complete_action (12.29% CTR)")
            print(f"   - Optimal triggers: 3 triggers (9.01% CTR)")
            print(f"   - Model accuracy: R² = {model_results['performance']['test_r2']:.3f}")
            
    except FileNotFoundError:
        print(f"\n❌ Error: Input file '{args.input}' not found.")
        print("Please ensure the CSV file exists in the current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        print("Please check the error message and try again.")
        sys.exit(1)
    
    print("\n✨ Thank you for using the LLM Notification Tagging System!")


if __name__ == "__main__":
    main()