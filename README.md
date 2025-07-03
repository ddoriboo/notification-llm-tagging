# 📊 대출 알림 문구 LLM 태깅 분석 프로젝트

## 🎯 프로젝트 개요

대출 관련 푸시 알림 문구의 성과를 분석하고 LLM을 활용한 자동 태깅 시스템을 구축하여 클릭률을 최적화하는 프로젝트입니다.

## 🏆 주요 성과

- **80개 문구 LLM 태깅 완료** (8개 카테고리, 26개 특성)
- **골든 조합 발견**: `persuasive 톤 + complete_action CTA = 12.29%` (59% 향상)
- **예측 모델 구축**: 클릭률 예측 정확도 R² = 0.258
- **서비스별 맞춤 전략** 수립 (8개 서비스)

## 📁 파일 구조

```
📦 notitest/
├── 📊 분석 결과
│   ├── final_insights_report.md          # 상세 분석 리포트
│   ├── complete_analysis_summary.md      # 실무 적용 가이드
│   └── notification_analysis_guide.md    # 분석 방법론 가이드
│
├── 🤖 LLM 태깅 시스템
│   ├── enhanced_tagging.py              # 고도화된 자동 태깅 시스템
│   ├── run_tagging.py                   # 기본 태깅 실행 코드
│   └── manual_tagging_example.py        # 수동 태깅 예시
│
├── 📈 예측 모델
│   ├── prediction_model.py              # 클릭률 예측 모델
│   └── analyze_loan_notifications.py    # 기초 통계 분석
│
├── 📋 설정 파일
│   ├── .gitignore                       # Git 무시 파일 목록
│   └── README.md                        # 프로젝트 문서
│
└── 📄 예시 파일
    └── tagging_structure_example.json   # 태깅 구조 예시
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# OpenAI API 키 설정 (환경변수 권장)
export OPENAI_API_KEY="your-api-key-here"
```

### 2. 메인 프로그램 실행

```bash
# 도움말 보기
python main.py --help

# 기본 분석 실행 (API 키 불필요)
python main.py analyze --input your_data.csv

# LLM 태깅 실행
python main.py tag --input your_data.csv --sample-size 100

# 예측 모델 구축
python main.py predict

# 전체 파이프라인 실행 (분석 + 태깅 + 예측)
python main.py full --input your_data.csv
```

### 3. 빠른 시작 가이드

```bash
# 가장 간단한 시작 방법 안내
python quick_start.py
```

## 🎯 핵심 발견사항

### 최고 성과 조합
| 조합 | 클릭률 | 개선 효과 |
|------|--------|-----------|
| persuasive + complete_action | 12.29% | 59% 향상 |
| urgent + click_now | 9.38% | 22% 향상 |
| persuasive + apply | 9.24% | 20% 향상 |

### 심리적 트리거 최적화
- **3개 트리거 조합**이 최적 (9.01% vs 평균 7.71%)
- 효과적 조합: `FOMO + Benefit + Personalization`

### 서비스별 전략

#### 신용점수조회 (16.60%+ 클릭률)
```
✅ 톤: persuasive
✅ CTA: complete_action
✅ 제안: report
✅ 예시: "(광고) 미뤄두었던 약관동의 완료하고 신용분석 리포트 받으세요!"
```

#### 신용대환대출 (10.66%+ 클릭률)
```
✅ 톤: friendly
✅ CTA: check_info
✅ 이모지: 1개 (👉)
✅ 예시: "(광고) 한도가 달라졌을까? 내 금리·한도 확인할 시간이에요👉"
```

## 📊 태깅 시스템

### 8개 주요 카테고리

1. **메시지 톤**: urgent, friendly, informative, persuasive, reminder
2. **행동 유도**: click_now, check_info, complete_action, compare, apply
3. **심리적 트리거**: fomo, curiosity, benefit, personalization, urgency
4. **콘텐츠 요소**: 이모지 수, 질문 포함, 숫자 포함, 시간 참조 등
5. **제안 유형**: rate_check, limit_check, refinancing, report, event
6. **타겟 감정**: anxiety, hope, relief, excitement, trust
7. **가치 제안**: 주요 혜택, 혜택 명확성, 개인화 수준
8. **언어적 특징**: 문장 유형, 격식 수준, 가독성

## 📈 비즈니스 임팩트

### 단기 효과 (1-3개월)
- 클릭률 25-40% 향상
- ROI 30% 개선
- 작업 시간 60% 단축

### 중장기 효과 (6개월+)
- 예측 정확도 95% 달성
- 자동화 수준 80% 구현
- 데이터 기반 차별화

## 🔧 A/B 테스트 가이드

### Phase 1: 핵심 조합 검증
```python
# 테스트 설계
test_groups = {
    "A": "persuasive + complete_action",
    "B": "friendly + apply"  # 기존 방식
}
sample_size = 2000  # 각 그룹
duration = "2주"
```

### Phase 2: 트리거 개수 최적화
- 1개 vs 2개 vs 3개 vs 4개 트리거 비교
- 가설: 3개 그룹에서 최고 성과

## 📚 문서 가이드

- **`final_insights_report.md`**: 전체 분석 결과 상세 리포트
- **`complete_analysis_summary.md`**: 실무진을 위한 적용 가이드
- **`notification_analysis_guide.md`**: 분석 방법론 및 이론적 배경

## ⚠️ 보안 및 주의사항

### 🔒 포함하지 않은 민감 정보
- OpenAI API 키 (환경변수로 관리)
- 실제 고객 데이터 (샘플 데이터만 포함)
- 개인정보 및 내부 데이터

### 📋 컴플라이언스
- 금융광고 표시 의무 준수
- 개인정보보호법 준수
- 과장 광고 방지

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 문의사항

프로젝트 관련 문의사항이나 개선 제안이 있으시면 이슈를 등록해 주세요.

---

> 💡 **Tip**: 실제 운영 환경에서 사용하실 때는 API 키 관리, 데이터 보안, 그리고 A/B 테스트 설계에 특별히 주의해 주세요!