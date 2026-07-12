# Rebuttal 응답 포인트 정리 (초안 전 단계) — Submission 15674

- 이 문서는 **영문 초안을 쓰기 전 서술 포인트**를 개조식으로 정리한 것.
- 사용자가 이 포인트를 수정/확정하면, 그 내용을 반영해 리뷰어별 **영문 초안**을 작성.
- ⚠️ 표시 = 프레이밍 결정 또는 확인이 필요한 지점.
- 수치는 이번 rebuttal 실험(E1~E4)에서 실제 산출된 값. (근거 파일은 `results/E1_RESULT.md`, `results/E2CD_RESULT.md`, `results/E3_RESULT.md`, `results/e4/`)

---

## 확보된 실험 근거 (응답에 인용할 수치)

- **E1 — 학습셋 품질 감사** (저자 2인 독립, n=200, delayed/valid)
  - 5개 제약 pass rate: Spatial Separation 98.0% / Temporal Implicitness 92.8% / Mutual Exclusivity 94.8% / Speaker Consistency 100% / Duration Validity 95.5%
  - 저자 간 원 일치율 99.4%, overall Cohen's κ = 0.918 (almost perfect)
- **E2-a — 유의성 sign test** (실제 원 개수로 확정, two-sided exact binomial, tie 제외)
  - **turn-level Time-Specificity** (n=200): W/T/L = 108/48/44, decisive win-rate 71.1%, **p = 2.2×10⁻⁷ → 유의한 win**
  - **dialog-level Delay-Appropriateness** (n=80): W/T/L = 32/18/30, win-rate 51.6%, **p = 0.899 → 유의하지 않음**
  - **dialog-level Time-Specificity** (n=80): W/T/L = 32/18/30, win-rate 51.6%, **p = 0.899 → 유의하지 않음**
  - (tie를 반반 분배한 민감도 분석도 동일 결론)
- **E2-c — coarse-bucket accuracy** (TIMER, 6버킷=Table 6)
  - exact-bucket 35.2%, adjacent(±1) 74.7%
- **E2-d — Table 2 재현** (메인 TIMER-3B, 정확 재현)
  - Precision 0.781 / Recall 0.802 / F1 0.791 / FPR 0.041 / RMSLE 1.1903
- **E3 — 대체 simulator (Claude Sonnet 4.5)**
  - 시뮬레이터 교체해도 순위 보존: Coherence 순위 동일, Delay-Appr·Time-Spec는 3·4위 교체뿐이나 CI 겹쳐 유의차 아님
  - 각 에이전트 지연율도 두 시뮬레이터에서 거의 동일
  - **[확정] 상대 서술만 사용**: "시뮬레이터를 바꿔도 순위가 불변"으로만 서술. 절대 점수·Figure 3 재현 언급 금지. (우리 재현 기준선이 Fig.3 절대값을 재현하지 못하므로)
- **E4 — 순서 bias**
  - GPT-4o·Claude 두 생성기 모두 4-1/4-2 순서 무관 (모든 측정 유의차 없음, 가장 가까운 INSTANT 길이도 p=0.059이며 교차생성기에서 재현 안 됨)
  - 시간누출률 0~4% → Temporal Implicitness 준수 방증
- **논거(실험 아님) — GPT-4o generator 실패**
  - 벤치마크 생성기 GPT-4o 자신이 timing prediction F1 0.263 (vs TIMER 0.791). 벤치마크가 생성기 스타일을 보상한다면 GPT-4o가 최강이어야 함 → 스타일이 아닌 temporal reasoning 측정 증거

---

## Reviewer CYaW (Overall 2.0) — novelty 반박 (실험 불필요, 논증)

- 감사 인사 후, 단일 쟁점(GapChat 대비 incremental)에 하나의 강한 논증으로 대응
- **(a) 과제 정의가 다름**
  - GapChat: *주어진* 시간 간격에 조건화해 이벤트 진행 반영
  - 우리: 에이전트가 delay 여부·길이를 *스스로 예측* (response timing prediction이 first-class subtask, Eq. 2)
  - "predicting timing" vs "conditioning on given gaps"는 방법 차이가 아니라 문제 정의 차이
- **(b) 실증 증거**
  - 차이가 marginal이라면 GapChat 학습 모델이 이 태스크를 풀었어야 함
  - GapChat-3B는 이미 baseline(Table 3): R-L 8.61 vs TIMER 22.26, time-specificity 1.05 vs 3.36 → 별개 능력임의 직접 증거
- **(c) 자원 기여**
  - 이벤트 128개·2.6K 대화(GapChat) vs 55K 이벤트·55K 대화, 분 단위 granularity, timing supervision 포함 (Table 1)
  - 학습 스킴(멀티태스크·special token) 효과는 ablation(Table 5)으로 개별 검증
  - ⚠️ 감사 결과(E1) 인용 가능: 학습셋 품질을 수치로 뒷받침(pass rate 92.8~100%)
- **(d) 마무리**
  - related work에서 GapChat 구분 이미 명시(L138-143), 개정판 intro에서 더 일찍 명확화 약속
  - sV8H 리뷰 직접 인용은 피함(리뷰어 간 대립 유도 역효과)
  - 질문으로 마감: "위 구분(정의+실증)을 고려해도 incremental로 보시는지, 남은 우려 구체화 요청"
- 리스크: confidence 4·짧은 리뷰 → AC도 읽는다는 전제로 자기완결적 표 형태 대비 포함

---

## Reviewer ka1t (Overall 3.0) — 신뢰 구축형 (concern 1 집중)

- **#1 Human eval (soundness 2.5 원인) — 부분 수용 + 완화**
  - 선제 인정: dialog-level Delay-Appr·Time-Spec (32/18/30)는 단독으로 유의하지 않음 (E2-a sign test **p=0.899**)
  - L539 "outperforms" → "achieves comparable or better"로 완화 약속
  - 유의성 근거 재배치: (i) **논문 게재 Fig. 3의 pairwise t-test p<0.05** (논문 값 그대로 인용), (ii) 이중 judge 일치(C.5, ρ≥0.88), (iii) E3 대체 simulator 순위 보존(상대 서술)
  - human eval 역할 재정의: "LLM 평가의 방향성 검증" — 이 역할에서는 성공(순위 일치)
  - turn-level Time-Specificity (108/48/44, win-rate 71.1%)는 명확한 win (E2-a **p=2.2×10⁻⁷**)
  - κ 0.45~0.50은 주관적 대화 품질 평가의 통상 수준(문헌 근거)
  - **[확정] E2-b 재현 수치는 응답에서 제외.** 저자 원본 per-dialog 점수가 없고 재현이 절대값 미일치 → **논문에 이미 게재된 Fig.3 결과(유의성 asterisk 포함)를 그대로 인용**. 새 CI/재현 숫자는 도입하지 않음
- **#2 Intro 두 데이터셋 혼용 — 전면 수용**
  - 개정 문안 직접 제시 (benchmark=MC-TACO 기반 §4 / training=ATOMIC 기반 pseudo-label §5.1 명확 분리)
  - "pseudo-label 학습 + human label 평가"는 의도된 설계(annotation 없이 스케일링)임을 명확화
  - 지난 사이클(Lp85)에서도 지적 → "이번에 확실히 고친다" 신호
- **#3 Eq. 2 표기 — 수용 + 해명**
  - τ̂는 LM 디코딩 토큰 시퀀스, Eq. 2 분포는 LM 출력 분포 그 자체 → 표기 오류 아님
  - 표현·디코딩 방식(<TIME> 토큰 뒤 숫자+단위, beam search)을 §5.2에 명시 약속
- **#4 순서 bias — E4 결과로 답**
  - GPT-4o·Claude 두 생성기 모두 순서 무관(유의차 없음), 시간누출 0~4%
  - 보조 논거: 4-1/4-2는 동일 최종 턴의 두 대안 생성이지 순차 생성 아님
  - 정직 서술: 가장 가까운 측정(INSTANT 길이 p=0.059)도 n.s.이며 교차생성기 재현 안 됨 → 노이즈
- **#5 55K 품질 — E1 감사로 답**
  - 5제약 pass rate 92.8~100%, 저자 원 일치율 99.4%, κ=0.918
  - "high-quality" 문구를 감사 수치 기반 서술로 교체 약속
  - **[확정] 감사 대상은 `refined/delayed/valid` (5,478행)이며, 55K training set과 동일한 합성 파이프라인으로 만든 in-distribution subset.** 따라서 감사 pass rate는 학습셋 전체 품질의 대표값으로 서술 가능. 응답에는 "학습 데이터와 동일 방식으로 생성된 표본"임을 명시
- **#6 + typos — 전부 수용, 일괄 수정 목록 제시**
  - 디테일 지향 리뷰어 → typo 하나하나 "fixed" 응답 자체가 점수에 긍정적

---

## Reviewer sV8H (Overall 3.5) — 유지 + 상향

- 우호적·전문적 리뷰. 감사 표현 후 번호별 응답
- **#1 합성 데이터 / artifact 위험**
  - 소스·생성기 분리: benchmark(MC-TACO annotation + GPT-4o) vs training(ATOMIC + GPT-3.5) → 학습-평가 shortcut 제한적
  - 결정타: 벤치마크 생성기 GPT-4o가 timing prediction 실패(F1 0.263 vs TIMER 0.791) → 벤치마크가 스타일 아닌 temporal reasoning 측정
  - E1 감사로 품질 보강(5제약 92.8~100%)
  - 324 규모 인정 + 수동 스크리닝·MC-TACO 커버리지 서술
- **#2 단일 point estimate 한계 — E2-c로 답**
  - 원칙 동의(Limitations L614-618)
  - (i) 1차 결정 binary delayed/instant에서 F1 0.791, (ii) RMSLE는 log-scale이라 비례 오차 관대
  - (iii) **E2-c 신규**: 6-bucket exact 35.2%, adjacent(±1) 74.7% → point estimate는 거칠어도 방향은 신뢰(4번 중 3번 한 버킷 이내)
  - 분포 예측은 future work 명시 동의
- **#3 실세계 검증 부재 — 정직하게 인정**
  - Appendix A(12.7M+25.6M 실코퍼스, 40%/27% delayed)로 현상 실재성 입증
  - event-grounded 부분집합 기계적 분리 불가함을 논문이 이미 인정(L983-995)
  - cross-dataset 평가는 기존 벤치마크에 timing supervision 없어 불가
  - 실사용 검증은 한계로 인정 + future work 약속 (무리한 신규 실험 대신 정직한 인정)
- **#4 LLM simulator/judge 의존 — 3축 robustness**
  - judge축: 기존 C.5 cross-judge (ρ≥0.88)
  - simulator축: **E3 신규** — 대체 simulator(Claude)에서 순위 보존
  - **[확정] E3는 상대 서술만** — "시뮬레이터를 바꿔도 순위 불변". 절대값·Fig.3 재현 언급 없음
  - generator축: GPT-4o 실패 논거(#1 교차 참조)
- **#5 human eval 비교 대상 제한 — 정직 + 범위 재진술**
  - annotator 추가 불가 → human eval 목적은 "최강 baseline 대비 LLM 평가 신뢰성 검증"이었음
  - LLaMA-70B·GapChat 비교는 LLM-judge 결과(Table 3, Fig. 3)가 큰 마진으로 커버
  - camera-ready에서 확대 human eval 약속(선택)
  - "TIMER가 전반적으로 더 나은 대화 모델은 아니다" 수용 — 주장은 "temporal control 우수"이지 전반 우위 아님(Limitations L619-623)

---

## 공통 마무리

- 각 리뷰어 응답 말미에 **Revision plan** 요약 리스트 첨부
  - intro 데이터셋 명확화(ka1t#2), "outperforms"→"comparable or better"(ka1t#1), Eq.2 디코딩 명시(ka1t#3), "high-quality"→감사 수치 서술(ka1t#5/sV8H#1), typo 일괄, GapChat 구분 intro 조기 명시(CYaW)
- 톤: 방어적이지 않게, 수용할 것 빠르게 수용, 개정 항목 구체적으로

---

## 결정 사항 (확정됨)

1. **[확정] E3 프레이밍**: 상대 서술(순위 불변)로만. Fig.3 절대값 언급 없음.
2. **[확정] Fig.3(ka1t#1)**: 저자 원본 per-dialog 점수 없음 → **논문 게재 Fig.3 결과를 그대로 인용**, E2-b 재현 수치 제외.
3. **[확정] E1 규모(ka1t#5)**: valid는 55K와 동일 파이프라인의 in-distribution subset → 학습셋 품질 대표값으로 서술.
4. **[확정] E2-a 정확 p값**: 실제 원 개수 확보(turn n=200 → %×2, dialog n=80). turn-level Time-Spec 108/48/44 p=2.2×10⁻⁷ (유의), dialog-level DA·TS 32/18/30 p=0.899 (n.s.).
5. **[확정] E4**: 순서 무관 결론 + INSTANT 길이 p=0.059가 경계값이나 교차생성기 재현 안 됨을 **정직하게 서술**.

## 초안 미반영 / 제외 항목

- **E2-b (Fig.3 CI 재현)**: 응답에서 제외 (결정 2).
- **E3 절대 점수 표**: 제외 (결정 1). E3는 순위 불변 서술만.
