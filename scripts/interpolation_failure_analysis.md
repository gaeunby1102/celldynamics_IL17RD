# Interpolation 실패 원인 분석

**작성일**: 2026-02-22
**대상 실험**: `step2_simulate_scdiffeq_env.ipynb` (구 모델, epoch=999)

---

## 1. 실험 설정

| 항목 | 내용 |
|---|---|
| 시뮬레이션 구간 | t=0.165 → t=0.261 |
| 출발 세포 | 훈련 데이터 t=0.165 (19,805 cells) |
| 비교 대상 | 실제 test set t=0.261 (8,844 cells) |
| 평가 지표 | EMD + Cell type composition |

---

## 2. 실패 원인

### 2-1. 생물학적 불연속 (구조적 문제)

t=0.165와 t=0.296 사이에 **훈련 데이터가 전혀 없는 gap**이 존재하는데,
이 구간에서 cell type 구성이 완전히 뒤집힌다.

```
t=0.165  Fetal_Inh:35%  Neuroblast:34%  RG:14%  Fetal_Ext:11%  Ext: 5%
           ↓
           ↓  ← 이 구간 훈련 데이터 없음 (t=0.261 test set만 존재)
           ↓
t=0.296  Ext:47%  Fetal_Ext:33%  Inh:11%  Fetal_Inh: 6%  Neuroblast:0.6%
```

모델은 t=0.165 → t=0.296 사이의 전환을 학습할 데이터 자체가 없으므로,
이 구간에서의 interpolation이 구조적으로 불가능하다.

### 2-2. Test set의 계통 편향 (Fair하지 않은 비교)

| 타임포인트 | 주요 cell type | 지배 계통 |
|---|---|---|
| t=0.165 (출발) | Fetal_Inh 35%, Neuroblast 34%, RG 14% | **억제성 계열** |
| t=0.261 (목표) | Fetal_Ext 52%, Ext 30% | **흥분성 계열** |

시뮬레이션 출발점(t=0.165)은 억제성(Inh) 계열 세포로 구성되어 있고,
실제 test set(t=0.261)은 흥분성(Ext) 계열 세포가 지배적이다.
서로 다른 lineage이므로 아무리 잘 시뮬레이션해도 분포를 맞출 수 없다.

### 2-3. 체크포인트 경로 오류

노트북에 명시된 체크포인트:
```
results/train_scVI/dim_test/IL17RD_SDE_20260120_221359/.../epoch=1949-step=58500.ckpt
```
→ **해당 경로 존재하지 않음**. 실제로 사용된 것은 `dim_test/dim30/` 아래의 `epoch=999` 모델.

---

## 3. 타임포인트 전체 Cell Type 구성

| 타임포인트 | 총 세포 수 | Top-3 Cell Type |
|---|---|---|
| t=0.000 | 3,578 | RG:56%, Neuroblast:44% |
| t=0.033 | 33,214 | Neuroblast:53%, RG:47% |
| t=0.073 | 18,068 | Neuroblast:64%, RG:34% |
| t=0.122 | 8,898 | Neuroblast:56%, RG:32%, Fetal_Inh:12% |
| **t=0.165** | **19,805** | **Fetal_Inh:35%, Neuroblast:34%, RG:14%** |
| *(t=0.261 test)* | *8,844* | *Fetal_Ext:52%, Ext:30%, Fetal_Inh:11%* |
| t=0.296 | 14,259 | Ext:47%, Fetal_Ext:33%, Inh:11% |
| t=0.871 | 11,186 | Ext:66%, Inh:28% |
| t=1.000 | 7,521 | Ext:50%, Inh:41% |

---

## 4. 새 실험 설계 (개선)

### 개선 방향: t=0.165 holdout

| | 구 실험 | **새 실험** |
|---|---|---|
| Holdout | t=0.261 (test set) | **t=0.165** |
| 시뮬레이션 구간 | t=0.165 → t=0.261 | **t=0.122 → t=0.165** |
| 출발 계통 | Fetal_Inh/Neuroblast | Neuroblast/RG/Fetal_Inh |
| 목표 계통 | **Fetal_Ext/Ext (다른 계통!)** | **Fetal_Inh/Neuroblast (같은 계통 ✓)** |
| Gap 내 훈련 데이터 | 없음 | 없음 (holdout이므로) |
| 평가 가능성 | 구조적으로 불가 | **Fair한 평가 가능** |

### t=0.122 → t=0.165 전환의 연속성

```
t=0.122  Neuroblast:56%  RG:32%  Fetal_Inh:12%
           ↓  (같은 계통 내 점진적 전환)
t=0.165  Fetal_Inh:35%  Neuroblast:34%  RG:14%
```

같은 Neuroblast/RG → Fetal_Inh 계통 내의 연속적인 전환이므로,
모델이 실제로 학습한 dynamics를 검증할 수 있다.

### 추가 조치

- **데이터 불균형 보정**: t=0.033 (33,214 cells) → cap 10,000 cells/timepoint
- **GPU**: CUDA_VISIBLE_DEVICES=1
- **학습 epoch**: 2,000

---

## 5. 결론

구 실험의 interpolation이 나빴던 것은 **모델 성능의 문제가 아니라
애초에 fair하지 않은 평가 설계** 때문이었다.
t=0.261 test set은 t=0.165 훈련 데이터와 전혀 다른 lineage(Exc계열)로
구성되어 있어, 어떤 SDE 모델도 이 비교에서 좋은 결과를 내기 어렵다.

새 실험에서는 같은 계통 내의 연속적인 타임포인트(t=0.122→t=0.165)를
holdout으로 설계하여 실질적인 모델 검증이 가능하다.
