# Interpolation Evaluation Method

> **Notebook**: `step2_simulate_scdiffeq_env.ipynb` (Section 3)
> **Model**: `LightningSDE-FixedPotential-RegularizedVelocityRatio`
> **Checkpoint**: `epoch=1949` (best val loss 38.44 @ epoch=1954)

---

## 1. 개요

scDiffeq 모델이 **한 번도 본 적 없는 발달 시점**의 세포 분포를 얼마나 정확하게 예측할 수 있는지 평가하는 실험이다.

훈련 시 특정 시점(`t = 0.261`)을 의도적으로 제외(hold-out)한 뒤, 학습된 SDE 모델로 인접 시점에서부터 해당 시점까지 시뮬레이션하여 실제 세포 분포와 비교한다.

---

## 2. 데이터 구성

### 2-1. 발달 시간의 정규화 (`age_time_norm`)

실제 뇌 발달 기간을 `[0, 1]`로 선형 정규화한 값이다.

| `age_time_norm` | 대략적 의미 | 세포 수 |
|---|---|---|
| 0.000 | 가장 이른 태아기 | 3,578 |
| 0.033 | 초기 태아기 | 33,214 |
| 0.073 | 초기 태아기 | 18,068 |
| 0.122 | 중기 태아기 | 8,898 |
| 0.165 | 중기 태아기 | 19,805 |
| **0.261** | **← Hold-out (test only)** | **8,844** |
| 0.296 | 후기 태아기 | 14,259 |
| 0.871 | 신생아기 | 11,186 |
| 1.000 | 성인 | 7,521 |

### 2-2. Train / Test 분리

- **Train set** (`adata_train`): 위 표에서 `t = 0.261`을 **제외**한 8개 시점 × 116,529개 세포
- **Test set** (`z_test`): `t = 0.261194`에 해당하는 8,844개 세포 (scVI dim30으로 재인코딩 후 저장)

---

## 3. 모델: scDiffeq (Stochastic Differential Equation)

### 3-1. 학습 공간

- scVI (dim=30) latent space에서 동작한다.
  - 원본 유전자 공간(49,133 genes) → scVI 인코더 → **30차원 latent vector**
  - 각 세포는 30차원 공간의 한 점으로 표현됨
- 모델이 학습하는 것: **이 30차원 공간에서의 세포 집단 이동 규칙**

### 3-2. SDE 수식

세포 집단의 시간에 따른 변화를 아래 SDE로 모델링한다.

```
dX_t = μ(X_t) dt + σ(X_t) dW_t
```

| 항 | 이름 | 역할 |
|---|---|---|
| `μ(X_t)` | **Drift** (결정론적 부분) | 시간에 따른 주요 발달 방향 |
| `σ(X_t)` | **Diffusion** (확률론적 부분) | 세포 운명의 불확실성/이질성 |
| `dW_t` | Brownian motion | 확률적 노이즈 |

- **Drift**: 발달 궤적의 평균적 방향을 학습 (NPC → Neuroblast → Excitatory Neuron 등)
- **Diffusion**: 동일 시점의 세포들이 서로 다른 운명을 가질 확률적 분기를 표현

### 3-3. 학습 목표 (Sinkhorn Loss)

인접한 두 시점의 세포 분포 사이의 **Sinkhorn distance**를 최소화:

```
L = Σ_{(t_i, t_{i+1})} Sinkhorn( p̂(X_{t_{i+1}}), p(X_{t_{i+1}}) )
```

- `p̂(X_{t+1})`: SDE 시뮬레이션으로 예측한 분포
- `p(X_{t+1})`: 실제 훈련 데이터의 분포

---

## 4. Interpolation 절차

### 4-1. 시작점 선택

```
T_SRC     = 0.165  (hold-out 직전 훈련 시점, 19,805개 세포)
T_HOLDOUT = 0.261  (hold-out 시점, 실제 테스트 세포 분포)
```

훈련 데이터에서 `t = 0.165`에 해당하는 세포들의 **30차원 scVI 임베딩**을 초기 상태(`x0`)로 사용한다.

### 4-2. SDE 수치 적분 (Euler-Maruyama)

```python
trajectory = model.diffeq.solve_sde(
    x0,
    t_start = 0.165,
    t_end   = 0.261,
    n_steps = 10   # dt=0.01 기준
)
x_pred = trajectory[-1]  # 최종 시점의 예측 세포 분포
```

- `solve_sde`는 Euler-Maruyama 방법으로 SDE를 수치적으로 적분
- 출력: `[n_steps+1, n_cells, 30]` 형태의 궤적 텐서
- 최종 스텝(`trajectory[-1]`)이 `t = 0.261`에서의 예측 세포 분포

```
t=0.165  →  (SDE steps)  →  t=0.170  →  ...  →  t=0.261
  x0                                               x_pred
[19805, 30]                                      [19805, 30]
```

### 4-3. 예측값과 실측값 비교

```
x_pred : 모델이 시뮬레이션한 분포  [19,805 cells × 30 dims]
z_test : 실제 hold-out 세포 분포  [ 8,844 cells × 30 dims]
```

두 분포는 **세포 수가 다르기 때문에** 개별 세포 단위 비교가 아닌 **분포 단위 비교**를 수행한다.

---

## 5. 평가 지표

### 5-1. EMD (Earth Mover's Distance / Wasserstein-1 Distance)

두 세포 집단 분포 사이의 **최적 수송 비용**을 측정한다.

```
EMD = min_{γ ∈ Π(p, q)}  Σ_{i,j} γ_{ij} · ||x_i - x_j||
```

- `p`: 예측 분포 (x_pred), `q`: 실제 분포 (z_test)
- `γ`: 두 분포를 연결하는 최적 수송 계획 (transport plan)
- 두 분포가 완전히 같으면 EMD = 0, 멀수록 커짐

계산 비용을 줄이기 위해 **2,000개 세포를 랜덤 샘플링**하여 계산한다.

```python
M = cdist(x_pred_sub, z_test_sub)          # Euclidean cost matrix
w2_pred = ot.emd2(uniform_p, uniform_q, M) # POT 라이브러리 사용
```

### 5-2. Baseline 비교

모델의 기여가 실제로 있는지 확인하기 위해 **시뮬레이션 없이** 시작점 세포(`t=0.165`)를 그대로 test set과 비교:

```
w2_base = EMD( x_src,  z_test )   ← 시뮬레이션 없이 비교
w2_pred = EMD( x_pred, z_test )   ← SDE 시뮬레이션 후 비교

Improvement = (w2_base - w2_pred) / w2_base × 100 (%)
```

`w2_pred < w2_base`이면 모델이 발달 방향을 올바르게 학습했다고 볼 수 있다.

### 5-3. Cell Type Composition 비교

30차원 latent space는 직접 해석하기 어려우므로, 예측 세포들에 **kNN으로 cell type을 할당**하여 비교한다.

```python
# Training 데이터의 kNN graph 구축
knn.fit(adata_train.obsm['X_scVI'])

# 예측 세포 각각에 대해 가장 가까운 훈련 세포 10개를 찾아 majority vote
_, idx = knn.kneighbors(x_pred)
ct_pred = majority_vote(train_celltypes[idx])

# 비율 비교
comp_pred = x_pred의 cell type 비율
comp_true = z_test의 실제 cell type 비율 (test_obs에서 로드)
```

Bar plot으로 각 세포 타입의 비율을 predicted vs actual 나란히 비교한다.

---

## 6. 시각화 출력

| 파일 | 내용 |
|---|---|
| `interp_pred_vs_actual_umap.pdf` | 예측 vs 실제 세포를 합쳐 UMAP으로 시각화 |
| `interp_celltype_composition.pdf` | Cell type 비율 bar plot (Predicted vs Actual) |

---

## 7. 해석 방법

| 결과 | 해석 |
|---|---|
| `w2_pred ≪ w2_base` | 모델이 발달 궤적을 잘 학습함 |
| `w2_pred ≈ w2_base` | 모델이 시간 방향을 제대로 학습하지 못함 |
| UMAP에서 Predicted와 Actual이 겹침 | 분포가 유사하게 예측됨 |
| Cell type 비율이 유사 | 세포 타입별 발달 속도도 올바르게 학습됨 |

### 주의사항

- **시뮬레이션은 확률적(stochastic)**이므로 매 실행마다 결과가 약간씩 달라진다. Seed 고정 또는 여러 번 반복 실행이 권장된다.
- 시작점(`t=0.165`)의 세포 수(19,805)와 테스트 세포 수(8,844)가 다르다. EMD 계산 시 균일 가중치를 사용하므로 세포 수 차이는 문제되지 않는다.
- `t=0.165 → t=0.261` 구간은 **훈련 중 본 적 없는 구간**이므로 순수한 SDE 외삽(extrapolation)이 아닌, 훈련 시 학습된 dynamics를 해당 구간에 적용하는 **interpolation**이다.
