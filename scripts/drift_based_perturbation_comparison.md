# Drift 기반 Perturbation 비교 방법론

---

## 1. Drift vs Diffusion: 개념적 차이

| | Diffusion `σ(X)` | Drift `μ(X)` |
|---|---|---|
| **의미** | 세포 운명의 **불확실성** | 세포 이동의 **방향과 속도** |
| **질문** | 이 세포가 얼마나 unstable한가? | 이 세포가 어디로, 얼마나 빠르게 가는가? |
| **기존 비교** | `obs['diffusion']` 분포 비교 | — |
| **저장 형태** | scalar (`obs['diffusion']`) | **vector** (`obsm['X_drift']`, 30dim) + scalar (`obs['drift']`) |
| **계산** | `DiffEq.g(t, X)` | `DiffEq.f(t, X)` |

**결정적 차이**: diffusion은 "얼마나 랜덤한가"를 보고, drift는 "어느 방향으로 가는가"를 본다.
Perturbation이 세포 **운명**을 바꾸는지 보려면 drift 비교가 더 직접적이다.

---

## 2. Drift 기반 비교 전략

### 전략 A: 초기 drift 방향 비교 (t=0에서 즉각 반응)

> "Perturbation 직후, 세포들이 가려는 방향이 얼마나 달라졌는가?"

**계산**:
```python
# μ(z_ctrl) vs μ(z_perturb) — 같은 세포를 perturb한 직후 비교
import torch
import torch.nn.functional as F

model.diffeq.eval()
with torch.no_grad():
    mu_ctrl   = model.diffeq.f(None, torch.tensor(z_ctrl,  device=model.device))  # [N, 30]
    mu_ko     = model.diffeq.f(None, torch.tensor(z_ko,    device=model.device))  # [N, 30]
    mu_il17rd = model.diffeq.f(None, torch.tensor(z_il17rd_ko, device=model.device))
```

**비교 지표**:

```python
# 1. 코사인 유사도 (방향 변화)
cos_sim = F.cosine_similarity(mu_ctrl, mu_ko, dim=1)   # [N]
# cos_sim = 1.0  → 같은 방향
# cos_sim < 1.0  → 운명 방향 변화
# cos_sim < 0    → 반대 방향 (역분화 혹은 다른 계통으로 전환)

# 2. 각도 (도 단위)
angle_deg = torch.acos(cos_sim.clamp(-1, 1)) * 180 / torch.pi  # [N]

# 3. Drift 크기(속도) 변화
mag_ctrl = mu_ctrl.norm(dim=1)   # [N]
mag_ko   = mu_ko.norm(dim=1)     # [N]
speed_ratio = mag_ko / mag_ctrl  # >1: 빨라짐, <1: 느려짐

# 4. Drift 벡터 L2 거리 (방향+크기 종합)
drift_L2_diff = (mu_ko - mu_ctrl).norm(dim=1)  # [N]
```

**해석 예시**:
- `cos_sim ≈ 0.95` → 방향 거의 유지 (같은 운명, 다른 타이밍)
- `cos_sim ≈ 0.50` → 운명 방향이 크게 달라짐
- `speed_ratio < 0.8` → KO 후 분화가 느려짐 (progenitor 상태 유지)

---

### 전략 B: 궤적 따라 Drift 발산 분석 (시간 해상도 비교)

> "언제, 어느 지점에서 두 trajectories의 발달 방향이 갈라지기 시작하는가?"

**계산**:
```python
cos_sim_over_time = []
speed_ratio_over_time = []

model.diffeq.eval()
with torch.no_grad():
    for step in range(traj_ctrl.shape[0]):  # 각 timestep
        X_ctrl = torch.tensor(traj_ctrl[step], device=model.device)   # [N, 30]
        X_ko   = torch.tensor(traj_ko[step],   device=model.device)   # [N, 30]

        mu_c = model.diffeq.f(None, X_ctrl)  # [N, 30]
        mu_k = model.diffeq.f(None, X_ko)    # [N, 30]

        cs = F.cosine_similarity(mu_c, mu_k, dim=1).mean().item()
        sr = (mu_k.norm(dim=1) / mu_c.norm(dim=1)).mean().item()

        cos_sim_over_time.append(cs)
        speed_ratio_over_time.append(sr)
```

**시각화**: 시간축(x) × 코사인 유사도(y) 곡선

```
코사인 유사도
1.0 |──────────
    |          ╲
    |            ╲____
0.8 |                 ╲___
    |                     ╲______
0.6 |─────────────────────────────
    t=0  0.1   0.2   0.3  ...  t=1
```

- 초반에 급격히 떨어지면 → 초기 발달 단계가 IL17 신호에 민감
- 특정 `t` 이후 떨어지면 → 해당 발달 단계가 핵심 time window
- 끝까지 높으면 → drift 방향은 유지되나 diffusion만 다를 수 있음

---

### 전략 C: Cell type별 Drift 방향 변화 (세포 종류별 민감도)

> "어떤 세포 타입의 발달 방향이 IL17 perturbation에 가장 많이 영향받는가?"

**계산**:
```python
# t=0에서의 drift 비교 (cell type은 start_obs에 기록됨)
cell_types = start_obs['CellType_refine'].values

results = []
for ct in np.unique(cell_types):
    mask = cell_types == ct
    if mask.sum() < 10:
        continue
    cs = F.cosine_similarity(
        mu_ctrl[mask], mu_ko[mask], dim=1
    ).mean().item()
    sr = (mu_ko[mask].norm(dim=1) / mu_ctrl[mask].norm(dim=1)).mean().item()
    results.append({'CellType': ct, 'cos_sim': cs, 'speed_ratio': sr, 'n': mask.sum()})

df_ct = pd.DataFrame(results).sort_values('cos_sim')
# cos_sim 낮을수록 해당 세포 타입의 운명이 크게 바뀜
```

**결과 해석**:

| CellType | cos_sim | speed_ratio | 해석 |
|---|---|---|---|
| RG HES1 WNT1 | 0.62 | 0.78 | 방향도 바뀌고 분화 느려짐 |
| Neuroblast NEUROD6 | 0.91 | 0.99 | 거의 영향 없음 |
| Exc Fetal STK32B | 0.48 | 1.21 | 완전히 다른 방향, 빨라짐 |

---

### 전략 D: Drift 방향 PCA — "운명 벡터장" 비교

> "Perturbation이 latent space의 velocity field를 어떻게 바꾸는가?"

**계산**:
```python
# 모든 세포의 drift 벡터를 PCA로 2차원에 투영
from sklearn.decomposition import PCA

mu_ctrl_np = mu_ctrl.cpu().numpy()  # [N, 30]
mu_ko_np   = mu_ko.cpu().numpy()    # [N, 30]

pca_drift = PCA(n_components=2)
pca_drift.fit(mu_ctrl_np)

ctrl_2d = pca_drift.transform(mu_ctrl_np)   # [N, 2] — drift 방향의 2D 표현
ko_2d   = pca_drift.transform(mu_ko_np)     # [N, 2]

# Quiver plot (화살표) — 세포 위치에서 drift 방향 표시
# X축: latent PC1, Y축: latent PC2 (latent space에서 위치)
# 화살표: ctrl(파랑) vs KO(빨강) drift 방향
```

- 화살표 방향이 같으면 → 같은 운명 경로
- 화살표가 엇갈리면 → 다른 계통으로 분기

---

## 3. 방법별 요약 및 추천

| 방법 | 계산 비용 | 해석 용이성 | 권장 순서 |
|---|---|---|---|
| **A. 초기 drift 코사인 유사도** | 매우 낮음 | 높음 | ★ 1순위 |
| **B. 궤적 따라 drift 발산** | 중간 | 높음 | ★ 2순위 |
| **C. Cell type별 drift 변화** | 낮음 | 매우 높음 | ★ 2순위 |
| **D. Drift 벡터장 PCA** | 중간 | 중간 | 3순위 |

---

## 4. Diffusion과 Drift를 함께 보는 2D 분석

두 지표를 같이 보면 perturbation의 성격을 4가지로 분류할 수 있다.

```
                   Drift cos_sim (방향 유지도)
                  낮음 ◄─────────────────────► 높음

         높음  │ [운명 전환]          [빨리 다른 곳으로]
Diffusion      │ 방향이 바뀌고        방향은 같은데
변화량         │ 불확실성도 증가       분화 속도만 증가
         낮음  │ [느린 전환]          [미미한 변화]
               │ 방향이 바뀌나        방향도 속도도
               │ 조용히 천천히        거의 그대로
```

```python
# 2D scatter: x=drift cos_sim, y=diffusion 변화량
diff_ctrl = model.diffeq.g(None, torch.tensor(z_ctrl, device=model.device)).norm(dim=1).cpu()
diff_ko   = model.diffeq.g(None, torch.tensor(z_ko,   device=model.device)).norm(dim=1).cpu()
delta_diffusion = (diff_ko - diff_ctrl).numpy()   # 양수=불확실성 증가

plt.scatter(cos_sim.cpu().numpy(), delta_diffusion,
            c=mag_ctrl.cpu().numpy(), cmap='viridis', alpha=0.5, s=10)
plt.axhline(0, color='k', lw=0.5)
plt.axvline(1, color='k', lw=0.5)
plt.xlabel('Drift Cosine Similarity (방향 유지도)')
plt.ylabel('Δ Diffusion (불확실성 변화)')
plt.title('IL17 KO: 운명 변화 분류')
```

---

## 5. 구현 위치

위 분석들은 `step2_simulate_scdiffeq_env.ipynb`에 섹션을 추가하여 구현한다.
`traj_ctrl`, `traj_ko`, `traj_il17rd_ko` 등의 trajectory 배열이 이미 계산되어 있으므로,
`model.diffeq.f(None, X)` 호출만 추가하면 된다.

```python
# step2 노트북에서 trajectory 계산 직후 실행
model.diffeq.eval()
with torch.no_grad():
    mu_ctrl_t0 = model.diffeq.f(None, torch.tensor(z_ctrl, device=model.device))
    mu_ko_t0   = model.diffeq.f(None, torch.tensor(z_ko,   device=model.device))
    # → 이후 전략 A, C, D 적용 가능

    # 전략 B (trajectory 전 구간)
    mu_ctrl_traj = [model.diffeq.f(None, torch.tensor(traj_ctrl[i], device=model.device))
                    for i in range(traj_ctrl.shape[0])]
    # → cos_sim_over_time 계산
```
