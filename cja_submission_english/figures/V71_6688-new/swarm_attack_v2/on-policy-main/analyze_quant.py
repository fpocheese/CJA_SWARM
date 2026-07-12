import numpy as np, os, glob

d = 'onpolicy/scripts/results/simple_converge_v7'
NORM = 65.0
ALGOS = ['Advanced-MAPPO', 'MAPPO', 'IPPO', 'IA2C', 'IQL']

results = {}
for algo in ALGOS:
    files = sorted(glob.glob(os.path.join(d, '%s_seed*_rewards.npy' % algo)))
    arrs = [np.load(f) for f in files]
    ml = min(len(a) for a in arrs)
    mat = np.array([a[:ml] for a in arrs]) / NORM
    mu = mat.mean(axis=0)
    results[algo] = mu

def smooth(x, w=100):
    k = np.ones(w)/w
    return np.convolve(x, k, mode='same')

# ===== 1. 各算法相对自身最优值的90%收敛点 =====
print('=== 各算法相对自身渐近值的90%收敛点 (w=100 smooth) ===')
ep90_map = {}
for algo in ALGOS:
    mu = results[algo]
    mu_s = smooth(mu, 100)
    own_asym = mu[-500:].mean()
    thr = own_asym * 0.9
    cross = np.where(mu_s >= thr)[0]
    ep90 = int(cross[0]) if len(cross) > 0 else None
    ep90_map[algo] = ep90
    print('%s: asymp=%.4f, 90%%_thr=%.4f, ep_90%%=%s' % (algo, own_asym, thr, ep90))

print()
# ===== 2. 各里程碑处的归一化奖励 =====
print('=== 各里程碑处的均值平滑奖励 (normalized) ===')
milestones = [1000, 2000, 4000, 6000, 8000, 9500]
print('%-22s' % 'Algo', end='')
for m in milestones:
    print('  ep%-5d' % m, end='')
print()
for algo in ALGOS:
    mu = results[algo]
    mu_s = smooth(mu, 100)
    print('%-22s' % algo, end='')
    for m in milestones:
        idx = min(m, len(mu_s)-1)
        print('  %7.4f' % mu_s[idx], end='')
    print()

print()
# ===== 3. ART-MAPPO vs others 详细对比 =====
art_s = smooth(results['Advanced-MAPPO'], 100)
print('=== ART-MAPPO vs baselines at key episodes ===')
for ep in [2000, 4000, 6000, 8000, 9500]:
    art_v = art_s[min(ep, len(art_s)-1)]
    print('  ep=%d: ART=%.4f' % (ep, art_v))
    for algo in ['MAPPO', 'IPPO', 'IQL', 'IA2C']:
        bas_s = smooth(results[algo], 100)
        bas_v = bas_s[min(ep, len(bas_s)-1)]
        pct = (art_v - bas_v) / max(abs(bas_v), 1e-6) * 100
        print('         vs %s: %.4f  ART lead=+%.1f%%' % (algo, bas_v, pct))

print()
# ===== 4. 最终性能汇总 =====
art_asym = results['Advanced-MAPPO'][-500:].mean()
print('=== 最终性能汇总 (last 500 eps) ===')
for algo in ALGOS:
    mu = results[algo]
    asym = mu[-500:].mean()
    std = mu[-500:].std()
    raw = asym * NORM
    vs_art = (asym / art_asym - 1.0) * 100
    print('%s: norm=%.4f ± %.4f  raw=%.2f  vs_ART=%.1f%%  ep90=%s' % (
        algo, asym, std, raw, vs_art, ep90_map[algo]))

print()
# ===== 5. 收敛速度：ART-MAPPO ep90 vs IQL ep90 (only one that reaches) =====
art_ep90 = ep90_map['Advanced-MAPPO']
iql_ep90 = ep90_map['IQL']
if art_ep90 and iql_ep90:
    print('ART-MAPPO ep90=%d, IQL ep90=%d, ART faster by %d eps (%.1f%%)' % (
        art_ep90, iql_ep90, iql_ep90-art_ep90, (iql_ep90-art_ep90)/iql_ep90*100))

# ===== 6. 标准差分析 =====
print()
print('=== 跨seed稳定性（last 500 eps，5 seeds 的seed间std） ===')
for algo in ALGOS:
    files = sorted(glob.glob(os.path.join(d, '%s_seed*_rewards.npy' % algo)))
    arrs = [np.load(f) for f in files]
    ml = min(len(a) for a in arrs)
    mat = np.array([a[:ml] for a in arrs]) / NORM
    per_seed_final = mat[:, -500:].mean(axis=1)
    print('%s: seed means=%s  mean=%.4f  seed-std=%.4f' % (
        algo, np.round(per_seed_final, 4).tolist(), per_seed_final.mean(), per_seed_final.std()))
