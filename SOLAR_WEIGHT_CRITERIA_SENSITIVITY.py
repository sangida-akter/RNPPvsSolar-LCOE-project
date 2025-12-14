

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

# =====================================================
# 1. INPUT DATA
# =====================================================

scores_RNPP = np.array([
    0.647, 0.123, 0.444, 0.973, 0, 0, 0.2, 0.047,
    0.5, 0.5, 0.75, 0.75, 0.75, 0.5, 0.75
])

scores_Solar = np.array([
    0.334, 0.965, 0.928, 0.013, 0.857, 0.5, 0.932, 0.733,
    0.75, 0.75, 0.5, 0.5, 0.5, 0.75, 0.75
])

weights_base = np.array([
    0.0474, 0.0366, 0.0346, 0.0770, 0.0276,
    0.0276, 0.0164, 0.0345, 0.1324, 0.1254,
    0.0701, 0.0701, 0.0701, 0.1151, 0.1151
])

variation_pct = np.array([
    0.20, 0.20, 0.20, 0.20, 0.20,
    0.25, 0.25, 0.25, 0.10, 0.10,
    0.15, 0.15, 0.15, 0.10, 0.10
])

criteria = [
    "LCOE", "Capital cost", "O&M cost", "Capacity factor",
    "Construction time", "Land use", "Decommissioning cost",
    "Job creation", "Safety / Risk", "Environmental impact",
    "Energy security", "Technology maturity",
    "Grid compatibility", "Social acceptance",
    "Policy / regulatory support"
]

n_sim = 10000

# =====================================================
# 2. MONTE CARLO SIMULATION
# =====================================================

RNPP_total, Solar_total = [], []

for _ in range(n_sim):
    sampled_weights = weights_base * (1 + np.random.uniform(-variation_pct, variation_pct))
    sampled_weights /= sampled_weights.sum()

    RNPP_total.append(np.sum(sampled_weights * scores_RNPP))
    Solar_total.append(np.sum(sampled_weights * scores_Solar))

RNPP_total = np.array(RNPP_total)
Solar_total = np.array(Solar_total)

print("\nMonte Carlo Summary:")
print(f"RNPP  → min={RNPP_total.min():.4f}, max={RNPP_total.max():.4f}, mean={RNPP_total.mean():.4f}")
print(f"Solar → min={Solar_total.min():.4f}, max={Solar_total.max():.4f}, mean={Solar_total.mean():.4f}")

# =====================================================
# 3. PROBABILITY DENSITY PLOT
# =====================================================

plt.figure(figsize=(8, 6), dpi=300)

sns.kdeplot(RNPP_total, label="RNPP", fill=True)
sns.kdeplot(Solar_total, label="Solar PV", fill=True)

plt.xlabel("Total Weighted Score", fontsize=12, labelpad=10)
plt.ylabel("Probability Density", fontsize=12, labelpad=10)

plt.legend(fontsize=10)
plt.grid(True, linestyle="--", alpha=0.5)
plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

plt.savefig("Figure_Probability_Density.png",
            dpi=300, bbox_inches="tight", pad_inches=0.25)
plt.show()

# =====================================================
# 4. TORNADO ANALYSIS + PRINT RESULTS
# =====================================================

tornado_RNPP, tornado_Solar = [], []

print("\nPercentage Change in Total Weighted Score for Each Criterion:")
print("{:<30s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
    "Criterion", "RNPP Low", "RNPP High", "Solar Low", "Solar High"))

for i, var in enumerate(variation_pct):
    w_low = weights_base.copy()
    w_high = weights_base.copy()

    w_low[i] *= (1 - var)
    w_high[i] *= (1 + var)

    w_low /= w_low.sum()
    w_high /= w_high.sum()

    rnpp_low = np.sum(w_low * scores_RNPP)
    rnpp_high = np.sum(w_high * scores_RNPP)
    solar_low = np.sum(w_low * scores_Solar)
    solar_high = np.sum(w_high * scores_Solar)

    tornado_RNPP.append((rnpp_low, rnpp_high))
    tornado_Solar.append((solar_low, solar_high))

    print("{:<30s} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
        criteria[i], rnpp_low, rnpp_high, solar_low, solar_high))

# =====================================================
# 5. SORT BY RNPP SENSITIVITY
# =====================================================

RNPP_sensitivity = np.abs(np.array([h - l for l, h in tornado_RNPP]))
sorted_idx = np.argsort(RNPP_sensitivity)[::-1]

sorted_criteria = [criteria[i] for i in sorted_idx]
RNPP_low = [tornado_RNPP[i][0] for i in sorted_idx]
RNPP_high = [tornado_RNPP[i][1] for i in sorted_idx]
Solar_low = [tornado_Solar[i][0] for i in sorted_idx]
Solar_high = [tornado_Solar[i][1] for i in sorted_idx]

y_pos = np.arange(len(sorted_criteria))

# =====================================================
# 6. TORNADO PLOT – RNPP
# =====================================================

plt.figure(figsize=(11, 8), dpi=300)

plt.hlines(y_pos, RNPP_low, RNPP_high, lw=6, alpha=0.7)
plt.yticks(y_pos, sorted_criteria, fontsize=10)
plt.gca().invert_yaxis()

plt.xlabel("Total Weighted Score (RNPP)", fontsize=12, labelpad=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

plt.subplots_adjust(left=0.38, bottom=0.18, right=0.97, top=0.97)

plt.savefig("Figure_Tornado_RNPP.png",
            dpi=300, bbox_inches="tight", pad_inches=0.3)
plt.show()

# =====================================================
# 7. TORNADO PLOT – SOLAR PV
# =====================================================

plt.figure(figsize=(11, 8), dpi=300)

plt.hlines(y_pos, Solar_low, Solar_high, lw=6, alpha=0.7)
plt.yticks(y_pos, sorted_criteria, fontsize=10)
plt.gca().invert_yaxis()

plt.xlabel("Total Weighted Score (Solar PV)", fontsize=12, labelpad=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

plt.subplots_adjust(left=0.38, bottom=0.18, right=0.97, top=0.97)

plt.savefig("Figure_Tornado_Solar.png",
            dpi=300, bbox_inches="tight", pad_inches=0.3)
plt.show()

