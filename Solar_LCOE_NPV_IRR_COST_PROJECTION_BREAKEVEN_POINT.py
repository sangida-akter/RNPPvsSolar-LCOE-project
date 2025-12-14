import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

try:
    import numpy_financial as npf
    have_npf = True
except ImportError:
    have_npf = False
    print("Install numpy_financial: pip install numpy-financial")

warnings.simplefilter(action="ignore", category=FutureWarning)

# ------------------------------
# SIMULATION PARAMETERS
# ------------------------------
n_sims = 3000
analysis_life = 60
pv_module_life = 25
hours_per_year = 8760
rng = np.random.default_rng(123456)

# ------------------------------
# PV SYSTEM INPUTS
# ------------------------------
capacity_MW = rng.uniform(459, 2000, n_sims)
capacity_kW = capacity_MW * 1000
pv_capex_per_kW = rng.triangular(800, 1000, 1200, n_sims)
pv_fom_per_kWyr = rng.uniform(12, 25, n_sims)
pv_cf_arr = rng.triangular(0.17, 0.18, 0.19, n_sims)
pv_deg_arr = rng.uniform(0.005, 0.01, n_sims)

# ------------------------------
# BESS INPUTS
# ------------------------------
bess_power_MW = rng.uniform(180, 220, n_sims)
bess_energy_MWh = bess_power_MW * 4
bess_power_kW = bess_power_MW * 1000
bess_energy_kWh = bess_energy_MWh * 1000
bess_capex_per_kWh = rng.triangular(250, 300, 400, n_sims)
bess_power_cost_per_kW = rng.triangular(100, 120, 150, n_sims)
bess_opex_per_kWyr = rng.uniform(10, 15, n_sims)
bess_life = 15
bess_replace_frac = rng.uniform(0.75, 0.85, n_sims)

# ------------------------------
# FINANCIAL INPUTS
# ------------------------------
wacc = rng.triangular(0.08, 0.095, 0.11, n_sims)
construction_years = rng.integers(1, 3, n_sims)
capex_decline = rng.uniform(0.01, 0.03, n_sims)
tariff = rng.uniform(0.12, 0.15, n_sims)

# ------------------------------
# LAND & DECOMMISSIONING
# ------------------------------
land_per_MW = rng.uniform(50_000, 100_000, n_sims)
decom_per_MW = rng.triangular(300_000, 368_000, 440_000, n_sims)

# ------------------------------
# RESULTS STORAGE
# ------------------------------
lcoe_25y = np.zeros(n_sims)
npv_25y = np.zeros(n_sims)
irr_25y = np.zeros(n_sims)
payback_25y = np.full(n_sims, np.nan)

lcoe_60y = np.zeros(n_sims)
npv_60y = np.zeros(n_sims)
irr_60y = np.zeros(n_sims)
payback_60y = np.full(n_sims, np.nan)

cashflow_25_avg = np.zeros(pv_module_life + 2)
cashflow_60_avg = np.zeros(analysis_life + 2)

# ------------------------------
# STATISTICS FUNCTION
# ------------------------------
def stats_ci(x):
    x_valid = x[~np.isnan(x)]
    mean = np.mean(x_valid)
    p5 = np.percentile(x_valid, 5)
    p95 = np.percentile(x_valid, 95)
    return round(mean,2), round(p5,2), round(p95,2)

# ------------------------------
# NPV & IRR helpers
# ------------------------------
def calc_npv(cf, r):
    if have_npf:
        return npf.npv(r, cf)
    else:
        disc = (1+r) ** np.arange(len(cf))
        return np.sum(cf / disc)

def calc_irr(cf):
    if have_npf:
        try:
            irr_value = npf.irr(cf)
            if np.isnan(irr_value) or irr_value < -1e3 or irr_value > 1e3:
                return np.nan
            return irr_value
        except:
            return np.nan
    else:
        return np.nan

# ------------------------------
# MONTE CARLO LOOP
# ------------------------------
for i in range(n_sims):
    r = wacc[i]
    cf = pv_cf_arr[i]
    deg = pv_deg_arr[i]
    build = int(construction_years[i])

    cap_kW = capacity_kW[i]
    cap_MW = capacity_MW[i]
    land_cost = land_per_MW[i] * cap_MW
    decom_cost = decom_per_MW[i] * cap_MW
    annual_kwh = cap_kW * cf * hours_per_year

    bess_power_kW_i = bess_power_kW[i]
    bess_energy_kWh_i = bess_energy_kWh[i]

    inverter_life = 11
    inverter_frac = 0.06

    # ------------------------------
    # 25-YEAR PV-ONLY
    # ------------------------------
    project_years_25 = build + pv_module_life
    cashflow_25 = np.zeros(project_years_25)
    disc_costs_25 = 0.0
    disc_gen_25 = 0.0

    # Construction cost
    if build == 1:
        cashflow_25[0] -= pv_capex_per_kW[i]*cap_kW + land_cost
        disc_costs_25 += (pv_capex_per_kW[i]*cap_kW + land_cost)/(1+r)**0.5
    else:
        cashflow_25[0] -= 0.6*(pv_capex_per_kW[i]*cap_kW + land_cost)
        cashflow_25[1] -= 0.4*(pv_capex_per_kW[i]*cap_kW + land_cost)
        disc_costs_25 += 0.6*(pv_capex_per_kW[i]*cap_kW + land_cost)/(1+r)**0.5
        disc_costs_25 += 0.4*(pv_capex_per_kW[i]*cap_kW + land_cost)/(1+r)**1.5

    for y in range(pv_module_life):
        t = build + y + 0.5
        disc_costs_25 += pv_fom_per_kWyr[i]*cap_kW/(1+r)**t
        disc_gen_25 += annual_kwh*(1-deg)**y / (1+r)**t

        y_proj = build + y
        cf_year = annual_kwh*(1-deg)**y*tariff[i] - pv_fom_per_kWyr[i]*cap_kW

        if y>0 and y % inverter_life ==0:
            rep_cost = inverter_frac*pv_capex_per_kW[i]*cap_kW*(1-capex_decline[i])**y
            cf_year -= rep_cost
            disc_costs_25 += rep_cost/(1+r)**t

        cashflow_25[y_proj] += cf_year

    disc_costs_25 += decom_cost/(1+r)**(build + pv_module_life + 0.5)
    cashflow_25[-1] -= decom_cost

    lcoe_25y[i] = 1000*disc_costs_25/disc_gen_25 if disc_gen_25>0 else np.nan
    npv_25y[i] = calc_npv(cashflow_25, r)
    irr_25y[i] = calc_irr(cashflow_25)*100
    cashflow_25_avg[:len(cashflow_25)] += cashflow_25/n_sims

    # ------------------------------
    # 60-YEAR PV + BESS
    # ------------------------------
    project_years_60 = build + analysis_life
    cashflow_60 = np.zeros(project_years_60)
    disc_costs_60 = 0.0
    disc_gen_60 = 0.0

    if build == 1:
        total_capex = (pv_capex_per_kW[i]*cap_kW + land_cost
                       + bess_power_cost_per_kW[i]*bess_power_kW_i
                       + bess_capex_per_kWh[i]*bess_energy_kWh_i)
        cashflow_60[0] -= total_capex
        disc_costs_60 += total_capex/(1+r)**0.5
    else:
        total_capex = (pv_capex_per_kW[i]*cap_kW + land_cost
                       + bess_power_cost_per_kW[i]*bess_power_kW_i
                       + bess_capex_per_kWh[i]*bess_energy_kWh_i)
        cashflow_60[0] -= 0.6*total_capex
        cashflow_60[1] -= 0.4*total_capex
        disc_costs_60 += 0.6*total_capex/(1+r)**0.5
        disc_costs_60 += 0.4*total_capex/(1+r)**1.5

    for y in range(analysis_life):
        t = build + y + 0.5
        pv_gen = annual_kwh*(1-deg)**min(y, pv_module_life-1)
        cf_year = pv_gen*tariff[i] - pv_fom_per_kWyr[i]*cap_kW

        if y>0 and y % inverter_life ==0:
            rep_cost_pv = inverter_frac*pv_capex_per_kW[i]*cap_kW*(1-capex_decline[i])**y
            cf_year -= rep_cost_pv
            disc_costs_60 += rep_cost_pv/(1+r)**t

        if y>0 and y % bess_life ==0:
            rep_cost_bess = bess_replace_frac[i]*(bess_power_cost_per_kW[i]*bess_power_kW_i + bess_capex_per_kWh[i]*bess_energy_kWh_i)
            cf_year -= rep_cost_bess
            disc_costs_60 += rep_cost_bess/(1+r)**t

        disc_costs_60 += pv_fom_per_kWyr[i]*cap_kW/(1+r)**t
        disc_gen_60 += pv_gen/(1+r)**t
        cashflow_60[build + y] += cf_year

    disc_costs_60 += decom_cost/(1+r)**(build + analysis_life + 0.5)
    cashflow_60[-1] -= decom_cost

    lcoe_60y[i] = 1000*disc_costs_60/disc_gen_60 if disc_gen_60>0 else np.nan
    npv_60y[i] = calc_npv(cashflow_60, r)
    irr_60y[i] = calc_irr(cashflow_60)*100
    cashflow_60_avg[:len(cashflow_60)] += cashflow_60/n_sims

# Clean invalid values
lcoe_25y[lcoe_25y <= 0] = np.nan
lcoe_60y[lcoe_60y <= 0] = np.nan

# ------------------------------
# PRINT RESULTS
# ------------------------------
def print_results(lcoe, npv, irr, label):
    l_mean, l_p5, l_p95 = stats_ci(lcoe)
    n_mean, n_p5, n_p95 = stats_ci(npv)
    i_mean, i_p5, i_p95 = stats_ci(irr)
    print(f"\n--- {label} ---")
    print(f"LCOE ($/MWh): {l_mean} [{l_p5} – {l_p95}]")
    print(f"NPV ($): {n_mean} [{n_p5} – {n_p95}]")
    print(f"IRR (%): {i_mean} [{i_p5} – {i_p95}]")

print_results(lcoe_25y, npv_25y, irr_25y, "25-YEAR PV-ONLY")
print_results(lcoe_60y, npv_60y, irr_60y, "60-YEAR PV+BESS")

# ------------------------------
# CREATE PLOTS DIR
# ------------------------------
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# ------------------------------
# HISTOGRAM PLOTS
# ------------------------------
params = [("LCOE ($/MWh)", lcoe_25y, lcoe_60y),
          ("NPV ($)", npv_25y, npv_60y),
          ("IRR (%)", irr_25y, irr_60y)]

for name, arr25, arr60 in params:
    plt.figure(figsize=(10,6))
    plt.hist(arr25, bins=50, alpha=0.6, edgecolor='k', label="25y PV-only")
    plt.hist(arr60, bins=50, alpha=0.6, edgecolor='k', label="60y PV+BESS")
    plt.xlabel(name)
    plt.ylabel("Frequency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{name.replace(' ','_').replace('/','')}.pdf"))
    plt.close()

# ------------------------------
# BREAKEVEN PLOT
# ------------------------------
discount_rate_mean = np.mean(wacc)
years_25 = np.arange(len(cashflow_25_avg))
years_60 = np.arange(len(cashflow_60_avg))

cum_disc_cf_25 = np.cumsum(cashflow_25_avg / ((1 + discount_rate_mean) ** (years_25 + 0.5)))
cum_disc_cf_60 = np.cumsum(cashflow_60_avg / ((1 + discount_rate_mean) ** (years_60 + 0.5)))

plt.figure(figsize=(10,6))
plt.plot(years_25, cum_disc_cf_25, marker='o', label="25y PV-only (avg)")
plt.plot(years_60, cum_disc_cf_60, marker='s', label="60y PV+BESS (avg)")
if np.any(cum_disc_cf_25 >= 0):
    be25 = np.argmax(cum_disc_cf_25 >= 0)
    plt.axvline(be25, color='tab:blue', linestyle='--', label=f"25y breakeven: Year {be25}")
if np.any(cum_disc_cf_60 >= 0):
    be60 = np.argmax(cum_disc_cf_60 >= 0)
    plt.axvline(be60, color='tab:orange', linestyle='--', label=f"60y breakeven: Year {be60}")
plt.xlabel("Year")
plt.ylabel("Cumulative discounted cashflow (USD)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "breakeven_comparison.pdf"))
plt.close()

# ------------------------------
# SWANSON'S LAW LCOE PROJECTION
# ------------------------------
current_lcoe_pv_bess = np.nanmean(lcoe_60y)  # mean LCOE from Monte Carlo
years_proj = 60
learning_rate = 0.20  # 20% reduction per doubling
doubling_time = 4     # years per doubling

def swanson_projection(start_lcoe, years, lr, dt, capture=1.0):
    return np.array([
        start_lcoe * (1 - capture * (1 - (1 - lr) ** (t / dt)))
        for t in range(years + 1)
    ])

years_axis = np.arange(0, years_proj + 1)
lcoe_full = swanson_projection(current_lcoe_pv_bess, years_proj, learning_rate, doubling_time, 1.0)
lcoe_partial = swanson_projection(current_lcoe_pv_bess, years_proj, learning_rate, doubling_time, 0.75)

plt.figure(figsize=(10,6))
plt.plot(years_axis, lcoe_full, label="Swanson Full Capture", lw=2)
plt.plot(years_axis, lcoe_partial, label="Swanson Partial Capture (75%)", lw=2, linestyle="--")
plt.axhline(current_lcoe_pv_bess, color="red", linestyle=":", label=f"Current Mean LCOE: {current_lcoe_pv_bess:.2f} $/MWh")
plt.xlabel("Years")
plt.ylabel("LCOE ($/MWh)")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "swansons_law_projection.png"), dpi=300)
plt.show()

# ------------------------------
# CONVERGENCE PLOTS
# ------------------------------
def cumulative_stats(arr):
    """Compute cumulative mean for convergence plotting."""
    arr_clean = np.where(np.isnan(arr), 0, arr)  # Replace NaN with 0 for cumulative mean
    cum_mean = np.cumsum(arr_clean) / np.arange(1, len(arr)+1)
    return cum_mean

# Compute cumulative mean for each parameter
cum_lcoe_25 = cumulative_stats(lcoe_25y)
cum_lcoe_60 = cumulative_stats(lcoe_60y)

cum_npv_25 = cumulative_stats(npv_25y)
cum_npv_60 = cumulative_stats(npv_60y)

cum_irr_25 = cumulative_stats(irr_25y)
cum_irr_60 = cumulative_stats(irr_60y)
plt.figure(figsize=(12,8))

# LCOE
plt.subplot(3,1,1)
plt.plot(cum_lcoe_25, label="25y PV-only", color='tab:blue')
plt.plot(cum_lcoe_60, label="60y PV+BESS", color='tab:orange')
plt.ylabel("LCOE ($/MWh)")
plt.grid(alpha=0.3)
plt.legend()

# NPV
plt.subplot(3,1,2)
plt.plot(cum_npv_25, label="25y PV-only", color='tab:blue')
plt.plot(cum_npv_60, label="60y PV+BESS", color='tab:orange')
plt.ylabel("NPV ($)")
plt.grid(alpha=0.3)
plt.legend()

# IRR
plt.subplot(3,1,3)
plt.plot(cum_irr_25, label="25y PV-only", color='tab:blue')
plt.plot(cum_irr_60, label="60y PV+BESS", color='tab:orange')
plt.xlabel("Monte Carlo Iteration")
plt.ylabel("IRR (%)")
plt.grid(alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "convergence_plot.pdf"))
plt.show()
# ------------------------------
# SENSITIVITY ANALYSIS OF LCOE
# ------------------------------
import numpy as np
import matplotlib.pyplot as plt
import os
from SALib.sample import saltelli
from SALib.analyze import sobol

# ------------------------------
# USER SETTINGS
# ------------------------------
save_folder = "plots"
os.makedirs(save_folder, exist_ok=True)

analysis_life = 60
pv_module_life = 25
hours_per_year = 8760
inverter_life = 11
inverter_frac = 0.06
bess_life = 15
capex_decline = 0.02
build = 1
cap_kW = 1000
cap_MW = cap_kW / 1000

land_cost = 75000 * cap_MW
decom_cost = 368000 * cap_MW
bess_power_kW_i = 200*1000
bess_energy_kWh_i = 4*200*1000

# ------------------------------
# BASE INPUTS
# ------------------------------
base_inputs = {
    "pv_capex_per_kW": 1000,
    "pv_fom_per_kWyr": 18,
    "pv_cf": 0.18,
    "pv_deg": 0.0075,
    "bess_power_cost_per_kW": 120,
    "bess_capex_per_kWh": 300,
    "bess_replace_frac": 0.8,
    "wacc": 0.095,
}

param_names = {
    "pv_capex_per_kW": "Overnight CAPEX",
    "pv_fom_per_kWyr": "Fixed O&M cost",
    "pv_cf": "Capacity factor",
    "pv_deg": "Degradation rate",
    "bess_power_cost_per_kW": "BESS - Power CAPEX",
    "bess_capex_per_kWh": "BESS - Energy CAPEX",
    "bess_replace_frac": "BESS - Replacement cost",
    "wacc": "Discount rate (WACC)",
}

# ------------------------------
# LCOE CALCULATION FUNCTION
# ------------------------------
def calc_lcoe_60(**kwargs):
    r = kwargs.get("wacc", base_inputs["wacc"])
    pv_capex_per_kW = kwargs.get("pv_capex_per_kW", base_inputs["pv_capex_per_kW"])
    pv_fom_per_kWyr = kwargs.get("pv_fom_per_kWyr", base_inputs["pv_fom_per_kWyr"])
    pv_cf = kwargs.get("pv_cf", base_inputs["pv_cf"])
    pv_deg = kwargs.get("pv_deg", base_inputs["pv_deg"])
    bess_power_cost_per_kW = kwargs.get("bess_power_cost_per_kW", base_inputs["bess_power_cost_per_kW"])
    bess_capex_per_kWh = kwargs.get("bess_capex_per_kWh", base_inputs["bess_capex_per_kWh"])
    bess_replace_frac = kwargs.get("bess_replace_frac", base_inputs["bess_replace_frac"])

    annual_kwh_local = cap_kW * pv_cf * hours_per_year
    cashflow_60 = np.zeros(analysis_life + build)
    total_capex = pv_capex_per_kW*cap_kW + land_cost + bess_power_cost_per_kW*bess_power_kW_i + bess_capex_per_kWh*bess_energy_kWh_i
    cashflow_60[0] -= total_capex
    disc_gen = 0.0
    disc_costs = total_capex / (1+r)**0.5

    for y in range(analysis_life):
        t = build + y + 0.5
        pv_gen = annual_kwh_local * (1-pv_deg)**min(y, pv_module_life-1)
        cf_year = -pv_fom_per_kWyr*cap_kW

        if y>0 and y % inverter_life ==0:
            rep_cost_pv = inverter_frac*pv_capex_per_kW*cap_kW*(1-capex_decline)**y
            cf_year -= rep_cost_pv
            disc_costs += rep_cost_pv / (1+r)**t

        if y>0 and y % bess_life ==0:
            rep_cost_bess = bess_replace_frac*(bess_power_cost_per_kW*bess_power_kW_i + bess_capex_per_kWh*bess_energy_kWh_i)
            cf_year -= rep_cost_bess
            disc_costs += rep_cost_bess / (1+r)**t

        cashflow_60[build + y] += cf_year
        disc_costs += pv_fom_per_kWyr*cap_kW / (1+r)**t
        disc_gen += pv_gen / (1+r)**t

    disc_costs += decom_cost / (1+r)**(build + analysis_life + 0.5)
    lcoe = 1000*disc_costs / disc_gen if disc_gen>0 else np.nan
    return lcoe

# ------------------------------
# BASE LCOE
# ------------------------------
base_lcoe = calc_lcoe_60(**base_inputs)

# ------------------------------
# TORNADO PLOT (% CHANGE)
# ------------------------------
tornado_results = {}
for key in base_inputs.keys():
    kwargs_plus = base_inputs.copy()
    kwargs_plus[key] *= 1.1
    lcoe_plus = calc_lcoe_60(**kwargs_plus)

    kwargs_minus = base_inputs.copy()
    kwargs_minus[key] *= 0.9
    lcoe_minus = calc_lcoe_60(**kwargs_minus)

    perc_minus = (lcoe_minus-base_lcoe)/base_lcoe*100
    perc_plus  = (lcoe_plus-base_lcoe)/base_lcoe*100
    tornado_results[key] = (perc_minus, perc_plus)

# Sort by maximum % change
sorted_params = sorted(tornado_results.items(), key=lambda x: max(abs(x[1][0]), abs(x[1][1])), reverse=True)

# Print percentage changes
print("Percentage change in LCOE for +-10% input variation:")
for k,v in sorted_params:
    print(f"{param_names[k]:40s}: -10% -> {v[0]:+.2f}%, +10% -> {v[1]:+.2f}%")

# Plot tornado (%)
fig, ax = plt.subplots(figsize=(10,7))
labels = [param_names[k] for k,_ in sorted_params]
low = [v[0] for _,v in sorted_params]
high = [v[1] for _,v in sorted_params]
y_pos = np.arange(len(labels))
widths = [high[i]-low[i] for i in range(len(low))]
ax.barh(y_pos, widths, left=low, color='skyblue', edgecolor='k')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.set_xlabel("% Change in LCOE")
plt.grid(alpha=0.25, axis='x')
plt.tight_layout()
fig.savefig(os.path.join(save_folder, "tornado_LCOE_60y_percent.pdf"))
fig.savefig(os.path.join(save_folder, "tornado_LCOE_60y_percent.jpg"), dpi=600)
plt.show()

# ------------------------------
# GLOBAL SENSITIVITY ANALYSIS (S1 and ST)
# ------------------------------
problem = {
    'num_vars': len(base_inputs),
    'names': list(base_inputs.keys()),
    'bounds': [[0.9*v, 1.1*v] for v in base_inputs.values()]
}

# Generate samples
param_values = saltelli.sample(problem, 1024, calc_second_order=False)
Y = np.array([calc_lcoe_60(**dict(zip(base_inputs.keys(), row))) for row in param_values])

# Sobol analysis
Si = sobol.analyze(problem, Y, calc_second_order=False, print_to_console=False)

# Print S1 and ST values
print("\nGlobal sensitivity indices:")
for i, name in enumerate(problem['names']):
    print(f"{param_names[name]:40s}: S1 = {Si['S1'][i]:.4f}, ST = {Si['ST'][i]:.4f}")

# Plot S1 and ST
fig2, ax2 = plt.subplots(figsize=(10,6))
x_labels = [param_names[n] for n in problem['names']]
x = np.arange(len(x_labels))
width = 0.35
ax2.bar(x - width/2, Si['S1'], width, color='skyblue', label='S1')
ax2.bar(x + width/2, Si['ST'], width, color='salmon', label='ST')
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels, rotation=45, ha='right')
ax2.set_ylabel("Sobol Index")
ax2.legend()
plt.tight_layout()
fig2.savefig(os.path.join(save_folder, "sobol_LCOE_60y.pdf"))
fig2.savefig(os.path.join(save_folder, "sobol_LCOE_60y.jpg"), dpi=600)
plt.show()













