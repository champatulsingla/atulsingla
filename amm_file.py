import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the differential equations for Farmers, Bandits, and Rulers
def dynastic_model(t, y, r, K, a, b, h, e, m, c, d, f, g):
    F, B, R = y
    
    # Differential equations 
    dF_dt = r * F * (1 - F / K) - (a * F * B) / (b + F) - h * F * R
    dB_dt = (e * a * F * B) / (b + F) - m * B - (c * B * R) / (d + B)
    dR_dt = (f * a * F * B) / (b + F) - g * R
    
    return [dF_dt, dB_dt, dR_dt]

# Baseline parameter values from Table 5.2
params = {
    'r': 1.0, 'K': 1.0, 'a': 1.0, 'b': 0.17,
    'h': 0.1,  # Baseline h (taxation burden)
    'e': 1.2, 'm': 0.4, 'c': 0.4, 'd': 0.42,
    'f': 0.1, 'g': 0.009
}

# Initial population fractions
y0 = [0.7, 0.1, 0.2] # F(0), B(0), R(0)

# Simulate over 300 years
t_span = (0, 300)
t_eval = np.linspace(0, 300, 1000)

# 1. Solve for baseline (h = 0.1) -> Oscillating cycles
sol_oscillating = solve_ivp(
    dynastic_model, t_span, y0, t_eval=t_eval, args=tuple(params.values())
)

# 2. Solve for stable case (h = 2.0) -> Equilibrium
params_stable = params.copy()
params_stable['h'] = 2.0
sol_stable = solve_ivp(
    dynastic_model, t_span, y0, t_eval=t_eval, args=tuple(params_stable.values())
)

# Plotting the results
fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)

def plot_simulation(sol, col_idx, title):
    axes[0, col_idx].plot(sol.t, sol.y[0], color='black', linewidth=1)
    axes[0, col_idx].set_title(title)
    axes[0, col_idx].set_ylabel('F(t) - Farmers')
    axes[0, col_idx].set_ylim(0, 1.05)
    
    axes[1, col_idx].plot(sol.t, sol.y[1], color='black', linewidth=1)
    axes[1, col_idx].set_ylabel('B(t) - Bandits')
    axes[1, col_idx].set_ylim(0, 1.05)
    
    axes[2, col_idx].plot(sol.t, sol.y[2], color='black', linewidth=1)
    axes[2, col_idx].set_ylabel('R(t) - Rulers')
    axes[2, col_idx].set_xlabel('Time (years)')
    axes[2, col_idx].set_ylim(0, 1.05)

# Generate plots
plot_simulation(sol_oscillating, 0, 'Baseline Oscillations (h = 0.1)')
plot_simulation(sol_stable, 1, 'Stable Equilibrium (h = 2.0)')

plt.tight_layout()
plt.show()