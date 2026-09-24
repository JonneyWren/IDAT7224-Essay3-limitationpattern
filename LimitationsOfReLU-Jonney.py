import matplotlib.pyplot as plt
import numpy as np

# color - BWG
plt.style.use('grayscale')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pattern1 - Jonney - ReLU Limitations - Dying ReLU ↓↓↓
x = np.linspace(-10, 10, 500)
y = np.maximum(0, x)

ax1.plot(x, y, label=r'$f(x) = \max(0, x)$', color='black', linewidth=2)
# Mark the death zone with a gradient of 0 - Jonney - ↓↓↓
ax1.axvspan(-10, 0, color='gray', alpha=0.3, label='Dead Zone (Gradient = 0)')
ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)

ax1.set_title("The 'Dead Zone' of ReLU", fontsize=14, fontname='Times New Roman')
ax1.set_xlabel("Input $z = Wx + b$", fontsize=12, fontname='Times New Roman')
ax1.set_ylabel("Output", fontsize=12, fontname='Times New Roman')
ax1.legend(prop={'family': 'Times New Roman'})
ax1.grid(True, linestyle=':', alpha=0.7)

# Pattern2 - Jonney - ReLU Limitations - Excessive learning rates lead to changes in the distribution of neuronal necrosis ↓↓↓
np.random.seed(42)  

# normal - before activation
z_healthy = np.random.normal(1, 2.5, 1000)
# after major gradient update
z_dead = np.random.normal(-5, 2.5, 1000)

ax2.hist(np.maximum(0, z_healthy), bins=30, alpha=0.6, color='gray', label='Healthy Activations (Mixed outputs)')
ax2.hist(np.maximum(0, z_dead), bins=30, alpha=0.9, color='black', label='Dead Activations (Collapsed to 0)')

ax2.set_title("Effect of Large Update (Dying ReLU)", fontsize=14, fontname='Times New Roman')
ax2.set_xlabel("ReLU Output Value", fontsize=12, fontname='Times New Roman')
ax2.set_ylabel("Frequency", fontsize=12, fontname='Times New Roman')
ax2.legend(prop={'family': 'Times New Roman'})
ax2.grid(True, linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()