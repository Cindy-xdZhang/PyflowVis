import matplotlib.pyplot as plt
import numpy as np

# Example x-axis (epochs/iterations)
x = np.arange(10, 600, 10)

# Dummy y values (replace these with your actual data)
scratch = 60 + 5*np.sin(0.02*x)
scratch_frozen = scratch - 10 + 1*np.random.randn(len(x))
pointmae = 63 + 3*np.sin(0.02*x + 1)
pointmae_frozen = pointmae - 10 + 1*np.random.randn(len(x))
mae = 67 + 2*np.sin(0.02*x + 2)
mae_frozen = mae - 8 + 1*np.random.randn(len(x))

plt.figure(figsize=(8,5))

# Plot each line with style matching the figure
plt.plot(x, scratch, color='black', linewidth=2, label='MLP_Regressor')
plt.plot(x, scratch_frozen, color='black', linewidth=2, linestyle='--', label='FMT_Regressor')

plt.plot(x, pointmae, color='blue', linewidth=2, label='Unet')
plt.plot(x, pointmae_frozen, color='blue', linewidth=2, linestyle='--', label='FMT-Unet')

plt.plot(x, mae, color='red', linewidth=2, label='Vit')
plt.plot(x, mae_frozen, color='red', linewidth=2, linestyle='--', label='FMT-Vit')

# Axes formatting
plt.xlabel("Epoch", fontsize=13)
plt.ylabel("PSNR (dB)", fontsize=16, fontweight='bold')
plt.ylim(55, 80)
plt.xlim(0, 600)

# Legend
plt.legend(frameon=True, fontsize=11, loc='lower right')

# Grid and layout
plt.grid(False)
plt.tight_layout()
plt.show()
print(f"mse={scratch[-1]:.6f}, mae={pointmae[-1]:.6f}, maxe={mae[-1]:.6f}")
