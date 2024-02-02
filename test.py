import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import TransferFunction, step

numerator = [0.0363, 1]
denominator = [0.009, 0.33, 1]

system = TransferFunction(numerator, denominator)

t = np.linspace(0, 10, 1000)  # Adjust the time range as needed
u = np.ones_like(t)
t, response = step(system, T=t)

# Plot the original and shaped commands
plt.plot(t, u, label='Original Command (az,c)')
plt.plot(t, response, label='Shaped Command (a¯z,c)')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.legend()
plt.title('Original and Shaped Commands')
plt.grid(True)
plt.show()
