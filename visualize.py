from random import randint, uniform

import matplotlib.pyplot as plt
from matplotlib import animation

from constants import FPS, FRAME_COUNT, SIZE
from wave_simulation import WaterWaveSimulation

# Create initial simulation
disturbances = [
    [
        (randint(0, SIZE - 1), randint(0, SIZE - 1)),
        (randint(0, SIZE - 1), randint(0, SIZE - 1)),
        uniform(0.1, 1.0),
    ]
    for _ in range(randint(1, 8))
]
simulation = WaterWaveSimulation(size=SIZE, disturbances=disturbances)

# Prepare visualization
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
img1 = ax1.imshow(simulation.height, cmap="coolwarm", animated=True, vmin=0, vmax=1)
ax1.set_title("Actual")
plt.colorbar(img1)


def animate(frame):
    img1.set_array(simulation.height)
    simulation.update()  # Update actual simulation
    return [img1]


last_prediction = None
anim = animation.FuncAnimation(fig, animate, frames=FRAME_COUNT, interval=int(1000 / FPS), blit=True, repeat=False)
plt.show()
