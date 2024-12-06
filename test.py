from random import randint, uniform

import matplotlib.pyplot as plt
import torch
from matplotlib import animation

from constants import MODEL_NAME, MODEL_VERSION, MODELS_PATH, SIZE, USE_LAST_PREDICTION, FRAME_COUNT, FPS
from wave_simulation import WaterWaveSimulation, WaveUNet

loaded_model = WaveUNet()
loaded_model.load_state_dict(torch.load(f"{MODELS_PATH}\\{MODEL_NAME}_{MODEL_VERSION}_final.pth", weights_only=True))
loaded_model.eval()

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
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
img1 = ax1.imshow(simulation.height, cmap="coolwarm", animated=True, vmin=0, vmax=1)
img2 = ax2.imshow(simulation.height, cmap="coolwarm", animated=True, vmin=0, vmax=1)
ax1.set_title("Actual")
ax2.set_title("Predicted")
plt.colorbar(img1)
plt.colorbar(img2)


def animate(frame):
    global last_prediction
    img1.set_array(simulation.height)

    # Get prediction
    with torch.no_grad():
        if last_prediction is None or not USE_LAST_PREDICTION:
            input_tensor = torch.from_numpy(simulation.height).float().unsqueeze(0).unsqueeze(0)
        else:
            input_tensor = last_prediction.float().unsqueeze(0).unsqueeze(0)

        # Pass through the model
        prediction = loaded_model(input_tensor)

        # Reshape to 2D grid if necessary
        prediction = prediction.squeeze(0).squeeze(0)
        last_prediction = prediction

    img2.set_array(prediction.numpy())

    simulation.update()  # Update actual simulation

    return [img1, img2]


last_prediction = None
anim = animation.FuncAnimation(fig, animate, frames=FRAME_COUNT, interval=int(1000 / FPS), blit=True, repeat=False)
plt.show()
