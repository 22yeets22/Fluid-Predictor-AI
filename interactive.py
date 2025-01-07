import matplotlib.pyplot as plt
import torch
from matplotlib import animation
from constants import (
    FPS,
    FRAME_COUNT,
    MODEL_NAME,
    MODEL_VERSION,
    MODELS_PATH,
    SIZE,
    USE_LAST_PREDICTION,
    DRAW_SPEED,
    DRAW_RADIUS,
)
from wave_simulation import WaterWaveSimulation, WaveUNetV2  # noqa: F401

# Load the model
loaded_model = WaveUNetV2()
loaded_model.load_state_dict(torch.load(f"{MODELS_PATH}\\{MODEL_NAME}_{MODEL_VERSION}_final.pth", weights_only=True))
loaded_model.eval()

# Create initial simulation
simulation = WaterWaveSimulation(size=SIZE, disturbances=[])

# Prepare visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
img1 = ax1.imshow(simulation.height, cmap="coolwarm", animated=True, vmin=0, vmax=1)
img2 = ax2.imshow(simulation.height, cmap="coolwarm", animated=True, vmin=0, vmax=1)
ax1.set_title("Actual")
ax2.set_title("Predicted")
plt.colorbar(img1, ax=ax1)
plt.colorbar(img2, ax=ax2)

drawing = True
last_prediction = None


def on_click(event):
    if drawing and event.inaxes == ax1:
        x, y = int(event.xdata), int(event.ydata)
        update_simulation_height(x, y)


def on_motion(event):
    if drawing and event.inaxes == ax1 and event.button == 1:
        x, y = int(event.xdata), int(event.ydata)
        update_simulation_height(x, y)


def update_simulation_height(x, y):
    for i in range(max(0, x - DRAW_RADIUS), min(SIZE, x + DRAW_RADIUS)):
        for j in range(max(0, y - DRAW_RADIUS), min(SIZE, y + DRAW_RADIUS)):
            if (i - x) ** 2 + (j - y) ** 2 <= DRAW_RADIUS**2:
                simulation.height[j, i] = min(1, simulation.height[j, i] + DRAW_SPEED)


def animate(frame):
    global last_prediction
    img1.set_array(simulation.height)

    # Get prediction
    with torch.no_grad():
        input_tensor = (
            torch.from_numpy(simulation.height).float().unsqueeze(0).unsqueeze(0)
            if last_prediction is None or not USE_LAST_PREDICTION
            else last_prediction.float().unsqueeze(0).unsqueeze(0)
        )
        prediction = loaded_model(input_tensor).squeeze(0).squeeze(0)
        last_prediction = prediction

    img2.set_array(prediction.numpy())
    simulation.update()  # Update actual simulation

    return [img1, img2]


def on_key(event):
    global drawing
    if event.key == " ":
        simulation.reset()


# Connect events
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
cid = fig.canvas.mpl_connect("button_press_event", on_click)

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=FRAME_COUNT, interval=int(1000 / FPS), blit=True, repeat=True)
plt.show()
