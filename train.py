from random import randint, uniform

import numpy as np
import torch
import torch.nn as nn

from constants import BATCH_SIZE, CHECKPOINT_EVERY, EPOCHS, MODEL_NAME, MODEL_VERSION, MODELS_PATH, SIZE
from wave_simulation import WaterWaveSimulation, WaveUNet


def generate_batch(batch_size):
    batch_data = []
    batch_labels = []

    while len(batch_data) < batch_size:
        disturbances = [
            [
                (randint(0, SIZE - 1), randint(0, SIZE - 1)),
                (randint(0, SIZE - 1), randint(0, SIZE - 1)),
                uniform(0.1, 1.0),
            ]
            for _ in range(randint(1, 8))
        ]

        sim = WaterWaveSimulation(size=SIZE, disturbances=disturbances)
        if randint(1, 50) == 1:
            random_values = np.random.uniform(0.0, 1.0, (SIZE, SIZE))
            sim = WaterWaveSimulation(size=SIZE, disturbances=disturbances, heights=random_values)

        for _ in range(randint(1, batch_size - len(batch_data))):
            current_state = sim.height.flatten()
            sim.update()
            next_state = sim.height.flatten()
            batch_data.append(current_state)
            batch_labels.append(next_state)

    return torch.FloatTensor(np.array(batch_data)), torch.FloatTensor(np.array(batch_labels))


# Initialize Model, Loss, and Optimizer
model = WaveUNet()

# model.load_state_dict(torch.load("models\\wave_predictor_3_final.pth", weights_only=True))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Apply optimizer to model

# Modified training loop
for epoch in range(EPOCHS):
    x_batch, y_batch = generate_batch(BATCH_SIZE)  # Generate new batch each epoch
    x_batch = x_batch.view(BATCH_SIZE, 1, SIZE, SIZE)
    y_batch = y_batch.view(BATCH_SIZE, 1, SIZE, SIZE)

    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.8f}")

    if epoch % CHECKPOINT_EVERY == 0 and epoch > 0:
        print("Saving checkpoint...")
        torch.save(model.state_dict(), f"{MODELS_PATH}\\{MODEL_NAME}_{MODEL_VERSION}_{epoch}.pth")

# Save the trained model
print("Saving final model...")
torch.save(model.state_dict(), f"{MODELS_PATH}\\wave_predictor_{MODEL_VERSION}_final.pth")
