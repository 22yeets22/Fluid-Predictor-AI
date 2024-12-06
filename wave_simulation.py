import numpy as np
import torch.nn as nn

from constants import SIZE


# Wave simulation class
class WaterWaveSimulation:
    def __init__(self, size=SIZE, dt=0.1, damping=0.99, disturbances=None, heights=None):
        self.size = size
        self.dt = dt  # Time step
        self.damping = damping  # Damping factor

        # Initialize grid
        self.velocity = np.zeros((size, size))  # Wave velocity

        if heights is None:
            self.height = np.zeros((size, size))  # Wave height
        else:
            self.height = heights


        if disturbances is None:
            disturbances = [[(SIZE // 2 - 5, SIZE // 2 - 5), (SIZE // 2 + 5, SIZE // 2 + 5), 1.0]]

        # Create disturbances
        for (p1_x, p1_y), (p2_x, p2_y), height in disturbances:
            if p1_x == p2_x or p1_y == p2_y:
                continue

            # Ensure that p1_x < p2_x and p1_y < p2_y by swapping if needed
            if p1_x > p2_x:
                p1_x, p2_x = p2_x, p1_x
            if p1_y > p2_y:
                p1_y, p2_y = p2_y, p1_y

            # Apply the disturbance to the height array
            self.height[p1_x:p2_x, p1_y:p2_y] = height

    def update(self):
        # Compute Laplacian for the wave equation without wrapping around edges
        laplacian = np.zeros_like(self.height)
        laplacian[1:-1, 1:-1] = (
            self.height[:-2, 1:-1]
            + self.height[2:, 1:-1]
            + self.height[1:-1, :-2]
            + self.height[1:-1, 2:]
            - 4 * self.height[1:-1, 1:-1]
        )

        # Update velocity and height
        self.velocity[1:-1, 1:-1] += laplacian[1:-1, 1:-1] * self.dt
        self.height[1:-1, 1:-1] += self.velocity[1:-1, 1:-1] * self.dt

        # Apply damping
        self.velocity *= self.damping


# Neural Network Predictor for Wave Evolution
class WavePredictor(nn.Module):
    def __init__(self, input_size=SIZE**2):
        super(WavePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4096, input_size),
        )

        """
        # CNN-based architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, 4096),  # Reduced intermediate size
            nn.Unflatten(1, (1, SIZE, 4096)),
            nn.Conv2d(1, 128, kernel_size=5, padding=2),  # Increased filters and kernel
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),  # Increased filters
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=5, padding=2),  # Increased kernel
            nn.Flatten(),
            nn.Linear(4096, input_size),  # Matches the reduced size
        )"""

        """
        # Transformer-based architecture
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.TransformerEncoderLayer(d_model=512, nhead=8),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8), num_layers=3),
            nn.Linear(512, input_size)
        )"""

    def forward(self, x):
        return self.network(x)


class WaveUNet(nn.Module):
    def __init__(self):
        super(WaveUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
