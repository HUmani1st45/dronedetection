
from collections import defaultdict
import numpy as np

class SimpleKalman:
    """1D Kalman filter for distance only: state = [d, v]."""
    def __init__(self, q=1.0, r=4.0):
        self.x = np.array([[0.0],[0.0]])  # [distance, velocity]
        self.P = np.eye(2) * 1000.0
        self.F = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * q
        self.R = np.array([[r]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def correct(self, z):
        y = np.array([[z]]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P
        return float(self.x[0,0])

    def update(self, z):
        self.predict()
        return self.correct(z)

class SimpleKalmanBank:
    """Maintains a Kalman filter per track ID."""
    def __init__(self, q=1.0, r=4.0):
        self.filters = defaultdict(lambda: SimpleKalman(q=q, r=r))

    def update(self, track_id:int, z:float):
        return self.filters[track_id].update(z)
