import torch
import torch.nn as nn
import numpy as np


class DQNAgent(nn.Module):
    """Vanilla DQN Agent from original DeepMind paper"""

    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 1024, 7, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, n_actions),
        )

        self.network.apply(self._init_weights)

    def _init_weights(self, layer):
        if type(layer) in [nn.Linear, nn.Conv2d]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        scaled_state_t = state_t / 255.
        qvalues = self.agent(scaled_state_t)
        return qvalues

    def get_qvalues(self, states):
        """like forward, but works on numpy arrays, not tensors"""
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float32)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

    def get_actions(self, states):
        """pick actions given states"""
        qvalues = self.get_qvalues(states)
        return self.sample_actions(qvalues)


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN agent from 1511.06581"""

    def __init__(self, state_shape, n_actions, epsilon=0):
        super().__init__(state_shape, n_actions, epsilon)
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 1024, 7, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_head = nn.Linear(512, 1)
        self.advantage_head = nn.Linear(512, n_actions)

        self.encoder.apply(self._init_weights)
        self.value_head.apply(self._init_weights)
        self.advantage_head.apply(self._init_weights)

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        scaled_state_t = state_t / 255.
        encoded = self.encoder(scaled_state_t)

        value = self.value_head(encoded[:, :512])
        advantages = self.advantage_head(encoded[:, 512:])

        qvalues = advantages - torch.mean(advantages, dim=1, keepdim=True) + value

        return qvalues
