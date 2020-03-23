import torch
import torch.nn as nn
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful

        self.agent = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(1152, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.agent.apply(self._init_weights)

    def _init_weights(self, layer):
        if type(layer) in [nn.Linear, nn.Conv2d]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
#             layer.bias.data.fill_(0.01)

    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch states, shape = [batch_size, *state_dim=4]
        """
        qvalues = self.agent(state_t)
        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
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
        qvalues = self.get_qvalues(states)
        return self.sample_actions(qvalues)
