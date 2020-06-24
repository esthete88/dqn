import sys
import torch
import torch.nn as nn
import tqdm

from tqdm import trange

from loss import compute_td_loss
from utils import is_enough_ram, save_checkpoint, load_checkpoint
from player import play_and_record


class Trainer:
    """Class for training and evaluating agents.

    Parameters
    ----------
    agent : DQNAgent
        Agent for solving the environment.
    env : gym.env
        Gym environment.
    device : torch.device
        Either `cuda` or `cpu`.
    ddqn : bool
        Whether to use double DQN.
    opt : str
        Optimizer.
    opt_kwargs : dict
        Optimizer optional parameters.

    Attributes
    ----------
    target_network : DQNAgent
        Target network for TD loss.
    mean_rw_history : array
        History of mean rewards.
    td_loss_history : array
        History of losses.
    grad_norm_history : array
        History of unclipped gradient norms.
    initial_state_v_history : array
        History of initial state values.
    step : int
        Current step.
    best_reward : type
        Best reward achieved during training.
    """

    def __init__(self, agent, env, device, ddqn=True, opt='adam', opt_kwargs=None):
        self.agent = agent
        self.target_network = agent.__class__(agent.state_shape, agent.n_actions).to(device)
        self.target_network.load_state_dict(agent.state_dict())
        self.env = env
        self.device = device
        self.ddqn = ddqn

        self.step = 0
        self.best_reward = 0

        if opt_kwargs is None:
            opt_kwargs = {'lr': 1e-4}

        if opt == 'adam':
            self.opt = torch.optim.Adam(self.agent.parameters(), **opt_kwargs)
        else:
            raise NotImplementedError

    def fit(self, replay_buffer, batch_size=32, total_steps=10**7, timesteps_per_epoch=1,
            refresh_target_freq=5000, save_freq=10000, max_grad_norm=50,
            n_lives=5, checkpoint_dir=None, callbacks=None):
        """Trains the agent with linearly decreasing epsilon-greedy policy.

        Parameters
        ----------
        replay_buffer : ReplayBuffer
            Experience replay buffer where elements are stored.
        batch_size : int
        total_steps : int
        timesteps_per_epoch : int
            How many steps in the enviroment are stored in the replay buffer per epoch.
        init_epsilon : float
        refresh_target_freq : int
            How often target network parameters are updated.
        save_freq : int
            How often the agent state is saved.
        max_grad_norm : float
            Max value of norm of the gradient.
        n_lives : int
            How many lives the agent has.
        checkpoint_dir : str
            Directory where checkpoints are saved.
        callbacks : list
            List of callbacks that are called each epoch.
        """

        if hasattr(tqdm.tqdm, '_instances'):
            tqdm.tqdm._instances.clear()

        state = self.env.reset()
        for self.step in trange(self.step, total_steps):
            if not is_enough_ram():
                print('Less that 100 Mb RAM available, freezing', file=sys.stderr)
                print('Make sure everything is ok and make KeyboardInterrupt to continue', file=sys.stderr)
                try:
                    while True:
                        pass
                except KeyboardInterrupt:
                    pass

            _, state = play_and_record(state, self.agent, self.env, replay_buffer, timesteps_per_epoch)
            s, a, r, next_s, done = replay_buffer.sample(batch_size)
            loss = compute_td_loss(s, a, r, next_s, done, self.agent, self.target_network, self.device, self.ddqn)

            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
            self.opt.step()

            epoch_log = {
                'epoch': self.step,
                'loss': loss.data.cpu().item(),
                'buffer_length': len(replay_buffer),
                'n_lives': n_lives,
                'agent': self,
            }

            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch_log)

            if self.step % refresh_target_freq == 0:
                # Load agent weights into target_network
                self.target_network.load_state_dict(self.agent.state_dict())

            if checkpoint_dir is not None and self.step % save_freq == 0:
                self.save(checkpoint_dir)

    def save(self, checkpoint_dir):
        state = {
            'agent': self.agent.state_dict(),
            'opt': self.opt.state_dict(),
            'step': self.step,
        }
        save_checkpoint(state, checkpoint_dir)

    def load(self, checkpoint_dir):
        state = load_checkpoint(checkpoint_dir)
        self.agent.load_state_dict(state['agent'])
        self.opt.load_state_dict(state['opt'])
        self.step = state['step']
