import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm

from tqdm import trange
from IPython.display import clear_output
from PIL import Image

from loss import compute_td_loss
from utils import is_enough_ram, linear_decay, smoothen, save_checkpoint, load_checkpoint
from atari_wrappers import make_env
from player import play_and_record


class Trainer:
    """Module for training and evaluating agents"""

    def __init__(self, agent, env, device, ddqn=True, opt='adam', opt_kwargs=None):
        self.agent = agent
        self.target_network = agent.__class__(agent.state_shape, agent.n_actions).to(device)
        self.target_network.load_state_dict(agent.state_dict())
        self.env = env
        self.device = device
        self.ddqn = ddqn

        self.mean_rw_history = []
        self.td_loss_history = []
        self.grad_norm_history = []
        self.initial_state_v_history = []

        self.step = 0
        self.best_reward = 0

        if opt_kwargs is None:
            opt_kwargs = {'lr': 1e-4}

        if opt == 'adam':
            self.opt = torch.optim.Adam(self.agent.parameters(), **opt_kwargs)
        else:
            raise NotImplementedError

    def fit(self, replay_buffer, batch_size=32, total_steps=10**7, timesteps_per_epoch=1,
            decay_steps=10**6, init_epsilon=1, final_epsilon=0.1, epsilon_decay=None,
            loss_freq=50, refresh_target_freq=5000, eval_freq=5000, save_freq=10000, gif_freq=30000,
            max_grad_norm=50, n_lives=5, checkpoint_dir=None, gif_dir=None):
        if hasattr(tqdm.tqdm, '_instances'):
            tqdm.tqdm._instances.clear()

        if epsilon_decay is None:
            epsilon_decay = linear_decay

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

            self.agent.epsilon = epsilon_decay(init_epsilon, final_epsilon, self.step, decay_steps)

            _, state = play_and_record(state, self.agent, self.env, replay_buffer, timesteps_per_epoch)
            s, a, r, next_s, done = replay_buffer.sample(batch_size)
            loss = compute_td_loss(s, a, r, next_s, done, self.agent, self.target_network, self.device, self.ddqn)

            self.opt.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), max_grad_norm)
            self.opt.step()

            if self.step % loss_freq == 0:
                self.td_loss_history.append(loss.data.cpu().item())
                self.grad_norm_history.append(grad_norm)

            if self.step % refresh_target_freq == 0:
                # Load agent weights into target_network
                self.target_network.load_state_dict(self.agent.state_dict())

            if self.step % eval_freq == 0:
                self._show_progress(replay_buffer, n_lives)

            if checkpoint_dir is not None and self.step % save_freq == 0:
                self.save(checkpoint_dir)

            if gif_dir is not None and self.step % gif_freq == 0:
                rewards, frames = evaluate(make_env(seed=self.step, clip_rewards=False),
                                           self.agent, n_games=n_lives, greedy=True, render=True)
                total_reward = rewards * n_lives
                frames[0].save(gif_dir + '/step={},reward={:.0f}.gif'.format(self.step, total_reward),
                               save_all=True, append_images=frames[1:], duration=30)

    def _show_progress(self, replay_buffer, n_lives):
        self.mean_rw_history.append(evaluate(
            make_env(seed=self.step, clip_rewards=False), self.agent, n_games=n_lives, greedy=True) * 5
        )
        initial_state_q_values = self.agent.get_qvalues(
            [make_env(seed=self.step, clip_rewards=False).reset()]
        )
        self.initial_state_v_history.append(np.max(initial_state_q_values))

        clear_output(True)
        print("buffer size = %i, epsilon = %.5f" %
              (len(replay_buffer), self.agent.epsilon))

        plt.figure(figsize=[16, 9])
        plt.subplot(2, 2, 1)
        plt.title("Mean reward")
        plt.plot(self.mean_rw_history)
        plt.grid()

        assert not np.isnan(self.td_loss_history[-1])
        plt.subplot(2, 2, 2)
        plt.title("TD loss history (smoothened)")
        plt.plot(smoothen(self.td_loss_history))
        plt.grid()

        plt.subplot(2, 2, 3)
        plt.title("Initial state V")
        plt.plot(self.initial_state_v_history)
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.title("Grad norm history (smoothened)")
        plt.plot(smoothen(self.grad_norm_history))
        plt.grid()

        plt.show()

    def reset(self):
        self.mean_rw_history = []
        self.td_loss_history = []
        self.grad_norm_history = []
        self.initial_state_v_history = []

        self.step = 0

    def save(self, checkpoint_dir):
        state = {
            'agent': self.agent.state_dict(),
            'opt': self.opt.state_dict(),
            'step': self.step,
            'mean_rw_history': self.mean_rw_history,
            'td_loss_history': self.td_loss_history,
            'grad_norm_history': self.grad_norm_history,
            'initial_state_v_history': self.initial_state_v_history,
        }
        save_checkpoint(state, checkpoint_dir)

    def load(self, checkpoint_dir):
        state = load_checkpoint(checkpoint_dir)
        self.agent.load_state_dict(state['agent'])
        self.opt.load_state_dict(state['opt'])
        self.step = state['step']
        self.mean_rw_history = state['mean_rw_history']
        self.td_loss_history = state['td_loss_history']
        self.grad_norm_history = state['grad_norm_history']
        self.initial_state_v_history = state['initial_state_v_history']


def evaluate(env, agent, n_games=1, greedy=False, t_max=100000, render=False):
    """ Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward. """
    rewards = []
    frames = []

    for _ in range(n_games):
        s = env.reset()
        reward = 0

        for _ in range(t_max):
            frames.append(Image.fromarray(env.render('rgb_array')))

            qvalues = agent.get_qvalues([s])
            action = qvalues.argmax(axis=-1)[0] if greedy else agent.sample_actions(qvalues)[0]
            s, r, done, _ = env.step(action)
            reward += r
            if done:
                break

        rewards.append(reward)

    if render:
        return np.mean(rewards), frames

    return np.mean(rewards)
