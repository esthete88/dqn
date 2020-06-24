import matplotlib.pyplot as plt
from IPython.display import clear_output

from utils import smoothen
from atari_wrappers import make_env
from player import evaluate


class EpsilonLinearDecayCallback:
    def __init__(self, init_value=1, final_value=0.1, total_steps=1000000):
        self.init_value = init_value
        self.final_value = final_value
        self.total_steps = total_steps

    def __call__(self, epoch_log):
        agent = epoch_log['agent']
        cur_step = agent.step
        if cur_step >= self.total_steps:
            agent.epsilon = self.final_value
        else:
            agent.epsilon = (self.init_value * (self.total_steps - self.cur_step)
                             + self.final_value * cur_step) / self.total_steps


class GifCallback:
    def __init__(self, dir, freq=30000):
        self.dir = dir
        self.freq = freq

    def __call__(self, epoch_log):
        agent = epoch_log['agent']
        if agent.step % self.freq == 0:
            n_lives = epoch_log['n_lives']
            rewards, frames = evaluate(make_env(seed=agent.step, clip_rewards=False),
                                       agent, n_games=n_lives, greedy=True, render=True)
            total_reward = rewards * n_lives
            frames[0].save(self.dir + '/step={},reward={:.0f}.gif'.format(agent.step, total_reward),
                           save_all=True, append_images=frames[1:], duration=30)


class HistoryCallback:
    def __init__(self, loss_freq=50, eval_freq=5000):
        self.loss_freq = loss_freq
        self.eval_freq = eval_freq
        self.mean_rw_history = []
        self.td_loss_history = []

    def __call__(self, epoch_log):
        agent = epoch_log['agent']

        if agent.step % self.loss_freq == 0:
            loss = epoch_log['loss']
            self.td_loss_history.append(loss)

        if agent.step % self.eval_freq == 0:
            buffer_length = epoch_log['buffer_length']
            n_lives = epoch_log['n_lives']
            self.mean_rw_history.append(evaluate(
                make_env(seed=agent.step, clip_rewards=False), agent, n_games=n_lives, greedy=True) * 5
            )

            clear_output(True)
            print("buffer size = %i, epsilon = %.5f" % (buffer_length, agent.epsilon))

            plt.figure(figsize=[16, 5])

            plt.subplot(1, 2, 1)
            plt.title("Mean reward")
            plt.plot(self.mean_rw_history)
            plt.grid()

            plt.subplot(1, 2, 2)
            plt.title("TD loss history (smoothened)")
            plt.plot(smoothen(self.td_loss_history))
            plt.grid()

    def reset(self):
        self.mean_rw_history = []
        self.td_loss_history = []
