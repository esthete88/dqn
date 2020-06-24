import numpy as np
from PIL import Image


def play_and_record(initial_state, agent, env, exp_replay=None, n_steps=1):
    """
    Play the game for exactly n steps, record every (s,a,r,s', done) to replay buffer.
    Whenever game ends, add record with done=True and reset the game.
    It is guaranteed that env has done=False when passed to this function.

    :returns: return sum of rewards over time and the state in which the env stays
    """
    total_reward = 0.0
    s = initial_state

    for _ in range(n_steps):
        # get agent to pick action given state s
        a = agent.get_actions([s])[0]
        next_s, r, done, _ = env.step(a)

        if exp_replay is not None:
            # store current <s,a,r,s'> transition in buffer
            exp_replay.add(s, a, r, next_s, done)

        s = next_s
        total_reward += r
        if done:
            s = env.reset()

    return total_reward, s


def evaluate(env, agent, n_games=1, greedy=False, t_max=100000, render=False):
    """Plays n_games full games. If greedy, picks actions as argmax(qvalues). Returns mean reward."""
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
