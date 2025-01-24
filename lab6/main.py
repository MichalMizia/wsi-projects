import gymnasium as gym
import pandas as pd
import numpy as np
from Agent import Agent
from utils import (
    postprocess,
    plot_q_values_map,
    plot_states_actions_distribution,
    plot_steps_and_rewards,
)


def main():
    env: gym.Env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    n_runs, n_episodes, n_steps = 10, 1000, 100

    agent = Agent(
        env,
        epsilon=1,
        epsilon_decay=(1 / (n_episodes / 2)),
        min_epsilon=0.1,
        learning_rate=0.1,
        discount_factor=0.9,
    )

    # params to plot later
    rewards = np.zeros((n_runs, n_episodes))
    steps = np.zeros((n_runs, n_episodes))
    episodes = np.arange(n_episodes)
    all_states = []
    all_actions = []
    qtables = np.zeros((n_runs, env.observation_space.n, env.action_space.n))  # type: ignore

    for run in range(n_runs):
        agent.reset_q_values()
        for episode in range(n_episodes):
            state, info = env.reset()

            step = 0
            for _ in range(n_steps):
                action = agent.get_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, new_state, action, reward)

                all_states.append(new_state)
                all_actions.append(action)

                if terminated or truncated:
                    break

                state = new_state
                step += 1

            agent.decay_epsilon()

            # set the data to be plotted
            rewards[run, episode] = reward
            steps[run, episode] = step

        qtables[run] = agent.q_values

    # print(agent.q_values)

    qtable = qtables.mean(axis=0)  # average the Q-table between runs
    # print(qtable)

    # plot_states_actions_distribution(
    #     states=all_states, actions=all_actions, map_size=[4, 12]
    # )
    plot_q_values_map(qtable, env, map_size=[4, 12])
    # plot_steps_and_rewards(postprocess(episodes, steps))


if __name__ == "__main__":
    main()
