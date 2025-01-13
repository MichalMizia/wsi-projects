import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Agent import Agent


def compare_discount_factors():
    env: gym.Env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    n_episodes, n_steps = 100, 100

    agent = Agent(
        env,
        epsilon=1,
        epsilon_decay=(1 / (n_episodes / 2)),
        min_epsilon=0.1,
        learning_rate=0.5,
        discount_factor=0.9,
    )

    discount_factors = np.linspace(0.1, 0.9, 5)
    results = []

    for discount_factor in discount_factors:
        agent.reset_q_values()
        agent.discount_factor = discount_factor

        for episode in range(n_episodes):
            state, info = env.reset()

            total_reward = 0
            step = 0
            for _ in range(n_steps):
                action = agent.get_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, new_state, action, reward)

                if episode == n_episodes - 1:
                    total_reward += reward
                    step += 1

                if terminated or truncated:
                    break

                state = new_state

            agent.decay_epsilon()

        results.append((discount_factor, total_reward, step))

    # Create a table to display the results
    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = [["Discount Factor", "Reward", "Steps"]]
    for result in results:
        table_data.append([f"{result[0]:.2f}", f"{result[1]:.2f}", f"{result[2]}"])

    # Create the table
    table = ax.table(
        cellText=table_data, loc="center", cellLoc="center", colLabels=None
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Reward and Steps in Last Training Episode for Each Discount Factor")
    plt.show()


def compare_learning_rates():
    env: gym.Env = gym.make("CliffWalking-v0", render_mode="rgb_array")
    n_episodes, n_steps = 100, 100

    agent = Agent(
        env,
        epsilon=1,
        epsilon_decay=(1 / (n_episodes / 2)),
        min_epsilon=0.1,
        learning_rate=0.5,
        discount_factor=0.9,
    )

    learning_rates = np.linspace(0.1, 1.0, 4)
    results = []

    for learning_rate in learning_rates:
        agent.reset_q_values()
        agent.lr = learning_rate

        for episode in range(n_episodes):
            state, info = env.reset()

            total_reward = 0
            step = 0
            for _ in range(n_steps):
                action = agent.get_action(state)
                new_state, reward, terminated, truncated, info = env.step(action)
                agent.update(state, new_state, action, reward)

                if episode == n_episodes - 1:
                    total_reward += reward
                    step += 1

                if terminated or truncated:
                    break

                state = new_state

            agent.decay_epsilon()

        results.append((learning_rate, total_reward, step))

    # Create a table to display the results
    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    table_data = [["Learning Rate", "Reward", "Steps"]]
    for result in results:
        table_data.append([f"{result[0]:.2f}", f"{result[1]:.2f}", f"{result[2]}"])

    # Create the table
    table = ax.table(
        cellText=table_data, loc="center", cellLoc="center", colLabels=None
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title("Reward and Steps in Last Training Episode for Each Learning Rate")
    plt.show()


if __name__ == "__main__":
    # compare_discount_factors()
    compare_learning_rates()
