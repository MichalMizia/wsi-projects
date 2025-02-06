import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def postprocess(episodes, steps):
    """Convert the results of the simulation in dataframes."""
    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=0)})
    return st


def qtable_directions_map(qtable, map_size=[4, 12]):
    """Get the best learned action & map it to arrows."""
    map_height, map_width = map_size  # Unpack map_size into individual dimensions
    qtable_val_max = qtable.max(axis=1).reshape(map_height, map_width)
    qtable_best_action = np.argmax(qtable, axis=1).reshape(map_height, map_width)
    directions = {3: "←", 2: "↓", 1: "→", 0: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = -10  # Minimum float number on the machine
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps and val != 0:
            # Assign an arrow only if a minimal Q-value has been learned as best action
            # otherwise since 0 is a direction, it also gets mapped on the tiles where
            # it didn't actually learn anything
            qtable_directions[idx] = directions[val]
        else:
            qtable_directions[idx] = (
                " "  # Assign an empty space if no action is learned
            )
    qtable_directions = qtable_directions.reshape(map_height, map_width)
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, map_size=[4, 12]):
    """Plot the last frame of the simulation and the policy learned."""
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Plot the last frame
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    # Plot the policy
    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    img_title = f"q_values_map.png"
    savefig_folder = Path("plots")
    fig.savefig(savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_states_actions_distribution(states, actions, map_size=[4, 12]):
    """Plot the distributions of states and actions."""
    labels = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    img_title = f"states_actions_distrib.png"
    # make this relative
    savefig_folder = Path("plots")
    fig.savefig(savefig_folder / img_title, bbox_inches="tight")
    plt.show()


def plot_steps_and_rewards(steps_df):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))

    sns.lineplot(data=steps_df, x="Episodes", y="Steps", ax=ax)
    ax.set(ylabel="Averaged steps number")

    fig.tight_layout()
    img_title = "steps_and_rewards.png"
    savefig_folder = Path("plots")
    fig.savefig(savefig_folder / img_title, bbox_inches="tight")
    plt.show()
