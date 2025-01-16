import random
import matplotlib.pyplot as plt
from puzzle import GameGrid
from q_learning import QLearningAgent

def random_player(grid, episodes=10000, window=100):
    """
    Runs a random player on the game grid and calculates average scores over a sliding window.

    :param grid: GameGrid object.
    :param episodes: Number of episodes to run.
    :param window: Size of the sliding window for averaging scores.
    :return: List of average scores for each window.
    """
    scores = []
    for _ in range(episodes):
        grid.reset()
        while not grid.state()["game_over"]:
            direction = random.choice(["up", "down", "left", "right"])
            grid.move(direction)
        scores.append(grid.state()["score"])

    # Calculate moving average over the sliding window
    averages = []
    for i in range(len(scores) - window + 1):
        window_average = sum(scores[i:i + window]) / window
        averages.append(window_average)

    return averages

def q_learning_player(agent, episodes=10000, window=100):
    """
    Runs a Q-Learning agent on the game grid and calculates average scores over a sliding window.

    :param agent: Pre-trained QLearningAgent object.
    :param episodes: Number of episodes to run.
    :param window: Size of the sliding window for averaging scores.
    :return: List of average scores for each window.
    """
    scores = []
    for _ in range(episodes):
        agent.game.reset()
        while not agent.game.state()["game_over"]:
            state = agent.get_state()
            action = max(agent.q_table.get(state, {}), key=agent.q_table.get(state, {}).get, default='up')
            agent.game.move(action)
        scores.append(agent.game.state()["score"])

    # Calculate moving average over the sliding window
    averages = []
    for i in range(len(scores) - window + 1):
        window_average = sum(scores[i:i + window]) / window
        averages.append(window_average)

    return averages

# Create an instance of the game grid (2x2 field) and Q-learning agent
grid = GameGrid(grid_len=2)
agent = QLearningAgent(grid_len=2)
agent.train(episodes=1000)  # Pre-train the Q-Learning agent

# Run the random player and Q-Learning agent
print("Running random player on a 2x2 grid...")
random_averages = random_player(grid, episodes=10000, window=100)

print("Running Q-Learning agent on a 2x2 grid...")
q_learning_averages = q_learning_player(agent, episodes=10000, window=100)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(random_averages) + 1), random_averages, linestyle='-', label="Random Player")
plt.plot(range(1, len(q_learning_averages) + 1), q_learning_averages, linestyle='-', label="Q-Learning Agent")
plt.title("Comparison of Random Player and Q-Learning Agent (2x2 Grid)", fontsize=14)
plt.xlabel("Window Index", fontsize=12)
plt.ylabel("Average Score", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)
plt.show()