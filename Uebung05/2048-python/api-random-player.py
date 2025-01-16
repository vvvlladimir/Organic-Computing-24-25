import random
import matplotlib.pyplot as plt
from puzzle import GameGrid

def random_player(grid, iterations=100):
    """
    Runs a random player on the game grid.

    :param grid: GameGrid object.
    :param iterations: Number of games to calculate the average score.
    :return: Average score across all games.
    """
    total_score = 0
    for _ in range(iterations):
        grid.reset()
        while not grid.state()["game_over"]:
            direction = random.choice(["up", "down", "left", "right"])
            grid.move(direction)
        total_score += grid.state()["score"]
    return total_score / iterations

# Calculate the average score for different grid sizes
sizes = [2, 3, 4]
average_scores = []

for size in sizes:
    print(f"Running for grid size {size}x{size}...")
    grid = GameGrid(grid_len=size)  # Create an instance of GameGrid with the specified size
    avg_score = random_player(grid, iterations=100)  # Run the random player
    print(f"Average score for grid size {size}x{size}: {avg_score}")
    average_scores.append(avg_score)

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(sizes, average_scores, marker='o', linestyle='-', label="Average Score")
plt.title("Average Score of Random Player for Different Grid Sizes", fontsize=14)
plt.xlabel("Grid Size (NxN)", fontsize=12)
plt.ylabel("Average Score", fontsize=12)
plt.xticks(sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()