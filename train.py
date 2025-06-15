from agent import Agent
from game import SnakeGameAI
import matplotlib.pyplot as plt

MAX_GAMES = 1000          # Stop training after 1000 games
TARGET_SCORE = 50         # Stop if agent reaches this score
SHOW_PLOT = True          # Set False if you don't want to plot

agent = Agent()
game = SnakeGameAI()

scores = []
mean_scores = []
total_score = 0
record = 0

while agent.n_games < MAX_GAMES:
    # Get current state
    state_old = agent.get_state(game)

    # Get move
    final_move = agent.get_action(state_old)

    # Perform move
    reward, done, score = game.play_step(final_move)
    state_new = agent.get_state(game)

    # Train short memory
    agent.train_short_memory(state_old, final_move, reward, state_new, done)

    # Store transition
    agent.remember(state_old, final_move, reward, state_new, done)

    if done:
        # Game over, reset environment
        game.reset()
        agent.n_games += 1
        agent.train_long_memory()

        if score > record:
            record = score
            agent.model.save()

        scores.append(score)
        total_score += score
        mean_score = total_score / agent.n_games
        mean_scores.append(mean_score)

        print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Mean: {mean_score:.2f}")

        # Stop early if target score reached
        if score >= TARGET_SCORE:
            print(f"✅ Target score {TARGET_SCORE} reached! Training stopped.")
            break

# Plotting
if SHOW_PLOT:
    plt.figure(figsize=(10,5))
    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    plt.xlabel("Game")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Training Progress")
    plt.grid(True)
    plt.show()
