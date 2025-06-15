from agent import Agent
from game import SnakeGameAI
import torch
import time

# Disable exploration during inference
class InferenceAgent(Agent):
    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = [0, 0, 0]
        move[torch.argmax(prediction).item()] = 1
        return move

def play():
    agent = InferenceAgent()
    agent.model.load()  # Load trained model
    game = SnakeGameAI()

    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)
        reward, done, score = game.play_step(final_move)

        if done:
            print(f"Game over. Final score: {score}")
            game.reset()
            time.sleep(1)

if __name__ == "__main__":
    play()
