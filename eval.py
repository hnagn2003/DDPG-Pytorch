from typing import Any


class Evaluator:
    def __init__(self, save_path, num_episodes, interval, max_episode_length) -> None:
        self.save_path = save_path
        self.num_episodes = num_episodes
        self.interval = interval
        self.max_episode_length = max_episode_length
    
    def __call__(self, env, policy) -> Any:
        for episode in range(self.num_episodes):
            # TODO: reset env
            
        