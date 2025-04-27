import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
from datetime import datetime
import json
from tqdm import tqdm
import tensorflow as tf
from whist import Whist
from simple_whist_DQN import DQNAgent


class AgentEvaluator:
    def __init__(self, model_path='Models/full_agent_player_0.keras', results_dir='evaluation_results', agent_index=0):
        self.model_path = model_path
        self.results_dir = results_dir
        self.agent_index = agent_index
        self.model_name = os.path.basename(model_path).split('.')[0]

        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)

        # Load model
        self.model = tf.keras.models.load_model(model_path)
        self.dummy_agent = DQNAgent(input_size=None, gamma=None)
        self.dummy_agent.model = self.model
        self.dummy_agent.target_model = self.model

        # Metrics to track
        self.metrics = {
            'win_rate': [],
            'trick_win_rate': [],
            'avg_reward': [],
            'game_lengths': []
        }

        # Detailed game history
        self.game_history = []

    def evaluate(self, num_games=100, verbose=True):
        """Run evaluation for a specified number of games"""
        wins = 0
        total_tricks = 0
        agent_tricks = 0
        total_agent_reward = 0
        total_steps = 0

        env = Whist([1, 2, 3, 4])

        # Use tqdm for progress bar if verbose
        iterator = tqdm(range(num_games)) if verbose else range(num_games)

        for game in iterator:
            state = env.reset()
            done = False
            game_rewards = []
            game_steps = 0
            game_data = {'game_id': game, 'steps': [], 'final_score': None}

            trick_counts = [0, 0, 0, 0]

            while not done:
                # Get current player
                current_player_index = env.count % 4
                current_player = env.players[current_player_index]

                # Get valid actions
                valid_actions = [i for i, value in enumerate(env.player_hand(current_player)) if value != 0]

                if current_player_index == self.agent_index:  # Agent's turn
                    qs = self.dummy_agent.get_qs(state)
                    if valid_actions:
                        # Get Q-values for valid actions
                        valid_q_values = [qs[card] for card in valid_actions if card < len(qs)]

                        if valid_q_values:
                            # Find index of max Q-value within valid actions
                            best_valid_idx = np.argmax(valid_q_values)
                            action = best_valid_idx
                        else:
                            action = np.random.randint(len(valid_actions)) if valid_actions else 0
                    else:
                        action = 0
                else:
                    action = np.random.randint(len(valid_actions)) if valid_actions else 0

                next_state, reward, done = env.step(action)

                # Track agent's reward
                if current_player_index == self.agent_index:
                    game_rewards.append(reward[self.agent_index])
                    total_agent_reward += reward[self.agent_index]

                if current_player_index == self.agent_index:
                    game_rewards.append(reward[self.agent_index])
                    total_agent_reward += reward[self.agent_index]

                # Record step data
                step_data = {
                    'step': game_steps,
                    'player': env.players[current_player_index],
                    'action': action,
                    'reward': reward[current_player_index],
                    'valid_actions_count': len(valid_actions),
                }
                game_data['steps'].append(step_data)
                # print(env.players[current_player_index])

                if env.trick_winner is not None:
                    winner_id = env.trick_winner.id - 1  # Convert to 0-indexed
                    trick_counts[winner_id] += 1
                    total_tricks += 1
                    if winner_id == self.agent_index:
                        agent_tricks += 1
                        # if verbose and game_steps % 2 == 0:  # Only print occasionally to avoid spamming
                        #     print(f"Game {game}: Our agent won a trick")

                state = next_state
                game_steps += 1
                total_steps += 1

            # End of game statistics
            max_tricks = max(trick_counts)
            players_with_max = [i for i, count in enumerate(trick_counts) if count == max_tricks]
            if self.agent_index in players_with_max and len(players_with_max) == 1:
                wins += 1

            # Check if agent won
            # Record final game state
            game_data['final_score'] = trick_counts
            game_data['agent_won'] = trick_counts[self.agent_index] >= max(trick_counts)
            game_data['steps_count'] = game_steps
            game_data['avg_reward'] = np.mean(game_rewards) if game_rewards else 0

            # Add to history
            self.game_history.append(game_data)

            # Update metrics
            self.metrics['win_rate'].append(wins / (game + 1))
            self.metrics['trick_win_rate'].append(agent_tricks / total_tricks if total_tricks > 0 else 0)
            self.metrics['avg_reward'].append(total_agent_reward / total_steps if total_steps > 0 else 0)
            self.metrics['game_lengths'].append(game_steps)

        # After all games, print summary
        if verbose:
            print(f"\n🎯 Evaluation over {num_games} games:")
            print(f" - Agent won {wins} games ({wins / num_games * 100:.1f}%)")
            print(f" - Agent won {agent_tricks} out of {total_tricks} tricks ({agent_tricks / total_tricks * 100:.1f}%)")
            print(f" - Average reward per step: {total_agent_reward / total_steps:.2f}")
            print(f" - Average game length: {total_steps / num_games:.1f} steps")

        return {
            'win_rate': wins / num_games,
            'trick_win_rate': agent_tricks / total_tricks if total_tricks > 0 else 0,
            'avg_reward': total_agent_reward / total_steps if total_steps > 0 else 0,
            'avg_game_length': total_steps / num_games
        }

    def plot_learning_curve(self, save=True):
        """Plot the learning curve showing win rate over games"""
        plt.figure(figsize=(12, 8))

        # Create subplot grid
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Agent Performance Metrics - {self.model_name}', fontsize=16)

        # Win rate over games
        axs[0, 0].plot(self.metrics['win_rate'], color='blue')
        axs[0, 0].set_title('Win Rate Over Games')
        axs[0, 0].set_xlabel('Games')
        axs[0, 0].set_ylabel('Win Rate')
        axs[0, 0].grid(True)

        # Trick win rate
        axs[0, 1].plot(self.metrics['trick_win_rate'], color='green')
        axs[0, 1].set_title('Trick Win Rate Over Games')
        axs[0, 1].set_xlabel('Games')
        axs[0, 1].set_ylabel('Trick Win Rate')
        axs[0, 1].grid(True)

        # Average reward per step
        axs[1, 0].plot(self.metrics['avg_reward'], color='red')
        axs[1, 0].set_title('Average Reward Per Step')
        axs[1, 0].set_xlabel('Games')
        axs[1, 0].set_ylabel('Avg Reward')
        axs[1, 0].grid(True)

        # Game lengths
        axs[1, 1].plot(self.metrics['game_lengths'], color='purple')
        axs[1, 1].set_title('Game Lengths')
        axs[1, 1].set_xlabel('Games')
        axs[1, 1].set_ylabel('Steps')
        axs[1, 1].grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.results_dir}/{self.model_name}_metrics_{timestamp}.png")
            plt.close()
        else:
            plt.show()

    def plot_reward_distribution(self, save=True):
        """Plot the distribution of rewards"""
        all_rewards = []
        for game in self.game_history:
            for step in game['steps']:
                if step['player'] == self.agent_index:  # Only agent rewards
                    all_rewards.append(step['reward'])

        plt.figure(figsize=(10, 6))
        sns.histplot(all_rewards, kde=True)
        plt.title(f"Agent Reward Distribution - {self.model_name}")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.results_dir}/{self.model_name}_reward_dist_{timestamp}.png")
            plt.close()
        else:
            plt.show()

    def plot_action_frequency(self, save=True):
        """Plot frequency of actions taken by the agent"""
        action_counts = {}
        for game in self.game_history:
            for step in game['steps']:
                if step['player'] == self.agent_index:  # Only agent actions
                    action = step['action']
                    if action not in action_counts:
                        action_counts[action] = 0
                    action_counts[action] += 1

        # Sort by action number
        actions = sorted(action_counts.keys())
        counts = [action_counts[a] for a in actions]

        plt.figure(figsize=(12, 6))
        plt.bar(actions, counts)
        plt.title(f"Agent Action Frequency - {self.model_name}")
        plt.xlabel("Action Index")
        plt.ylabel("Frequency")
        plt.xticks(actions)

        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f"{self.results_dir}/{self.model_name}_action_freq_{timestamp}.png")
            plt.close()
        else:
            plt.show()

    def save_metrics(self):
        """Save metrics and game history to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save metrics as CSV
        metrics_df = pd.DataFrame(self.metrics)
        metrics_df.to_csv(f"{self.results_dir}/{self.model_name}_metrics_{timestamp}.csv", index=False)

        # Custom encoder for numpy types
        def convert_np(o):
            if isinstance(o, (np.integer, np.int_)): return int(o)
            if isinstance(o, (np.floating, np.float64)): return float(o)
            if isinstance(o, (np.bool_)): return bool(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)

        # Save game history as JSON
        with open(f"{self.results_dir}/{self.model_name}_history_{timestamp}.json", 'w') as f:
            json.dump(self.game_history, f, indent=2, default=convert_np)

        print(f"Saved metrics and game history to {self.results_dir}/")

    def compare_agents(self, model_paths, num_games=50):
        """Compare performance of multiple agents"""
        results = []

        for model_path in model_paths:
            model_name = os.path.basename(model_path).split('.')[0]
            print(f"\nEvaluating model: {model_name}")

            # Create a new evaluator for this model
            evaluator = AgentEvaluator(model_path=model_path, results_dir=self.results_dir)
            metrics = evaluator.evaluate(num_games=num_games)

            # Add model name to results
            metrics['model_name'] = model_name
            results.append(metrics)

        # Convert to DataFrame for easy comparison
        results_df = pd.DataFrame(results)

        # Plot comparison
        plt.figure(figsize=(14, 10))

        # Create subplot grid
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Agent Performance Comparison', fontsize=16)

        # Win rate comparison
        sns.barplot(x='model_name', y='win_rate', data=results_df, ax=axs[0, 0])
        axs[0, 0].set_title('Win Rate Comparison')
        axs[0, 0].set_xlabel('Model')
        axs[0, 0].set_ylabel('Win Rate')

        # Trick win rate comparison
        sns.barplot(x='model_name', y='trick_win_rate', data=results_df, ax=axs[0, 1])
        axs[0, 1].set_title('Trick Win Rate Comparison')
        axs[0, 1].set_xlabel('Model')
        axs[0, 1].set_ylabel('Trick Win Rate')

        # Average reward comparison
        sns.barplot(x='model_name', y='avg_reward', data=results_df, ax=axs[1, 0])
        axs[1, 0].set_title('Average Reward Comparison')
        axs[1, 0].set_xlabel('Model')
        axs[1, 0].set_ylabel('Avg Reward')

        # Game length comparison
        sns.barplot(x='model_name', y='avg_game_length', data=results_df, ax=axs[1, 1])
        axs[1, 1].set_title('Average Game Length Comparison')
        axs[1, 1].set_xlabel('Model')
        axs[1, 1].set_ylabel('Avg Steps')

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{self.results_dir}/model_comparison_{timestamp}.png")

        # Save results table
        results_df.to_csv(f"{self.results_dir}/model_comparison_{timestamp}.csv", index=False)

        return results_df


# Example usage:
if __name__ == "__main__":
    # Single model evaluation
    evaluator = AgentEvaluator(model_path='Models/full_agent_player_0.keras', agent_index=3)
    # evaluator.evaluate(num_games=50)
    # evaluator.plot_learning_curve()
    # evaluator.plot_reward_distribution()
    # evaluator.plot_action_frequency()
    # evaluator.save_metrics()

    # Compare multiple models
    models_to_compare = [
        'Models/full_agent_player_0.keras',
        'Models/full_agent_player_1.keras',
        'Models/full_agent_player_2.keras',
        'Models/full_agent_player_3.keras'
    ]
    comparison_results = evaluator.compare_agents(models_to_compare, num_games=150)
    print(comparison_results)