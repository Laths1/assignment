import gymnasium as gym
import time
import numpy as np
import matplotlib.pyplot as plt

class CliffWalkingEnv:

    def __init__(self, env, alpha, epsilon, discount, episodes):
        self.actions = env.action_space.n
        self.states = env.observation_space.n
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        self.episodes = episodes

    def averageList(self, allLists):
        return np.mean(allLists, axis=0).tolist()

    def q_learn(self, n):
        allRunsRewards = []
        for _ in range(n):
            self.q_values = np.zeros((self.states, self.actions))
            totalRewards = []
            for episode in range(self.episodes):
                currentState, _ = self.env.reset()
                terminated = False
                truncated = False
                total_reward = 0
                stepCount = 0
                while not (terminated or truncated) and stepCount < 100:
                    # ε-greedy action selection
                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(self.q_values[currentState])
                    
                    nextState, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward

                    # Q-learning update
                    best_next = np.max(self.q_values[nextState])
                    self.q_values[currentState, action] += self.alpha * (reward + self.discount * best_next - self.q_values[currentState, action])
                    
                    currentState = nextState
                    stepCount += 1
                totalRewards.append(total_reward)
            allRunsRewards.append(totalRewards)
        return self.averageList(allRunsRewards)

    def sarsa_learn(self, n):
        allRunsRewards = []
        for _ in range(n):
            self.q_values = np.zeros((self.states, self.actions))
            totalRewards = []
            for episode in range(self.episodes):
                currentState, _ = self.env.reset()
                terminated = False
                truncated = False
                total_reward = 0
                stepCount = 0

                # ε-greedy initial action
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_values[currentState])
                
                while not (terminated or truncated) and stepCount < 100:
                    nextState, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward

                    # ε-greedy next action
                    if np.random.rand() < self.epsilon:
                        nextAction = self.env.action_space.sample()
                    else:
                        nextAction = np.argmax(self.q_values[nextState])

                    # SARSA update
                    self.q_values[currentState, action] += self.alpha * (
                        reward + self.discount * self.q_values[nextState, nextAction] - self.q_values[currentState, action]
                    )
                    
                    currentState, action = nextState, nextAction
                    stepCount += 1
                totalRewards.append(total_reward)
            allRunsRewards.append(totalRewards)
        return self.averageList(allRunsRewards)

    def visualize_policy(self):
        render_env = gym.make('CliffWalking-v0', render_mode='human')
        state, _ = render_env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            time.sleep(0.2)
            action = np.argmax(self.q_values[state])
            state, _, terminated, truncated, _ = render_env.step(action)
        
        render_env.close()


if __name__ == "__main__":
    env = gym.make('CliffWalking-v0')

    epsilon = 0.1
    alpha = 0.1
    discount = 0.99
    episodes = 1000
    runs = 10  

    QLagent = CliffWalkingEnv(env, alpha, epsilon, discount, episodes)
    SARSAagent = CliffWalkingEnv(env, alpha, epsilon, discount, episodes)
    
    QL_List = QLagent.q_learn(runs)
    SARSA_List = SARSAagent.sarsa_learn(runs)
 
    # Plotting
    plt.plot(QL_List, label='Q-Learning')
    plt.plot(SARSA_List, label='SARSA')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Performance of Q-Learning vs SARSA on CliffWalking')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Visualizing Q-Learning policy...")
    QLagent.visualize_policy()

    print("Visualizing SARSA policy...")
    SARSAagent.visualize_policy()

    env.close()
