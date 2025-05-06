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
        self.q_values = np.zeros((self.states, self.actions))
        self.episodes = episodes

    def averageList(self, allLists):
        avgList = []
        for i in range(len(allLists[0])):
            avg = 0
            for j in range(len(allLists)):
                avg += allLists[j][i]
            avgList.append(avg / len(allLists))
        return avgList
    
    def q_learn(self, n):
        allLists = []
        for i in range(n):
            totalRewardsList = []
            for episode in range(self.episodes):
                currentState, _ = self.env.reset()
                truncated = False
                terminated = False
                total_reward = 0
                stepCount = 0
                while not (terminated or truncated) and stepCount < 100:
                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()  
                    else:
                        action = np.argmax(self.q_values[currentState]) 
                    nextState, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward
                    self.q_values[currentState, action] += self.alpha * (reward + self.discount * np.max(self.q_values[nextState]) - self.q_values[currentState, action])
                    currentState = nextState
                    stepCount += 1
                totalRewardsList.append(total_reward)
            allLists.append(totalRewardsList)
        return self.averageList(allLists)

    def sarsa_learn(self, n):
        allLists = []
        for i in range(n):
            totalRewardsList = []
            for episode in range(self.episodes):
                currentState, _ = self.env.reset()
                truncated = False
                terminated = False
                total_reward = 0
                stepCount = 0
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()  
                else:
                    action = np.argmax(self.q_values[currentState]) 
                while not (terminated or truncated) and stepCount < 100:
                    nextState, reward, terminated, truncated, _ = self.env.step(action)
                    total_reward += reward
                    if np.random.rand() < self.epsilon:
                        nextAction = self.env.action_space.sample()  
                    else:
                        nextAction = np.argmax(self.q_values[nextState]) 
                    self.q_values[currentState, action] += self.alpha * (reward + self.discount * self.q_values[nextState, nextAction] - self.q_values[currentState, action])
                    currentState, action = nextState, nextAction
                    stepCount += 1
                totalRewardsList.append(total_reward)
            allLists.append(totalRewardsList)
        return self.averageList(allLists)

    def visualize_policy(self):
        render_env = gym.make('CliffWalking-v0', render_mode='human')
        state, _ = render_env.reset()
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            render_env.render()
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

    QLagent = CliffWalkingEnv(env, alpha, epsilon, discount, episodes)
    SARSAagent = CliffWalkingEnv(env, alpha, epsilon, discount, episodes)
    
    QL_List = QLagent.q_learn(10)
    SARSA_List = SARSAagent.sarsa_learn(10)
 
    plt.plot(QL_List, label='Q-Learning')
    plt.plot(SARSA_List, label='Sarsa')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Performance')
    plt.legend()
    plt.show()
    QLagent.visualize_policy()
    SARSAagent.visualize_policy()
    env.close()
    