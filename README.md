# Q-Learning Agent - Taxi-v3 (OpenAI Gym)

This project demonstrates a simple implementation of the Q-learning algorithm in Python using the classic Taxi-v3 environment from OpenAI Gym.

## 🧠 Key Concepts
- Tabular Q-Learning
- Epsilon-greedy exploration strategy
- Epsilon decay
- Bellman equation update
- Environment rendering and testing

## 📈 Training
- Episodes: 10,000
- Learning Rate (alpha): 0.9
- Discount Factor (gamma): 0.95
- Epsilon Decay: 0.9995

## 💻 Requirements
- Python 3.x
- OpenAI Gym
- NumPy
- Matplotlib

## 🚀 Running the Code
```bash
import random
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('Taxi-v3')  # choose environment

alpha = 0.9  # learning rate (to what extent new info overrides old info)
gamma = 0.95  # discount (how future actions are important)
epsilon = 1  # randomness of actions
epsilon_decay = 0.9995  # how much it decays while learning
min_epsilon = 0.01  # minimum randomness we want (1%)
num_episodes = 10000  # how many times it trains
max_steps = 100  # max number of steps per one episode to avoid being stuck on one episode

q_table = np.zeros((env.observation_space.n, env.action_space.n))  # initializing q table

def choose_action(state):  # in start, actions should be random, and while it is training, it should start sticking to the q table
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state, :])

'''start training'''
episode_rewards = []

for episodes in range(num_episodes):  # training stage
    state, info = env.reset()  # get a state from the environment
    total_reward = 0
    done = False

    for steps in range(max_steps):
        action = choose_action(state)  # choose an action
        next_state, reward, done, truncated, info = env.step(action)  # get a new state, reward, and know if done
        old_value = q_table[state, action]  # get old q value
        next_max = np.max(q_table[next_state, :])  # get next max q value

        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)  # update the q table

        state = next_state  # update the state
        total_reward += reward

        if done or truncated:  # exit loop if done
            break
    
    episode_rewards.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, min_epsilon)  # update randomness
np.save('q_table_taxi1', q_table)

'''done training'''

'''start testing'''
env = gym.make('Taxi-v3', render_mode='human')  # render the environment

for episode in range(5):
    state, info = env.reset()
    done = False

    for steps in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])  # sticking to the actions from q table
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        
        if done or truncated:
            env.render()
            print(f'Finished the game in episode {episode+1} with a reward of {reward}')  # get feedback
            break

env.close()  # close the environment
