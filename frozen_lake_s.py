import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

def run(episodes = 15000):
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    q = np.zeros((env.observation_space.n, env.action_space.n)) # Initialisation de la q-table

    learning_rate_a = 0.9           # alpha 
    discount_factor_g = 0.9         # gamma 
    epsilon = 1                     # 1 = 100% exploration
    epsilon_decay_rate = 0.0001     # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    for i in tqdm(range(episodes)):
        state = env.reset()[0]  # Reset de l'environnement
        terminated = False      
        truncated = False       

        while(not terminated and not truncated):
            if rng.random() < epsilon:          # Exploration
                action = env.action_space.sample() 
            else:                               # Exploitation
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)

            if rng.random() < epsilon:
                new_action = env.action_space.sample()
            else:
                new_action = np.argmax(q[new_state,:])

            q[state,action] = q[state,action] + learning_rate_a * (
                reward + discount_factor_g * q[new_state,new_action] - q[state,action]
            )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward > 0:
            rewards_per_episode[i] = reward

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake_sarsa.png')

    
    f = open("frozen_lake_s.pkl","wb")
    pickle.dump(q, f)
    f.close()
        
def evaluate_agent(episodes=5):
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')

    # Charger la Q-table entraînée
    with open('frozen_lake_s.pkl', 'rb') as f:
        q = pickle.load(f)

    total_wins = 0

    for episode in range(episodes):
        state = env.reset()[0]
        terminated, truncated = False, False
        step_count = 0

        while not terminated and not truncated:
            action = np.argmax(q[state, :])  # Prendre l'action optimale
            state, reward, terminated, truncated, _ = env.step(action)
            step_count += 1

            if reward == 1:
                total_wins += 1

        print(f"Épisode {episode + 1}/{episodes} terminé en {step_count} étapes.")

    env.close()
    print(f"Nombre de victoires : {total_wins}/{episodes}")

if __name__ == '__main__':
    run(15000) 
    evaluate_agent()
