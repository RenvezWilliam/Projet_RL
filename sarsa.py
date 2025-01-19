import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def run(episodes = 1000, is_training = True, render = False):
    env = gym.make("BipedalWalker-v3", render_mode='human' if render else None)

    # Division de l'espace d'observation
    hull_angle_speed_space      = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 5)
    angular_velocity_space      = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 2)
    horizontal_speed_space      = np.linspace(env.observation_space.low[2], env.observation_space.high[2], 3)
    pos_joint_1_space           = np.linspace(env.observation_space.low[4], env.observation_space.high[4], 4)
    pos_joint_2_space           = np.linspace(env.observation_space.low[6], env.observation_space.high[6], 4)
    touch_ground_space_1        = np.linspace(env.observation_space.low[8], 1.0, 2)
    pos_joint_3_space           = np.linspace(env.observation_space.low[8], env.observation_space.high[9], 4)
    pos_joint_4_space           = np.linspace(env.observation_space.low[10], env.observation_space.high[11], 4)
    touch_ground_space_2        = np.linspace(env.observation_space.low[13], 1.0, 2)
    lidar_1                     = np.linspace(env.observation_space.low[14], env.observation_space.high[14], 3)
    

    # Division de l'espace d'action
    action_1 = np.linspace(env.action_space.low[0], env.action_space.high[0], 2)
    action_2 = np.linspace(env.action_space.low[1], env.action_space.high[1], 2)
    action_3 = np.linspace(env.action_space.low[2], env.action_space.high[2], 2)
    action_4 = np.linspace(env.action_space.low[3], env.action_space.high[3], 2)

    action_ = []

    for a1 in action_1:
        for a2 in action_2:
            for a3 in action_3:
                for a4 in action_4:
                    action_.append(tuple((float(a1), float(a2), float(a3), float(a4))))
    
    action_ = tuple(action_)

    # Initialisation de la Q-table
    if is_training:
        q = np.zeros((
            len(hull_angle_speed_space),
            len(angular_velocity_space),
            len(horizontal_speed_space),
            len(pos_joint_1_space),
            len(pos_joint_2_space),
            len(touch_ground_space_1),
            len(pos_joint_3_space),
            len(pos_joint_4_space),
            len(touch_ground_space_2),
            len(lidar_1), 
            len(action_)))
    else:
        f = open('bipedal_walker_sarsa.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    alpha = 0.90 # Learning Rate
    gamma = 0.90 # Discounter factor

    epsilon = 1.0                   # Exploration rate
    epsilon_decay_rate = 2/episodes # Rate at which epsilon decreases
    rng = np.random.default_rng()   # Random number generator

    rewards_per_episodes = np.zeros(episodes)

    for i in tqdm(range(episodes)):
        state = env.reset()[0]

        shas    = np.digitize(state[0], hull_angle_speed_space) - 1
        sav     = np.digitize(state[1], angular_velocity_space) - 1
        shs     = np.digitize(state[2], horizontal_speed_space) - 1
        spj1    = np.digitize(state[4], pos_joint_1_space) - 1
        spj2    = np.digitize(state[6], pos_joint_2_space) - 1
        stg1     = np.digitize(state[8], touch_ground_space_1) - 1
        spj3    = np.digitize(state[9], pos_joint_3_space) - 1
        spj4    = np.digitize(state[11], pos_joint_4_space) - 1
        stg2     = np.digitize(state[13], touch_ground_space_2) - 1
        l1      = np.digitize(state[14], lidar_1) - 1

        terminated = False
        truncated = False
        
        best_reward = -999999 
        rewards = 0

        while(not terminated and not truncated):

            # Choix de l'action
            if rng.random() < epsilon and is_training:
                action = env.action_space.sample()

                action_1_idx = np.digitize(action[0], action_1) - 1
                action_2_idx = np.digitize(action[1], action_2) - 1
                action_3_idx = np.digitize(action[2], action_3) - 1
                action_4_idx = np.digitize(action[3], action_4) - 1

                n1, n2, n3, n4 = len(action_1), len(action_2), len(action_3), len(action_4)

                global_idx = (action_1_idx) * (n2 * n3 * n4) + (action_2_idx) * (n3 * n4) + (action_3_idx) * n4 + (action_4_idx)

            else:
                action_idx = q[shas, sav, shs, spj1, spj2, stg1, spj3, spj4, stg2, l1]

                global_idx = np.argmax(action_idx) - 1

                action = action_[global_idx]

            new_state, reward, terminated, truncated, _ = env.step(action)

            nshas    = np.digitize(new_state[0], hull_angle_speed_space) - 1
            nsav     = np.digitize(new_state[1], angular_velocity_space) - 1
            nshs     = np.digitize(new_state[2], horizontal_speed_space) - 1
            nspj1    = np.digitize(new_state[4], pos_joint_1_space) - 1
            nspj2    = np.digitize(new_state[6], pos_joint_2_space) - 1
            nstg1    = np.digitize(new_state[8], touch_ground_space_1) - 1
            nspj3    = np.digitize(new_state[9], pos_joint_3_space) - 1
            nspj4    = np.digitize(new_state[11], pos_joint_4_space) - 1
            nstg2    = np.digitize(new_state[13], touch_ground_space_2) - 1
            nl1      = np.digitize(new_state[14], lidar_1) - 1
            
            if is_training:
                # MISE A JOUR DE LA Q-TABLE -> SARSA : Q(S,A) = Q(S,A) + α * [reward + γ Q(S',A') - Q(S,A)]
                q[shas, sav, shs, spj1, spj2, stg1, spj3, spj4, stg2, l1, global_idx] += alpha * (reward + gamma * (q[nshas, nsav, nshs, nspj1, nspj2, nstg1, nspj3, nspj4, nstg2, nl1, global_idx]) - q[shas, sav, shs, spj1, spj2, stg1, spj3, spj4, stg2, l1, global_idx]) 

            state = new_state

            shas = nshas
            sav = nsav
            shs = nshs
            spj1 = nspj1
            spj2 = nspj2
            spj3 = nspj3
            spj4 = nspj4
            stg1 = nstg1
            stg2 = nstg2
            l1 = nl1

            rewards += reward

            if reward > best_reward:
                best_reward = reward

                if is_training:
                    f = open('bipedal_walker_sarsa.pkl', 'wb')
                    pickle.dump(q, f)
                    f.close()
        

        if not is_training:
            print(f"Episode: {i+1}, Reward: {rewards}")

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        rewards_per_episodes[i] = rewards

    env.close()

    if is_training:
        plt.plot(rewards_per_episodes)
        plt.savefig("BW_SARSA_rpe.png")
        

if __name__ == "__main__":
    run(episodes=1000, is_training=True, render=False)
    run(episodes=1, is_training=False, render=True)
