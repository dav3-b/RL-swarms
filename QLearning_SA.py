from SlimeEnvV2 import Slime

import json
import numpy as np
import random

PARAMS_FILE = "SlimeEnvV2-params.json"
EPISODES = 10
LOG_EVERY = 1

with open(PARAMS_FILE) as f:
    params = json.load(f)
env = Slime(render_mode="human", **params)

# Q-Learning
alpha = 0.25  # DOC learning rate (0 learn nothing 1 learn suddenly)
gamma = 0.75  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
epsilon = 0.9  # DOC chance of random action
decay = 0.95  # DOC di quanto diminuisce epsilon ogni episode

q_table = np.zeros([4, env.action_space.n])

# NB: la fase di training dura circa 10 minuti con 16 episodi
# TRAINING
print("Start training...")


def state_to_int_map(obs: [bool, bool]):
    if sum(obs) == 0:  # [False, False]
        s = sum(obs)  # 0
    elif sum(obs) == 2:  # [True, True]
        s = 3
    elif int(obs[0]) == 1 and int(obs[1]) == 0:  # [True, False] ==> si trova in un cluster ma non su una patch con feromone --> difficile succeda
        s = 1
    else:
        s = 2  # [False, True]
    return s


for ep in range(1, EPISODES):
    print(f"EPISODE: {ep}")
    print(f"\tepsilon: {epsilon}")
    state, reward, done, info = env.reset()
    s = state_to_int_map(state)
    for tick in range(params['episode_ticks']):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[s])  # Exploit learned values

        next_state, reward, done, info = env.step(action)
        next_s = state_to_int_map(next_state)

        old_value = q_table[s][action]
        next_max = np.max(q_table[s])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[s][action] = new_value

        s = next_s
    epsilon *= decay
    print(q_table)
print("Training finished!\n")


"""Evaluate agent's performance after Q-learning"""
for ep in range(1, EPISODES+1):
    reward_episode = 0
    state, reward, done, info = env.reset()
    state = sum(state)
    print(f"EPISODE: {ep}")
    for tick in range(params['episode_ticks']):
        action = np.argmax(q_table[state])
        state, reward, done, info = env.moving_turtle(action)
        state = sum(state)
        reward_episode += reward
        env.render()
env.close()

print("END")
