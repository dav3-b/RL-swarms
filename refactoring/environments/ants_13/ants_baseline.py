import random
import sys
from typing import Optional
import time
import warnings

import numpy as np
from scipy.ndimage import gaussian_filter

from gymnasium.spaces import Discrete, MultiBinary, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType

from ants import Ants, AntsVisualizer

def policy(turtle, obs, patches, th):
    max_ph_pos = obs[:patches].argmax()
    max_ph = obs[max_ph_pos]
    
    if max_ph >= th:
        action = 1
    else:
        action = 0

    if turtle['flags'] == [1, 0, 0] or turtle['flags'] == [1, 0, 1]:
        action = 2
    
    return action

def plot(ep, rewards_x_ep, actions_x_ep, ticks, actions_name):
    import matplotlib.pyplot as plt
    import pandas as pd

    data = {
        "Ticks": ticks,
        "Avg_Reward": rewards_x_ep,
    }
    data.update({actions_name[i]: actions_x_ep[:, i] for i in range(actions_x_ep.shape[1])})
    df = pd.DataFrame(data)
    df.to_csv("environments/ants_13/metrics.csv", sep=',', index=False)

    x = np.array([e for e in range(ep)])
    
    avg_reward = rewards_x_ep.mean()
    y = np.array([avg_reward for _ in range(ep)])
    fig = plt.figure(figsize=(10, 5), dpi=200)
    plt.ylabel(f"Avg Reward = {round(avg_reward, 4)}, Std = {round(rewards_x_ep.std(), 4)}")
    plt.scatter(x, rewards_x_ep, s=4, linewidth=1.0, color='#648FFF')
    plt.plot(x, y, label="mean", marker='x', markersize=.5, linewidth=.5, color='#DC267F')
    fig.tight_layout()
    plt.savefig("environments/ants_13/Avg_Reward")
    
    avg_tick = ticks.mean()
    y = np.array([avg_tick for _ in range(ep)])
    fig = plt.figure(figsize=(10, 5), dpi=200)
    plt.ylabel(f"Avg Tick = {round(avg_tick, 4)}, Std = {round(ticks.std(), 4)}")
    plt.scatter(x, ticks, s=4, linewidth=1.0, color='#648FFF')
    plt.plot(x, y, label="mean", marker='x', markersize=.5, linewidth=.5, color='#DC267F')
    fig.tight_layout()
    plt.savefig("environments/ants_13/Avg_Ticks")

    
    fig = plt.figure(figsize=(10, 5), dpi=200)
    for a in range(len(actions_name)):
        plt.plot(
            x,
            actions_x_ep[:, a] * 100,
            label=actions_name[a],
        )
    plt.yticks([i for i in range(0, 110, 10)])
    plt.title("Baseline normalized actions")
    plt.legend()
    plt.savefig("environments/ants_13/Actions")

    
def main():
    params = {
        "learners": 40,
        "actions": [
            "random-walk",
            "move-toward-chemical-0",
            "move-and-drop-chemical-1",
        ],
        "sniff_threshold": 0.9,
        "sniff_patches": 3, 
        "diffuse_area": 0.5,
        "diffuse_radius": 0,
        "follow_mode": "det",
        #"follow_mode": "prob",
        "wiggle_patches": 3,
        "lay_area": 1,
        "lay_amount": 5.0,
        "lay_amount_first": 1, 
        "lay_amount_min": 1.0,
        "ph_decay": 0.9,
        "evaporation": 0.95,
        "obs_type": "paper",
        #"obs_type": "variation1",
        "food_quantity": 3,
        "food_reward": 1,
        "nest_reward": 10,
        "penalty": -0.1,
        "max_episode_ticks": 1000,
        "W": 31,
        "H": 31,
        "PATCH_SIZE": 20,
        "TURTLE_SIZE": 16,
    }

    params_visualizer = {
        "FPS": 10,
        "SHADE_STRENGTH": 10,
        "SHOW_CHEM_TEXT_PH_0": False,
        "SHOW_CHEM_TEXT_PH_1": False,
        "CLUSTER_FONT_SIZE": 12,
        "CHEMICAL_FONT_SIZE": 8,
        "sniff_threshold": 0.0,
        "PATCH_SIZE": 20,
        "TURTLE_SIZE": 16,
        "show_dirs_view": False,
        "wiggle_patches": 5,
        "show_ph_0_view": False,
        "show_ph_1_view": False
    }

    from tqdm import tqdm

    EPISODES = 500
    SEED = 0
    np.random.seed(SEED)
    env = Ants(SEED, **params)
    #env_vis = AntsVisualizer(env.W_pixels, env.H_pixels, **params_visualizer)
    ACTION_NUM = len(params["actions"])
    AGENTS_NUM = env.num_learners 

    actions = (0, 2, 4)

    ticks = np.zeros(EPISODES)
    rewards_x_ep = np.zeros(EPISODES)
    actions_x_ep = np.zeros((EPISODES, ACTION_NUM))

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        tick = 1
        rewards = np.zeros(AGENTS_NUM)
        track_actions = np.zeros(ACTION_NUM)
        #for tick in tqdm(range(params['episode_ticks']), desc="Tick", leave=False):
        while not env.done: 
            actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                observation, reward, _ , _, info = env.last(agent)
                #breakpoint()
                id = env.convert_observation(observation)
                #action = np.random.randint(0, ACTION_NUM)
                #env.step(action)
                #action = random.choice(actions)
                action = policy(env.learners[int(agent)], observation, env.sniff_patches, env.sniff_threshold)
                env.step(action)
                rewards[int(agent)] += round(reward, 4)
                track_actions[action] += 1
                actions[int(agent)] = action
            #env_vis.render(
            #    env.patches,
            #    env.food_pos_1,
            #    env.food_pos_2,
            #    env.food_pos_3,
            #    env.nest_pos,
            #    env.patches_food_1,
            #    env.patches_food_2,
            #    env.patches_food_3,
            #    env.patches_nest,
            #    env.learners,
            #    env.fov,
            #    env.fov_dirs,
            #    env.ph_fov,
            #    env.ph_fov_dirs,
            #    env.min_coord,
            #    env.max_coord,
            #    actions
            #)
            tick += 1
            #breakpoint()
        #avg_cluster = env.avg_cluster()
        #print("Ticks: ", tick)
        ticks[ep - 1] = tick
        rewards_x_ep[ep - 1] = round((rewards.sum() / tick) / AGENTS_NUM, 4)
        actions_x_ep[ep - 1] = (track_actions / tick) / AGENTS_NUM

    print("Total time = ", time.time() - start_time)
    
    plot(EPISODES, rewards_x_ep, actions_x_ep.round(2), ticks, params['actions'])

    env.close()

if __name__ == "__main__":
    main()