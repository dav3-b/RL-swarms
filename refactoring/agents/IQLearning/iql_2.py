import itertools
import random
import numpy as np
from tqdm import tqdm

def create_agent(params: dict, l_params: dict, n_obs, n_actions, train):
    episodes =  l_params["train_episodes"] if train else l_params["test_episodes"]
    # DOC dict che tiene conto della frequenza di scelta delle action per ogni episodio {episode: {action: _, action: _, ...}}
    # Actions:
    #   0: random-walk 
    #   1: drop-chemical 
    #   2: move-toward-chemical 
    #   3: move-away-chemical 
    #   4: walk-and-drop 
    #   5: move-and-drop
    actions_dict = {
        str(ep): {
            str(ac): 0 
            for ac in range(n_actions)
        } for ep in range(1, episodes + 1)
    }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    # DOC dict che tiene conto della frequenza di scelta delle action di ogni agent per ogni episodio {episode: {agent: {action: _, action: _, ...}}}
    action_dict = {
        str(ep): {
            str(ag): {
                str(ac): 0 
                for ac in range(n_actions)
            } for ag in range(params["learners"])
        } for ep in range(1, episodes + 1)
    }
    # DOC dict che tiene conto della reward di ogni agente per ogni episodio {episode: {agent: _}}
    reward_dict = {
        str(ep): {
            str(ag): 0 
            for ag in range(params["learners"])
        }
        for ep in range(1, episodes + 1)
    }
    
    if train:
        # Q-Learning
        # Q_table
        qtable = np.zeros([params["learners"], n_obs, n_actions])
        alpha = l_params["alpha"]  # DOC learning rate (0 learn nothing 1 learn suddenly)
        gamma = l_params["gamma"]  # DOC discount factor (0 care only bout immediate rewards, 1 care only about future ones)
        epsilon = l_params["epsilon"]  # DOC chance of random action
        epsilon_min = l_params["epsilon_min"]  # DOC chance of random action
        decay_type = l_params["decay_type"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
        decay = l_params["decay"]  # DOC di quanto diminuisce epsilon ogni episode (e.g. 1500 episodes => decay = 0.9995)
        return (
            qtable,
            alpha,
            gamma,
            epsilon,
            epsilon_min,
            decay_type,
            decay,
            episodes,
            actions_dict,
            action_dict,
            reward_dict,
        )
    else:
        return (
            episodes,
            actions_dict,
            action_dict,
            reward_dict,
        )

def train(
        env, 
        params:dict, 
        qtable, 
        actions_dict,
        action_dict,
        reward_dict,
        train_episodes:int, 
        train_log_every, 
        alpha:float, 
        gamma:float, 
        decay_type:str,
        decay:float,
        epsilon:float,
        epsilon_min:float,
        print_metrics,
        logger,
        visualizer=None
    ):
    
    n_actions = env.actions_n()
    old_s = {}  # DOC old state for each agent {agent: old_state}
    old_a = {}
    #actions = [4, 5]
    AGENTS_NUM = env.num_learners

    # TRAINING
    print("Start training...\n")

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        tick = 1
        
        while not env.done: 
            actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                cur_state, reward, _, _, _ = env.last(agent)
                cur_s = env.convert_observation(cur_state)

                if ep == 1 and tick == 1:
                    #action = env.action_space(agent).sample()
                    action = np.random.randint(0, n_actions)
                else:
                    # QTable update
                    old_value = qtable[int(agent), old_s[agent], old_a[agent]]
                    next_max = np.max(qtable[int(agent), cur_s])  # QUESTION: was with [action] too
                    new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                    qtable[int(agent), old_s[agent], old_a[agent]] = new_value
                    
                    # next_action
                    if random.uniform(0, 1) < epsilon:
                        action = np.random.randint(0, n_actions)
                        #action = env.action_space(agent).sample()
                    else:
                        action = np.argmax(qtable[int(agent)][cur_s])

                #if env.learners[int(agent)]["mode"] == 's':
                #    env.step(scatter_actions[action].item())
                #else:
                #    env.step(action)
                env.step(action)
                actions[int(agent)] = action

                old_s[agent] = cur_s
                old_a[agent] = action

                actions_dict[str(ep)][str(action)] += 1
                action_dict[str(ep)][str(agent)][str(action)] += 1
                reward_dict[str(ep)][str(agent)] += round(reward, 2)
                
            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.food_pos_1,
                    #env.food_pos_2,
                    #env.food_pos_3,
                    env.nest_pos,
                    #env.reward_patches,
                    env.patches_food_1,
                    #env.patches_food_2,
                    #env.patches_food_3,
                    env.patches_nest,
                    env.learners,
                    env.fov,
                    env.fov_dirs,
                    env.ph_fov,
                    env.ph_fov_dirs,
                    env.min_coord,
                    env.max_coord,
                    actions
                )

            tick += 1

        if decay_type == "log":
            epsilon = max(epsilon * decay, epsilon_min)
        elif decay_type == "linear":
            epsilon = max(epsilon - (1 - decay), epsilon_min)
        
        if ep % train_log_every == 0:
            value = [ep, tick]
            avg_rew = round((sum(reward_dict[str(ep)].values()) / tick) / params["learners"], 4)
            value.append(avg_rew)
            value.extend(list(actions_dict[str(ep)].values()))
            tmp = [list(v.values()) for v in action_dict[str(ep)].values()]
            value.extend(list(itertools.chain(*tmp)))
            eps = round(epsilon, 4)
            value.append(eps)
            
            logger.load_value(value)

            if ep % print_metrics == 0:
                print("\nMetrics ")
                print(" - ticks: ", tick)
                print(" - avg_reward: ", avg_rew)
                print(" - epsilon: ", eps)

    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Training finished!\n")

    return qtable

def eval(
        env,
        params:dict, 
        actions_dict,
        action_dict,
        reward_dict,
        test_episodes:int,
        qtable,
        test_log_every:int,
        logger,
        visualizer=None
    ):
    # DOC Evaluate agent's performance after Q-learning
    #n_actions = env.actions_n()
    #actions = [4, 5]
    AGENTS_NUM = env.num_learners
    #actions = {str(agent): [] for agent in range(AGENTS_NUM)}
    
    print("Start testing...\n")
    
    for ep in tqdm(range(1, test_episodes + 1), desc="EPISODES", colour='red', leave=False):
        env.reset()
        tick = 1
        
        while not env.done: 
            actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                state, reward, _, _, _ = env.last(agent)
                s = env.convert_observation(state)
                action = np.argmax(qtable[int(agent)][s])
                #actions[agent].append(action)
                
                #if env.learners[int(agent)]["mode"] == 's':
                #    env.step(scatter_actions[action].item())
                #else:
                #    env.step(action)
                env.step(action)
                actions[int(agent)] = action
                
                actions_dict[str(ep)][str(action)] += 1
                action_dict[str(ep)][str(agent)][str(action)] += 1
                reward_dict[str(ep)][str(agent)] += round(reward, 2)
                
            if visualizer != None:
                visualizer.render(
                    env.patches,
                    env.food_pos_1,
                    #env.food_pos_2,
                    #env.food_pos_3,
                    env.nest_pos,
                    #env.reward_patches,
                    env.patches_food_1,
                    #env.patches_food_2,
                    #env.patches_food_3,
                    env.patches_nest,
                    env.learners,
                    env.fov,
                    env.fov_dirs,
                    env.ph_fov,
                    env.ph_fov_dirs,
                    env.min_coord,
                    env.max_coord,
                    actions
                )

            tick += 1
        
        if ep % test_log_every == 0:
            value = [ep, tick]
            avg_rew = round((sum(reward_dict[str(ep)].values()) / tick) / params["learners"], 4)
            value.append(avg_rew)
            value.extend(list(actions_dict[str(ep)].values()))
            tmp = [list(v.values()) for v in action_dict[str(ep)].values()]
            value.extend(list(itertools.chain(*tmp)))
            logger.load_value(value)
        
        #breakpoint()
    
    logger.empty_table()
    env.close()
    if visualizer != None:
        visualizer.close()
    print("Testing finished!")