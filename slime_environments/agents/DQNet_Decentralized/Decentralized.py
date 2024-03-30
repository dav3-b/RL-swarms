from slime_environments.environments.SlimeEnvMultiAgent import Slime

import sys
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from slime_environments.agents.utils.utils import read_params, save_env_image, setup, positional_encoding, update_summary, video_from_images
from slime_environments.agents.utils.DQN import DQN, ReplayMemory, optimize_model, select_action

import argparse

import os
import json
import random
import datetime
from collections import namedtuple
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
    
    
def train(env, 
          params, 
          l_params, 
          device, 
          policy_nets, 
          target_nets, 
          train_episodes,
          train_log_every,
          output_file,
          output_dir,
          normalize,
          pos_enc):
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    batch_size = l_params["batch_size"]
    learning_rate = l_params["lr"]
    epsilon_end = l_params["epsilon_end"]
    alpha = l_params["alpha"]
    gamma = l_params["gamma"]
    decay = l_params["decay"]
    n_actions = len(l_params["actions"])
    population = params['population']
    learner_population = params['learner_population']
    update_net_every = l_params['update_net_every']
    memory_capacity = l_params["memory_capacity"]
    
    optimizers = {i: optim.AdamW(policy_nets[i].parameters(), lr=learning_rate, amsgrad=True) for i in range(params['learner_population'])}
    schedulers = {i: StepLR(optimizers[i], step_size=1, gamma=l_params["step_lr"]) for i in range(params['learner_population'])}
    memory = {i: ReplayMemory(Transition, memory_capacity) for i in range(params['learner_population'])}
    
    old_s = {}
    old_a = {}
    cluster_dict = {}
    
    actions_dict = {str(ep): {str(ac): 0 for ac in range(n_actions)} for ep in range(1, train_episodes + 1)}  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
    action_dict = {str(ep): {str(ag): {str(ac): 0 for ac in range(n_actions)} for ag in range(population, population + learner_population)} for ep in range(1, train_episodes + 1)}
    reward_dict = {str(ep): {str(ag): 0 for ag in range(population, population + learner_population)} for ep in range(1, train_episodes + 1)}
    epsilon = 0
    cur_lr = 0
   
    max_possible_reward = (((params['episode_ticks'] - 150)/params['episode_ticks']) * params['rew']) + \
        ((params['learner_population'] / params["cluster_threshold"]) * (params['rew'] ** 2))
    max_possible_pherormone = env.lay_amount * params['learner_population'] * 5

    for ep in tqdm(range(1, train_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        losses = []
        
        # Initialize the environment and get it's state
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc="TICKS", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                next_state, reward, _, _, _ = env.last(agent)
                next_state = torch.tensor(next_state, dtype=torch.float32, device=device)

                if pos_enc:
                    new_pherormone = torch.tensor(env.get_neighborood_chemical(agent).reshape(-1,1), dtype=torch.float32).to(device).unsqueeze(0)
                    pos_encoding = torch.tensor(positional_encoding(new_pherormone.numel(), 2), dtype=torch.float32).to(device).unsqueeze(0)
                    new_pherormone = pos_encoding + new_pherormone 
                else:
                    new_pherormone = torch.tensor(env.get_neighborood_chemical(int(agent), True).reshape(-1,1), dtype=torch.float32).to(device).unsqueeze(0)
                
                #normalization is done considering all the agents in the same patch dropping at the same time pherormone
                if normalize:
                    new_pherormone /= max_possible_pherormone
                    
                next_state = torch.cat((torch.flatten(new_pherormone), next_state)).unsqueeze(0)
                
                if ep == 1 and tick == 1:
                    next_action = env.action_space(agent).sample()
                    next_action = torch.tensor([next_action], dtype=torch.long, device=device).unsqueeze(0)
                else:
                    state = old_s[agent]
                    action = old_a[agent]
                    next_action, policy_nets[int(agent)] = select_action(env, agent, next_state, ep, policy_nets[int(agent)], device, epsilon_end, decay)
                    
                    #normalization is done considering the max reward a single agent can receive
                    reward = torch.tensor([reward], device=device) if not normalize \
                        else torch.tensor([reward / max_possible_reward], device=device)
                    
                    # Store the transition in memory
                    memory[int(agent)].push(state, action, next_state, reward)
                    
                    # Perform one step of the optimization (on the policy network)
                    policy_nets[int(agent)], target_nets[int(agent)], loss_single = optimize_model(Transition, memory[int(agent)], policy_nets[int(agent)], target_nets[int(agent)], gamma, batch_size, device)
                    if loss_single is not None:
                        # Optimize the model
                        optimizers[int(agent)].zero_grad()
                        loss_single.backward()
                        losses.append(torch.Tensor.clone(loss_single.detach()))
                        
                        # In-place gradient clipping
                        torch.nn.utils.clip_grad_value_(policy_nets[int(agent)].parameters(), 100)
                        optimizers[int(agent)].step()
                        schedulers[int(agent)].step()
                    
                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    if (int(agent) + tick * learner_population + ep * params['episode_ticks'] * learner_population) % update_net_every == 0:
                        target_net_state_dict = target_nets[int(agent)].state_dict()
                        policy_net_state_dict = policy_nets[int(agent)].state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key] * alpha + target_net_state_dict[key] * (1 - alpha)
                        target_nets[int(agent)].load_state_dict(target_net_state_dict)
                    
                epsilon = policy_nets[int(agent)].epsilon
                cur_lr = optimizers[int(agent)].param_groups[0]['lr']
                    
                env.step(next_action.item())
                old_s[agent] = next_state
                old_a[agent] = next_action
                
                actions_dict[str(ep)][str(next_action.item())] += 1
                action_dict[str(ep)][str(agent)][str(next_action.item())] += 1
                reward_dict[str(ep)][str(agent)] += round(reward.item(), 2) if isinstance(reward, torch.Tensor) else round(reward, 2)                
                
            env.move()
            env._evaporate()
            env._diffuse()
            image = env.render()
            
            # if ep in [l_params["fist_saveimages_episode"], l_params["middle_saveimages_episode"], l_params["last_saveimages_episode"]]:
            #     if not os.path.exists(os.path.join(output_dir, "images")):
            #             os.makedirs(os.path.join(output_dir, "images"))
            #
            #     if ep == int(l_params["fist_saveimages_episode"]):
            #         save_env_image(image, tick, output_dir, "first_episode")
            #     elif ep == int(l_params["middle_saveimages_episode"]):
            #         save_env_image(image, tick, output_dir, "middle_episode")
            #     elif ep == int(l_params["last_saveimages_episode"]):
            #         save_env_image(image, tick, output_dir, "last_episode")
            #
            # elif ep == int(l_params["fist_saveimages_episode"]) + 1 and tick == 1:
            #     video_from_images(output_dir, "first_episode")
            # elif ep == int(l_params["middle_saveimages_episode"]) + 1 and tick == 1:
            #     video_from_images(output_dir, "middle_episode")
            
            
        cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
        if ep % train_log_every == 0:
            print("EPISODE: {}\tepsilon: {:.5f}\tavg loss: {:.8f}\tlearning rate {:.10f}".format(ep, epsilon, sum(losses)/len(losses), cur_lr))
            update_summary(output_file, ep, params, cluster_dict, actions_dict, action_dict, reward_dict, losses, cur_lr)
            
                    
    #print(json.dumps(cluster_dict, indent=2))
    print("Training finished!\n")
    video_from_images(output_dir, "last_episode")
    
    env.reset()
    now = datetime.datetime.now()
    for agent in range(params['learner_population']):
        policy_model_name = os.path.join(f"policy_{agent}_"  + now.strftime("%m_%d_%Y__%H_%M_%S") + ".pth")
        target_model_name = os.path.join(f"target_{agent}_"  + now.strftime("%m_%d_%Y__%H_%M_%S") + ".pth")
        torch.save(policy_nets[int(agent)].state_dict(), os.path.join(output_dir, "models", "policies", policy_model_name))
        torch.save(target_nets[int(agent)].state_dict(), os.path.join(output_dir, "models", "targets", target_model_name))

    return policy_nets, env


def test(env, params, l_params, policy_nets, test_episodes, test_log_every, device, normalize, pos_enc):
    cluster_dict = {}
    print("[INFO] Start testing...")
    
    epsilon_end = l_params["epsilon_end"]
    epsilon_test = l_params["epsilon_test"]
    decay = l_params["decay"]
    
    max_possible_pherormone = env.lay_amount * params['learner_population'] * 5
    for ep in tqdm(range(1, test_episodes + 1), desc="EPISODES", colour='red', position=0, leave=False):
        env.reset()
        for tick in tqdm(range(1, params['episode_ticks'] + 1), desc=f"TICKS (epsilon: {policy_net.epsilon})", colour='green', position=1, leave=False):
            for agent in env.agent_iter(max_iter=params['learner_population']):
                if ep == 1 and tick == 1:
                    policy_nets[agent].epsilon = epsilon_test
                state, reward, _, _, _ = env.last(agent)
                state = torch.tensor(state, dtype=torch.float32, device=device)

                if pos_enc:
                    new_pherormone = torch.tensor(env.get_neighborood_chemical(int(agent)).reshape(-1,1), dtype=torch.float32).to(device).unsqueeze(0)
                    pos_encoding = torch.tensor(positional_encoding(new_pherormone.numel(), 2), dtype=torch.float32).to(device).unsqueeze(0)
                    new_pherormone = pos_encoding + new_pherormone 
                else:
                    new_pherormone = torch.tensor(env.get_neighborood_chemical(int(agent), True).reshape(-1,1), dtype=torch.float32).to(device).unsqueeze(0)
                
                #normalization is done considering all the agents in the same patch dropping at the same time pherormone
                if normalize:
                    new_pherormone /= max_possible_pherormone
                    
                state = torch.cat((torch.flatten(new_pherormone), state)).unsqueeze(0)
                    
                action, policy_net = select_action(env, agent, state, ep, policy_nets[int(agent)], device, epsilon_end, decay)
                env.step(action)
                
            env.move()
            env._evaporate()
            env._diffuse()
            env.render()
            
        if ep % test_log_every == 0:
            print(f"EPISODE: {ep}")
            print(f"\tepsilon: {policy_net.epsilon}")
            # print(f"\tepisode reward: {reward_episode}")
        cluster_dict[str(ep)] = round(env.avg_cluster(), 2)
        
    print(json.dumps(cluster_dict, indent=2))
    print("Testing finished!\n")


def main(args):
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    params, l_params = read_params(args.params_path, args.learning_params_path)
    curdir = os.path.dirname(os.path.abspath(__file__))
    output_dir, output_file, alpha, gamma, epsilon, decay, train_episodes, train_log_every, test_episodes, test_log_every = setup(args.train, curdir, params, l_params)
    env = Slime(render_mode="human", **params)    
    
    if not os.path.isdir(os.path.join(output_dir, "models")) and args.train:
        os.makedirs(os.path.join(output_dir, "models"))
        
    if not os.path.isdir(os.path.join(output_dir, "models", "policies")) and args.train:
        os.makedirs(os.path.join(output_dir, "models", "policies"))
        
    if not os.path.isdir(os.path.join(output_dir, "models", "targets")) and args.train:
        os.makedirs(os.path.join(output_dir, "models", "targets"))
    
    n_actions = len(l_params["actions"])
    if args.positional_encoding:
        n_observations = 100
    else:
        n_observations = 51
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device selected: {device}")
    
    population = params['population']
    learner_population = params['learner_population']
    policy_nets = {ag: DQN(n_observations, n_actions, epsilon).to(device) for ag in range(population, population + learner_population)}
    target_nets = {ag: DQN(n_observations, n_actions, epsilon).to(device) for ag in range(population, population + learner_population)}
    
    if args.models_path == "" or args.train:
        args.model_path = output_dir
        
    policies_path = os.path.join(args.models_path, "models", "policies")
    targets_path = os.path.join(args.models_path, "models", "targets")
    if args.resume or args.test:
        if os.path.exists(policies_path) and os.path.exists(targets_path):
            policies = [os.path.join(root, file) for root, dirs, files in os.walk(policies_path) for file in files if os.path.isfile(os.path.join(root, file))]
            targets = [os.path.join(root, file) for root, dirs, files in os.walk(targets_path) for file in files if os.path.isfile(os.path.join(root, file))]

            assert len(policies) == params['learner_population'], f"policies weights {len(policies)} and learner population {params['learner_population']} are different!"
            assert len(targets) == params['learner_population'], f"targets weights {len(targets)} and learner population {params['learner_population']} are different!"
            
            for i, file in enumerate(policies):
                policy_nets[i].load_state_dict(torch.load(file), strict=False)
            
            for i, file in enumerate(targets):
                target_nets[i].load_state_dict(torch.load(file), strict=False)
    else:
        for ag in range(population, population + learner_population):
            target_nets[ag].load_state_dict(policy_nets[ag].state_dict())

    if args.train:
        print("[INFO] Start training...")
        train_start = datetime.datetime.now()
        policy_nets, env = train(env, params, l_params, device, policy_nets, target_nets, train_episodes, train_log_every, output_file, output_dir, args.normalize_input, args.positional_encoding)
        train_end = datetime.datetime.now()
        print(f"Training time: {train_end - train_start}")
        
    if args.test:
        print("[INFO] Start testing...")
        test_start = datetime.datetime.now()
        test(env, params, l_params, policy_nets, test_episodes, test_log_every, device, args.normalize_input, args.positional_encoding)
        test_end = datetime.datetime.now()
        print(f"Testing time: {test_end - test_start}")

    env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", type=str, default="dqdec-env-params.json", required=False)
    parser.add_argument("--learning_params_path", type=str, default="dqdec-learning-params.json", required=False)
    parser.add_argument("--policy-model-name", type=str, default="")
    parser.add_argument("--target-model-name", type=str, default="")
    parser.add_argument("--models-path", type=str, default="")
    parser.add_argument("--normalize-input", action="store_true")
    parser.add_argument("--positional-encoding", action="store_true")
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    
    args = parser.parse_args()
    
    assert args.params_path != "" and os.path.isfile(args.params_path) and args.params_path.endswith(".json"), "[ERROR] params path is empty or is not a file or is not a json file"
    assert args.learning_params_path != "" and os.path.isfile(args.learning_params_path) and args.learning_params_path.endswith(".json"), "[ERROR] learning params path is empty or is not a file or is not a json file"
    
    main(args)
