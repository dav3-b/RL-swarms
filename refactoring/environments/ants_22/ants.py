import random
import sys
from typing import Optional
import time
import warnings
import math

import numpy as np
from scipy.ndimage import gaussian_filter

from gymnasium.spaces import Discrete, MultiBinary, Box

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType

class Ants(AECEnv):
    def observe(self, agent: str) -> ObsType:
        return np.array(self.observations[agent])

    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
      
    def observations_n(self, same_obs=True):
        if same_obs:
            if isinstance(self.observation_space('0'), MultiBinary):
                return self.observation_space('0').n + 1
            elif isinstance(self.observation_space('0'), Box):
                return self.observation_space('0').shape[0] + 1

    def actions_n(self, same_actions=True):
        if same_actions:
            return self.action_space('0').n.item()

    metadata = {"render_modes": ["human", "server"]}

    def __init__(self, seed, render_mode: Optional[str] = None, **kwargs):
        """
        :param sniff_threshold:     Controls how sensitive slimes are to pheromone (higher values make slimes less
                                    sensitive to pheromone)—unclear effect on learning, could be negligible
        :param diffuse_area         Controls the diffusion radius
        :param follow_mode          Controls how non-learning agents follow pheromone:
                                        'det' = follow greatest pheromone
                                        'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
        :param lay_area:            Controls the radius of the square area sorrounding the turtle where pheromone is laid
        :param lay_amount:          Controls how much pheromone is laid
        :param evaporation:         Controls how much pheromone evaporates at each step
        :param cluster_threshold:   Controls the minimum number of slimes needed to consider an aggregate within
                                    cluster-radius a cluster (the higher the more difficult to consider an aggregate a
                                    cluster)—the higher the more difficult to obtain a positive reward for being within
                                    a cluster for learning slimes
        :param cluster_radius:      Controls the range considered by slimes to count other slimes within a cluster (the
                                    higher the easier to form clusters, as turtles far apart are still counted together)
                                    —the higher the easier it is to obtain a positive reward for being within a cluster
                                    for learning slimes
        :param rew:                 Base reward for being in a cluster
        :param penalty:             Base penalty for not being in a cluster
        :param episode_ticks:       Number of ticks for episode termination
        :param W:                   Window width in # patches
        :param H:                   Window height in # patches
        :param PATCH_SIZE:          Patch size in pixels
        :param TURTLE_SIZE:         Turtle size in pixels
        :param FPS:                 Rendering FPS
        :param SHADE_STRENGTH:      Strength of color shading for pheromone rendering (higher -> brighter color)
        :param SHOW_CHEM_TEXT:      Whether to show pheromone amount on patches (when >= sniff-threshold)
        :param CLUSTER_FONT_SIZE:   Font size of cluster number (for overlapping agents)
        :param CHEMICAL_FONT_SIZE:  Font size of phermone amount (if SHOW_CHEM_TEXT is true)
        :param render_mode:
        """

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        np.random.seed(seed)
        random.seed(seed)
        
        self.num_learners = kwargs['learners'] 
        self.sniff_threshold = kwargs['sniff_threshold']
        self.diffuse_area = kwargs['diffuse_area']
        self.diffuse_radius = kwargs['diffuse_radius']
        self.lay_area = kwargs['lay_area']
        
        self.lay_amount_food = kwargs['lay_amount']
        self.lay_amount = np.array([self.lay_amount_food for _ in range(self.num_learners)])
        self.lay_amount_min = kwargs['lay_amount_min']
        self.lay_amount_first = kwargs['lay_amount_first']
        self.first_drop = np.array([True for _ in range(self.num_learners)])
        
        self.ph_decay = kwargs['ph_decay']
        self.evaporation = kwargs['evaporation']
        self.follow_mode = kwargs['follow_mode']
        self.MAX_TICKS = kwargs['max_episode_ticks']
    
        self.food_quantity = kwargs['food_quantity']

        self.col_rew_weight = kwargs['collective_reward_weight']
        self.food_reward = kwargs['food_reward']
        self.nest_reward = kwargs['nest_reward']
        self.penalty = kwargs['penalty']

        self.W = kwargs['W']
        self.H = kwargs['H']
        assert (self.W == self.H), "Error! W and H must be equal."
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.N_DIRS = 8
        self.sniff_patches = kwargs['sniff_patches']
        self.wiggle_patches = kwargs['wiggle_patches'] 
        assert (
            self.sniff_patches in (1, 3, 5, 7, 8)
        ), "Error! sniff_patches admitted values are: 1, 3, 5, 7, 8."
        assert (
            self.wiggle_patches in (1, 3, 5, 7, 8)
        ), "Error! wiggle_patches admitted values are: 1, 3, 5, 7, 8."
        # Used to calculate the agent's directions.
        # It's a personal convention.

        self.coords = []
        self.offset = self.patch_size // 2
        self.W_pixels = self.W * self.patch_size
        self.H_pixels = self.H * self.patch_size
        
        for x in range(self.offset, (self.W_pixels - self.offset) + 1, self.patch_size):
            for y in range(self.offset, (self.H_pixels - self.offset) + 1, self.patch_size):
                self.coords.append((x, y))  # "centre" of the patch or turtle (also ID of the patch)
        
        self.pos_to_idx = {}
        ids = [(i, j) for i in range(self.W) for j in range(self.H)]
        for p, i in zip(self.coords, ids):
            self.pos_to_idx[p] = i

        self.possible_agents = [str(i) for i in range(self.num_learners)]  # DOC learning agents IDs
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent = self._agent_selector.reset()

        n_coords = len(self.coords)
        self.min_coord = min(self.coords)
        self.max_coord = max(self.coords)
        # patches-own [chemical] - amount of pheromone in each patch
        self.patches = {
            self.coords[i]: {
                'id': i,
                'chemical_0': 0.0,
                'chemical_1': 0.0,
                'turtles': []
            }
            for i in range(n_coords)
        }

        # pre-compute relevant structures to speed-up computation during rendering steps
        # DOC {(x,y): [(x,y), ..., (x,y)]} pre-computed lay area for each patch, including itself
        self.lay_patches = self._find_neighbours(self.lay_area)
        self.lay_patches2 = self._find_neighbours2(self.lay_area)
        self.lay_patches3 = self._find_neighbours3(self.lay_area)
        
        # Agent's field of view
        #self.fov = self._field_of_view(self.wiggle_patches)
        self.fov, self.fov_dirs = self._field_of_view2(self.wiggle_patches)
        #self.fov3 = self._field_of_view3(self.wiggle_patches)
        # Agent's pheromone field of view
        self.ph_fov, self.ph_fov_dirs = self._field_of_view2(self.sniff_patches)
        #self.ph_fov = self._field_of_view(self.sniff_patches)

        self.actions = kwargs['actions']
        self._action_spaces = {
            a: Discrete(len(self.actions))
            for a in self.possible_agents
        }  # DOC 0 = walk, 1 = lay_pheromone, 2 = follow_pheromone
        
        assert kwargs['obs_type'] in ("paper", "variation1", "variation2"), "Error!"
        self.obs_type = kwargs['obs_type']
        # DOC obervation is an array of 8 real elements.
        # This array indicates the pheromone values in the 8 patches around the agent.
        
        if self.obs_type == "paper":
            self._observation_spaces = {
                a: Box(low=0.0, high=np.inf, shape=(self.sniff_patches * 2 + 2,), dtype=np.float32)
                for a in self.possible_agents
            }
        elif self.obs_type == "variation1":
            pass
    
        #self.REWARD_MAX = self.cluster_reward + (((self.cluster_learners - 1) / self.cluster_threshold) * (self.cluster_reward ** 2))

        #self.ph_pos1 = (self.coords[0][0] + self.patch_size * int(self.W * 1/4), self.coords[0][1] + self.patch_size * int(self.H * 1/2)) 
        #self.ph_pos2 = (self.coords[0][0] + self.patch_size * int(self.W * 3/4), self.coords[0][1] + self.patch_size * int(self.H * 1/2)) 
        #self.ph_pos1 = (90, 230)
        #self.ph_pos2 = (350, 230)
        self.food_pos_1 = (self.coords[0][0] + self.patch_size * int(self.W * 1/12), self.coords[0][1] + self.patch_size * int(self.H * 1/12)) 
        self.food_pos_2 = (self.coords[0][0] + self.patch_size * int(self.W * 1/3), self.coords[0][1] + self.patch_size * int(self.H * 3/4)) 
        self.food_pos_3 = (self.coords[0][0] + self.patch_size * int(self.W * 5/6), self.coords[0][1] + self.patch_size * int(self.H * 1/6)) 
        self.food_found = True
        #self.nest_pos = (self.coords[0][0] + self.patch_size * int(self.W * 3/4), self.coords[0][1] + self.patch_size * int(self.H * 3/4)) 
        self.nest_pos = (self.coords[0][0] + self.patch_size * int(self.W * 1/2), self.coords[0][1] + self.patch_size * int(self.H * 1/2)) 
        
        #self.possible_states = 2**4
        self.possible_states = 2**3
        
        # create learners turtle
        self.learners = {
            i: {
                'pos': self.nest_pos,
                'dir': np.random.randint(self.N_DIRS), 
                'flags': [0, 0, 0] 
            }
            for i in range(self.num_learners)
        }
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtles

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.num_learners)))
        )

    def _field_of_view(self, n_patches):
        # Pre-compute every possible agent's direction
        movements = np.array([
            (0, -self.patch_size),                  # dir 0
            (self.patch_size, -self.patch_size),    # dir 1
            (self.patch_size, 0),                   # dir 2
            (self.patch_size, self.patch_size),     # dir 3
            (0, self.patch_size),                   # dir 4
            (-self.patch_size, self.patch_size),    # dir 5
            (-self.patch_size, 0),                  # dir 6
            (-self.patch_size, -self.patch_size),   # dir 7
        ])
        fov = {}
        
        if n_patches < self.N_DIRS:
            central = n_patches // 2
            sliding_window = []
            
            for i in range(self.N_DIRS):
                tmp = []
                for j in range(n_patches):
                    tmp.append((i + j) % self.N_DIRS)
                sliding_window.append(tmp)
            sliding_window = sorted(sliding_window, key=lambda x: x[central])
            
            for c in self.coords:
                tmp_fov = movements + c
                tmp_fov[:, 0] %= self.W_pixels 
                tmp_fov[:, 1] %= self.H_pixels 
                fov[c] = tmp_fov[sliding_window, :]
        else:
            for c in self.coords:
                tmp_fov = movements + c
                tmp_fov[:, 0] %= self.W_pixels 
                tmp_fov[:, 1] %= self.H_pixels 
                fov[c] = tmp_fov

        return fov
    
    def _field_of_view2(self, n_patches):
        # Pre-compute every possible agent's direction
        movements = np.array([
            (0, -self.patch_size),                  # dir 0
            (self.patch_size, -self.patch_size),    # dir 1
            (self.patch_size, 0),                   # dir 2
            (self.patch_size, self.patch_size),     # dir 3
            (0, self.patch_size),                   # dir 4
            (-self.patch_size, self.patch_size),    # dir 5
            (-self.patch_size, 0),                  # dir 6
            (-self.patch_size, -self.patch_size),   # dir 7
        ])
        fov = {}
        fov_dirs = {}
        
        if n_patches < self.N_DIRS:
            central = n_patches // 2
            sliding_window = []
            
            for i in range(self.N_DIRS):
                tmp = []
                for j in range(n_patches):
                    tmp.append((i + j) % self.N_DIRS)
                sliding_window.append(tmp)
            sliding_window = sorted(sliding_window, key=lambda x: x[central])

            for c in self.coords:
                tmp_fov = movements + c
                
                v = tmp_fov[sliding_window,:] 
                fov[c] = v
                
                # Convenzione del momento:
                # se l'agente ha una direzione perpendicolare all'ostacolo inverte la sua direzione,
                # altrimenti viene approssimata con la posizione più vicina.
                x = np.any((v < self.min_coord[0]) | (v > self.max_coord[0]), axis=-1)
                y = np.all(x, axis=-1)
                z = np.where(y == True)[0]
                if z.size != 0:
                    w = (z + 4) % self.N_DIRS   # 4 is the offset to the opposite direction
                    s = np.stack((z, w))
                    fov_dirs[c] = s
                #v[z] = v[w]
        else:
            for c in self.coords:
                v = movements + c
                x = np.any((v < self.min_coord[0]) | (v > self.max_coord[0]), axis=-1)
                z = np.where(x == True)[0]
                e = (z + 4) % self.N_DIRS   # 4 is the offset to the opposite direction
                v[z] = v[e]
                tmp_fov = np.clip(v, self.min_coord[0], self.max_coord[0])  # approximate to the nearest pos
                fov[c] = tmp_fov

        return fov, fov_dirs
    
    def _field_of_view3(self, n_patches):
        # Pre-compute every possible agent's direction
        movements = np.array([
            (0, -self.patch_size),                  # dir 0
            (self.patch_size, -self.patch_size),    # dir 1
            (self.patch_size, 0),                   # dir 2
            (self.patch_size, self.patch_size),     # dir 3
            (0, self.patch_size),                   # dir 4
            (-self.patch_size, self.patch_size),    # dir 5
            (-self.patch_size, 0),                  # dir 6
            (-self.patch_size, -self.patch_size),   # dir 7
        ])
        fov = {}
        
        if n_patches < self.N_DIRS:
            central = n_patches // 2
            sliding_window = []
            
            for i in range(self.N_DIRS):
                tmp = []
                for j in range(n_patches):
                    tmp.append((i + j) % self.N_DIRS)
                sliding_window.append(tmp)
            sliding_window = sorted(sliding_window, key=lambda x: x[central])
            
            for c in self.coords:
                tmp_fov = np.clip(movements + c, self.min_coord[0], self.max_coord[0])
                fov[c] = tmp_fov[sliding_window, :]
        else:
            for c in self.coords:
                tmp_fov = movements + c
                #tmp_fov[:, 0] %= self.W_pixels 
                #tmp_fov[:, 1] %= self.H_pixels 
                fov[c] = tmp_fov

        return fov

    def _find_neighbours(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area'
        """
        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    x, y = self._wrap(x, y)
                    neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

        return neighbours
    
    def _find_neighbours2(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area'
        """
        neighbours = {}
        
        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

        return neighbours
    
    def _find_neighbours3(self, area: int):
        """
        For each patch, find neighbouring patches within square radius 'area'
        """
        neighbours = {}
        min_x = min(self.coords)[0]
        min_y = min(self.coords)[1]
        max_x = max(self.coords)[0]
        max_y = max(self.coords)[1]

        for p in self.patches:
            neighbours[p] = []
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            for x in range(p[0], p[0] + (area * self.patch_size) + 1, self.patch_size):
                for y in range(p[1], p[1] - (area * self.patch_size) - 1, -self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            for x in range(p[0], p[0] - (area * self.patch_size) - 1, -self.patch_size):
                for y in range(p[1], p[1] + (area * self.patch_size) + 1, self.patch_size):
                    if (x >= min_x and y >= min_y) and (x <= max_x and y <= max_y):
                        neighbours[p].append((x, y))
            neighbours[p] = list(set(neighbours[p]))

        return neighbours

    def _wrap(self, x: int, y: int):
        """
        Wrap x,y coordinates around the torus

        :param x: the x coordinate to wrap
        :param y: the y coordinate to wrap
        :return: the wrapped x, y
        """
        return x % self.W_pixels, y % self.H_pixels
    
    def lay_nest_pheromone(self, patches):
        """
        Lay 'amount' pheromone in square 'area' centred in 'pos'
        """
        grid = np.zeros(self.W * self.H)

        for p in self.patches_nest:
            nest_id = patches[p]['id']
            grid[nest_id] = 450
            #grid[nest_id] = 900

        #grid = gaussian_filter(grid.reshape((self.W, self.H)), sigma=7, radius=None, mode='wrap').flatten()
        grid = gaussian_filter(grid.reshape((self.W, self.H)), sigma=12, radius=None, mode='constant').flatten()
        #grid = gaussian_filter(grid.reshape((self.W, self.H)), sigma=25, radius=None, mode='constant').flatten()
        #for p in self.lay_patches[self.nest_pos]:
        #    idx = patches[p]['id']
        #    patches[p]['chemical_1'] = grid[idx]
        for p, g in zip(patches, grid):
            patches[p]['chemical_1'] = g
        
        return patches
    
    def _get_new_positions(self, possible_patches, agent, random_walk=False):
        #breakpoint()
        pos = agent["pos"]
        dir = agent["dir"]
        fov = self.fov_dirs if random_walk else self.ph_fov_dirs

        if pos in fov.keys():
            # Convenzione del momento:
            # se l'agente ha una direzione perpendicolare all'ostacolo inverte la sua direzione,
            # altrimenti viene approssimata con la posizione più vicina.
            i, j = np.where(fov[pos] == dir)
            if i.size != 0 and j.size != 0:
                new_dir = fov[pos][-1][j].item()
                #new_pos = possible_patches[pos][new_dir]
            else:
                new_dir = dir

            new_pos = possible_patches[pos][new_dir]
            #else:
            #    breakpoint()
            #    x = np.any(
            #        (possible_patches[pos][dir] < self.min_coord[0]) | (possible_patches[pos][dir] > self.max_coord[0]), 
            #        axis=-1
            #    )
            #    y = np.where(x == False)[0]
            #    new_pos = possible_patches[pos][dir][y]
            #    new_dir = dir
        else:
            if len(possible_patches[pos].shape) > 2:
                new_pos = possible_patches[pos][dir]
            else:
                new_pos = possible_patches[pos]
            new_dir = dir

        return new_pos, new_dir
    
    def _get_new_direction(self, n_patches, old_dir, idx_dir):
        start = (old_dir - (n_patches // 2)) % self.N_DIRS 
        new_dirs = np.array([(i + start) % self.N_DIRS for i in range(n_patches)])
        return new_dirs[idx_dir]
    
    def _get_idx_dir(self, pos):
        x = np.any((pos < self.min_coord[0]) | (pos > self.max_coord[0]), axis=-1)
        y = np.where(x == False)[0]
        idx_dir = np.random.choice(y)
        return idx_dir

    def _reset_flags(self, agent):
        self.learners[agent]['flags'] = [0, 0, 0]

    """
    def _get_reward(self):
        if self.learners[self.agent]['flags'] == [1, 0, 0]: 
            self.learners[self.agent]['flags'][2] = 1
            self.food_counts[self.learners[self.agent]['pos']] -= 1
            if self.food_counts[self.learners[self.agent]['pos']] == 0:
                if self.learners[self.agent]['pos'] in self.patches_food_1:
                    self.patches_food_1.remove(self.learners[self.agent]['pos'])
                elif self.learners[self.agent]['pos'] in self.patches_food_2:
                    self.patches_food_2.remove(self.learners[self.agent]['pos'])
                elif self.learners[self.agent]['pos'] in self.patches_food_3:
                    self.patches_food_3.remove(self.learners[self.agent]['pos'])
            return self.food_reward
        elif self.learners[self.agent]['flags'] == [1, 0, 1]:
            dist = self.distance_to_goal()
            return -dist
        elif self.learners[self.agent]['pos'] in self.patches_nest and self.learners[self.agent]['flags'] == [1, 1, 1]:
            self._reset_flags(self.agent)
            self.first_drop[self.agent] = True
            return self.nest_reward
        elif self.learners[self.agent]['pos'] in self.patches_nest and self.learners[self.agent]['flags'] == [0, 0, 0]:
            #breakpoint()
            return self.penalty 
        #elif self.learners[self.agent]['pos'] not in self.patches_nest and self.learners[self.agent]['flags'] == [0, 0, 0]:
        #    return 0.0
        else:
            return 0.0
    
    def distance_to_goal(self):
        breakpoint()
        agent_pos = self.learners[self.agent]['pos']
        x = (agent_pos[0] - self.nest_pos[0]) // self.patch_size
        y = (agent_pos[1] - self.nest_pos[1]) // self.patch_size
        dist = round(np.sqrt(x**2 + y**2), 3) # Scalo per la grandezza della patch
        return dist
    """
    def _check_rewards(self):
        if not self.patches_food_1 and not self.patches_food_1_empty:
            self.food_rewards_grid -= self.food_rewards_grid_1
            self.patches_food_1_empty = True
        
        if not self.patches_food_2 and not self.patches_food_2_empty:
            self.food_rewards_grid -= self.food_rewards_grid_2
            self.patches_food_2_empty = True
        
        if not self.patches_food_3 and not self.patches_food_3_empty:
            self.food_rewards_grid -= self.food_rewards_grid_3
            self.patches_food_3_empty = True

    
    def _get_reward(self):
        if self.learners[self.agent]['flags'] == [1, 0, 0]:
            #breakpoint() 
            self.learners[self.agent]['flags'][2] = 1
            
            self.food_counts[self.learners[self.agent]['pos']] -= 1
            if self.food_counts[self.learners[self.agent]['pos']] == 0:
                if self.learners[self.agent]['pos'] in self.patches_food_1:
                    self.patches_food_1.remove(self.learners[self.agent]['pos'])
                elif self.learners[self.agent]['pos'] in self.patches_food_2:
                    self.patches_food_2.remove(self.learners[self.agent]['pos'])
                elif self.learners[self.agent]['pos'] in self.patches_food_3:
                    self.patches_food_3.remove(self.learners[self.agent]['pos'])

            agent_pos = self.learners[self.agent]['pos']
            reward = self.food_reward + round(self.food_rewards_grid[self.pos_to_idx[agent_pos]], 2)  
            self._check_rewards()
            
            return reward
        elif self.learners[self.agent]['flags'] == [1, 0, 1]:
            #breakpoint() 
            agent_pos = self.learners[self.agent]['pos']
            reward = round(self.nest_rewards_grid[self.pos_to_idx[agent_pos]], 2)

            return reward
        elif self.learners[self.agent]['flags'] == [1, 1, 1]:
            #breakpoint() 
            self._reset_flags(self.agent)
            self.first_drop[self.agent] = True
            agent_pos = self.learners[self.agent]['pos']
            reward = self.nest_reward + round(self.nest_rewards_grid[self.pos_to_idx[agent_pos]], 2)
            
            return reward
        #elif self.learners[self.agent]['flags'] == [0, 0, 0]:
        #    agent_pos = self.learners[self.agent]['pos']
        elif self.learners[self.agent]['pos'] in self.patches_nest and self.learners[self.agent]['flags'] == [0, 0, 0]:
            return self.penalty 
        else:
            #breakpoint() 
            agent_pos = self.learners[self.agent]['pos']
            reward = round(self.food_rewards_grid[self.pos_to_idx[agent_pos]], 2)

            return reward

    
    def _get_obs2(self, agent):
        f, _ = self._get_new_positions(self.ph_fov, agent)

        #breakpoint()
        x = np.any((f < self.min_coord[0]) | (f > self.max_coord[0]), axis=-1)
        y = np.where(x == False)[0]
        new_f = f[y]
        #mask = np.logical_not(x).astype(np.uint8)

        obs_ph_0 = np.zeros(self.sniff_patches)
        obs_ph_0[y] += np.array([self.patches[tuple(i)]["chemical_0"] for i in new_f])
        #obs_ph_0.extend([0.0 for _ in range(self.sniff_patches - len(obs_ph_0))])        

        obs_ph_1 = np.zeros(self.sniff_patches)
        obs_ph_1[y] += np.array([self.patches[tuple(i)]["chemical_1"] for i in new_f])
        #obs_ph_1.extend([0.0 for _ in range(self.sniff_patches - len(obs_ph_1))])

        #obs = obs_ph_0 + obs_ph_1
        #obs.extend([0, 0])
        flags = np.zeros(2)
        obs = np.concatenate((obs_ph_0, obs_ph_1, flags))
        
        #if agent['pos'] in self.reward_patches[self.food_pos_1]:
        #    obs.extend([1, 0])
        #elif agent['pos'] in self.reward_patches[self.food_pos_2]:
        #    obs.extend([1, 0])
        #elif agent['pos'] in self.reward_patches[self.food_pos_3]:
        #    obs.extend([1, 0])
        #elif agent['pos'] in self.reward_patches[self.nest_pos]:
        #    obs.extend([0, 1])
        #else:
        #    obs.extend([0, 0])
        
        #if agent['flags'] == [1, 0, 1] or agent['flags'] == [1, 0, 0]:
        #    obs[-2] = 1
        #elif agent['flags'] == [1, 1, 1]:
        #    obs.extend([0, 1])
        #else:
        #    obs.extend([0, 0])
        
        if agent['flags'] != [0, 0, 0]:
            obs[-2] = 1

        if agent['pos'] in self.patches_nest:
            obs[-1] = 1 

        return obs
        #return np.array(obs)
    
    def process_agent(self, cluster_ticks, rewards_cust):
        """
        In this methods we compute the agent's reward and it's observation.
        """
        if self.obs_type == "paper":
            observations = self._get_obs2(self.learners[self.agent])
        elif self.obs_type == "variation1":
            observations = self._get_obs3(self.learners[self.agent])

        reward = self._get_reward()
        rewards_cust[self.agent].append(reward)
        #reward = self.distance_to_goal()
        #rewards_cust[self.agent].append(-reward)


        #if reward == 100:
        #    self._reset_flags(self.agent)
        #breakpoint()
        return observations, cluster_ticks, rewards_cust

    def _check_pos(self, turtle):
        #if turtle['pos'] == self.ph_pos1 and turtle['flags'][0] == 0: 
        #    turtle['flags'][0] = 1
        #elif turtle['pos'] == self.ph_pos2 and turtle['flags'][0] == 1: 
        #    turtle['flags'][1] = 1

        if turtle['pos'] in self.patches_food_1 and turtle['flags'] == [0, 0, 0]: 
            #if self.agent == 2:
            #    breakpoint()
            #breakpoint()
            #self.rewards_cust[self.agent].append(0.0)
            turtle['flags'][0] = 1
        elif turtle['pos'] in self.patches_food_2 and turtle['flags'] == [0, 0, 0]: 
        #    #if self.agent == 2:
        #    #    breakpoint()
        #    #breakpoint()
        #    #self.rewards_cust[self.agent].append(0.0)
            turtle['flags'][0] = 1
        elif turtle['pos'] in self.patches_food_3 and turtle['flags'] == [0, 0, 0]: 
        #    #if self.agent == 2:
        #    #    breakpoint()
        #    #breakpoint()
        #    #self.rewards_cust[self.agent].append(0.0)
            turtle['flags'][0] = 1
        elif turtle['pos'] in self.patches_nest and turtle['flags'] == [1, 0, 1]: 
            #if self.agent == 2:
            #    breakpoint()
            #breakpoint()
            #self.rewards_cust[self.agent].append(0.0)
            turtle['flags'][1] = 1

        return turtle

    def _walk2(self, patches, turtle):
        """
        Action 0: move in random direction (8 sorrounding cells)
        """      
        f, direction = self._get_new_positions(self.fov, turtle, True)
        patches[turtle['pos']]['turtles'].remove(self.agent)
        idx_dir = self._get_idx_dir(f)
        turtle["pos"] = tuple(f[idx_dir])
        patches[turtle['pos']]['turtles'].append(self.agent)

        if self.wiggle_patches < self.N_DIRS:
            turtle["dir"] = self._get_new_direction(self.wiggle_patches, direction, idx_dir)
        else:
            turtle["dir"] = idx_dir
        
        turtle = self._check_pos(turtle)

        return patches, turtle

    def do_action0(self):
        self.patches, self.learners[self.agent] = self._walk2(self.patches, self.learners[self.agent])

    def _find_max_pheromone2(self, agent, obs):
        """
        Following pheromone modeis controlled by param self.follow_mode:
            'det' = follow greatest pheromone
            'prob' = follow greatest pheromone probabilistically (pheromone strength as weight)
        """
        # Det = follow greatest pheromone
        f, direction = self._get_new_positions(self.ph_fov, agent)

        #if self.follow_mode == "prob": 
        #    total = obs.sum()
        #    if total == 0.0:
        #        probs = np.ones_like(obs) / obs.shape[0]
        #    else:
        #        probs = obs / obs.sum()
        #    idx = np.random.choice(np.arange(obs.shape[0]), p=probs)
        #else:
        #    idx = obs.argmax()
        
        
        #if np.any((f < self.min_coord[0]) | (f > self.max_coord[0])):
        x = np.any((f < self.min_coord[0]) | (f > self.max_coord[0]), axis=-1)
        y = np.where(x == False)[0]
        idx = y[obs[y].argmax()]
        #z = np.where(y == idx)[0]
        ph_pos = tuple(f[idx])
        ph_val = obs[idx]
        #ph_pos = tuple(new_f[idx])
        #else:
        #    ph_pos = tuple(f[idx])
        
        if self.sniff_patches < self.N_DIRS:
            ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
        else:
            ph_dir = idx
        return ph_val, ph_pos, ph_dir

    def _follow_pheromone2(self, patches, ph_coords, ph_dir, turtle):
        """
        Action 2: move turtle towards greatest pheromone found
        """
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        
        turtle = self._check_pos(turtle)
        
        return patches, turtle

    def lay_pheromone(self, patches, pos):
        # Bisogna calcolarlo per ogni agente
        #breakpoint()
        
        if self.first_drop[self.agent] and (self.learners[self.agent]['flags'] == [1, 0, 0] or self.learners[self.agent]['flags'] == [1, 0, 1]):
            ph = self.lay_amount_food * self.lay_amount_first
            self.first_drop[self.agent] = False 
        elif self.learners[self.agent]['flags'] == [1, 0, 0] or self.learners[self.agent]['flags'] == [1, 0, 1]:
            ph = max(self.lay_amount[self.agent] * self.ph_decay, self.lay_amount_min)
            #self.lay_amount[self.agent] = ph
            #breakpoint()
        else:
            ph = self.lay_amount_min #0.0 

        self.lay_amount[self.agent] = ph 
        
        #for p in self.lay_patches[pos]:
        for p in self.lay_patches3[pos]:
            patches[p]['chemical_0'] +=  ph #self.lay_amount[self.agent]

        return patches
                
    def do_action1(self):
        if self.obs_type == "paper":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                self.learners[self.agent],
                self.observations[str(self.agent)][:self.sniff_patches]        
            )
            if max_pheromone >= self.sniff_threshold:
                self.patches, self.learners[self.agent] = self._follow_pheromone2(
                    self.patches,
                    max_coords,
                    max_ph_dir,
                    self.learners[self.agent]
                )
            #else:
            #    self.do_action0()
        elif self.obs_type == "variation1":
            pass
    
    def do_action2(self):
        if self.obs_type == "paper":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                self.learners[self.agent],
                self.observations[str(self.agent)][self.sniff_patches:-2]        
            )
            if max_pheromone >= self.sniff_threshold:
                self.patches, self.learners[self.agent] = self._follow_pheromone2(
                    self.patches,
                    max_coords,
                    max_ph_dir,
                    self.learners[self.agent]
                )
            #else:
            #    self.do_action0()
        elif self.obs_type == "variation1":
            pass
    
    def do_action3(self):
        if self.obs_type == "paper":
            max_pheromone, max_coords, max_ph_dir = self._find_max_pheromone2(
                self.learners[self.agent],
                self.observations[str(self.agent)][self.sniff_patches:-2]        
            )
            if max_pheromone >= self.sniff_threshold:
                self.patches = self.lay_pheromone(self.patches, self.learners[self.agent]['pos'])
                self.patches, self.learners[self.agent] = self._follow_pheromone2(
                    self.patches,
                    max_coords,
                    max_ph_dir,
                    self.learners[self.agent]
                )
            #else:
            #    self.do_action0()
        elif self.obs_type == "variation1":
            pass

    def _find_non_max_pheromone(self, agent, obs):
        #breakpoint()
        f, direction = self._get_new_positions(self.ph_fov, agent)
        ids = np.where(obs < self.sniff_threshold)[0]
        x = np.any((f < self.min_coord[0]) | (f > self.max_coord[0]), axis=-1)
        y = np.where(x == False)[0]
        idx = y[obs[y].argmin()]
        
        #if ids.shape[0] == 0:
        #    idx = obs.argmin()
        #else:
        #    idx = np.random.choice(ids)

        ph_pos = tuple(f[idx])
        
        if self.sniff_patches < self.N_DIRS:
            ph_dir = self._get_new_direction(self.sniff_patches, direction, idx)
        else:
            ph_dir = idx
            #ph_dir = idx % self.sniff_patches
        return ph_pos, ph_dir

    def _avoid_pheromone(self, patches, ph_coords, ph_dir, turtle):
        """
        Avoid the pheromone.
        """
        patches[turtle['pos']]['turtles'].remove(self.agent)
        turtle["pos"] = ph_coords
        patches[turtle['pos']]['turtles'].append(self.agent)
        turtle["dir"] = ph_dir
        
        turtle = self._check_pos(turtle)
        
        return patches, turtle

    #def do_action4(self):
    #    if np.any(self.observations[str(self.agent)] >= self.sniff_threshold):
    #        if self.obs_type == "paper":
    #            ph_pos, ph_dir = self._find_non_max_pheromone(
    #                self.learners[self.agent], 
    #                self.observations[str(self.agent)][:self.sniff_patches]        
    #            )
    #            self.patches, self.learners[self.agent] = self._avoid_pheromone(
    #                self.patches,
    #                ph_pos,
    #                ph_dir,
    #                self.learners[self.agent]
    #            )
    #        elif self.obs_type == "variation1":
    #            pass
    #    else:
    #        self.do_action0()
    
    def do_action4(self):
        if np.any(self.observations[str(self.agent)] >= self.sniff_threshold):
            if self.obs_type == "paper":
                ph_pos, ph_dir = self._find_non_max_pheromone(
                    self.learners[self.agent], 
                    self.observations[str(self.agent)][self.sniff_patches:-2]        
                )
                self.patches, self.learners[self.agent] = self._avoid_pheromone(
                    self.patches,
                    ph_pos,
                    ph_dir,
                    self.learners[self.agent]
                )
            elif self.obs_type == "variation1":
                pass
        else:
            self.do_action0()
    
    def _get_global_reward_1(self, rewards):
        # agents average reward 
        return np.array(rewards).mean()
    
    def _get_global_reward_2(self, rewards):
        # agents average reward 
        return min(rewards) #np.array(rewards).min()

    def _save_rewards(self):
        rewards = [self.rewards_cust[self.agent_name_mapping[ag]][-1] for ag in self.agents]
        global_reward = self._get_global_reward_2(rewards)
        
        for ag in self.agents:
            ind_reward = self.rewards_cust[self.agent_name_mapping[ag]][-1]
            reward = (1 - self.col_rew_weight) * ind_reward + self.col_rew_weight * global_reward
            self.rewards[ag] = round(reward, 2)

    def _diffuse_and_evaporate(self, patches):
        """
        This diffuse method use a gaussian filter for the process.
        This is a kind of parallel diffusion.
        Evaporates pheromone from each patch according to param self.evaporation
        """
        #breakpoint()
        # Diffusion
        grid0 = np.array([patches[p]["chemical_0"] for p in patches.keys()]).reshape((self.W, self.H))
        #grid1 = np.array([patches[p]["chemical_1"] for p in patches.keys()]).reshape((self.W, self.H))
        
        if self.diffuse_radius == 0:
            grid0 = gaussian_filter(grid0, sigma=self.diffuse_area, mode='constant')
            #grid0 = gaussian_filter(grid0, sigma=self.diffuse_area, mode="wrap")
            #grid1 = gaussian_filter(grid1, sigma=self.diffuse_area, mode="wrap")
        else:
            grid0 = gaussian_filter(grid0, sigma=self.diffuse_area, radius=self.diffuse_radius, mode='constant')
            #grid1 = gaussian_filter(grid1, sigma=self.diffuse_area, radius=self.diffuse_radius, mode="wrap")
        
        grid0 = grid0.flatten()
        #grid1 = grid1.flatten()
        # Evaporation
        grid0 *= self.evaporation
        #grid1 *= self.evaporation
        # Write values
        #for p, g0, g1 in zip(patches, grid0, grid1):
        #    patches[p]['chemical_0'] = g0
        #    patches[p]['chemical_1'] = g1
        for p, g in zip(patches, grid0):
            patches[p]['chemical_0'] = g
        
        return patches
    
    def _check_food(self):
        if(
            len(self.patches_food_1) == 0 and
            len(self.patches_food_2) == 0 and
            len(self.patches_food_3) == 0
        ):
            self.no_food = True

    def _check_termination(self):
        self._check_food()

        tmp = []
        for agent in self.learners: 
            if self.learners[agent]['flags'] == [0, 0, 0]:
                tmp.append(True)
            else:
                tmp.append(False)
        
        if (all(tmp) and self.no_food) or self.current_ticks == self.MAX_TICKS:
            #breakpoint()
            self.done = True

    def step(self, action: int):
        if(self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        self.agent = self.agent_name_mapping[self.agent_selection]  # ID of agent

        if action == 0:     # Random walk
            self.do_action0()   
        elif action == 1:   # Follow pheromone 0
            self.do_action1()
        elif action == 2:   # Follow pheromone 1
            self.do_action2()
        elif action == 3:   # Follow pheromone 1 and drop pheromone 0
            self.do_action3()
        #elif action == 4:   # Avoid pheromone 1
        #    self.do_action4()
        else:
            raise ValueError("Action out of range!")

        self.observations[str(self.agent)], self.cluster_ticks, self.rewards_cust = self.process_agent(
            self.cluster_ticks,
            self.rewards_cust,
        )

        if self._agent_selector.is_last():
            self._save_rewards()
            #for ag in self.agents:
            #    self.rewards[ag] = self.rewards_cust[self.agent_name_mapping[ag]][-1]

            self.patches = self._diffuse_and_evaporate(self.patches)
            self.current_ticks += 1
        else:
            self._clear_rewards()
         
        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[str(self.agent)] = 0
        self._accumulate_rewards()

        self._check_termination()
        
    def _get_dense_rewards(self, pos, val, area, radius=None):
        reward_grid = np.zeros((self.W, self.H))

        for p in pos:
            reward_grid[self.pos_to_idx[p]] = val  
        
        reward_grid = gaussian_filter(reward_grid, sigma=area, radius=radius, mode='constant')
        return reward_grid


    def reset(self, seed=None, return_info=True, options=None):
        """
        Reset env.
        """
        # empty stuff
        #Different from AECEnv attribute self.rewards - only keeps last step rewards
        self.rewards_cust = {i: [] for i in range(self.num_learners)}
        self.cluster_ticks = {i: 0 for i in range(self.num_learners)}

        self.current_ticks = 1
        
        #Initialize attributes for PettingZoo Env
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: None for agent in self.agents}
        
        # re-position learner turtle
        for l in self.learners:
            self.patches[self.learners[l]['pos']]['turtles'].remove(l)
            self.learners[l]['pos'] = self.nest_pos
            self._reset_flags(l)
            self.patches[self.learners[l]['pos']]['turtles'].append(l)  # DOC id of learner turtle

        tmp = self._find_neighbours(1)
        self.patches_food_1 = tmp[self.food_pos_1]
        self.patches_food_1_empty = False
        tmp = self._find_neighbours(1)
        self.patches_food_2 = tmp[self.food_pos_2]
        self.patches_food_2_empty = False
        tmp = self._find_neighbours(1)
        self.patches_food_3 = tmp[self.food_pos_3]
        self.patches_food_3_empty = False
        tmp = self._find_neighbours(1)
        self.patches_nest = tmp[self.nest_pos]
        
        self.no_food = False
        self.done = False

        food_pos = []
        food_pos.extend(self.patches_food_1)
        food_pos.extend(self.patches_food_2)
        food_pos.extend(self.patches_food_3)
        self.food_counts = {p: self.food_quantity for p in food_pos}

        self.food_rewards_grid_1 = self._get_dense_rewards(self.patches_food_1, 30.0, 4.0)
        self.food_rewards_grid_2 = self._get_dense_rewards(self.patches_food_2, 10.0, 2.0)
        self.food_rewards_grid_3 = self._get_dense_rewards(self.patches_food_3, 20.0, 3.0)
        self.food_rewards_grid = self.food_rewards_grid_1 + self.food_rewards_grid_2 + self.food_rewards_grid_3
        self.nest_rewards_grid = self._get_dense_rewards(self.patches_nest, 40.0, 5.0)

        # patches-own [chemical] - amount of pheromone in the patch
        for p in self.patches:
            self.patches[p]['chemical_0'] = 0.0
            #self.patches[p]['chemical_1'] = 0.0
        self.patches = self.lay_nest_pheromone(self.patches)

        if self.obs_type == "paper":
            self.observations = {
                a: np.zeros(self.sniff_patches * 2 + 2, dtype=np.float32)
                for a in self.agents
            }
        elif self.obs_type == "variation1":
            pass
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
    
    def convert_observation(self, obs):
        """
        This method returns the conversion of the observation to an integer.
        It's useful for IQL.
        """
        #if not obs.any():
        #    obs_id = 0
        #else:
        #    if self.obs_type == "paper":
        #        if np.unique(obs).shape[0] == 1:
        #            obs_id = np.random.randint(self.sniff_patches * 2) + 1
        #        else:
        #            obs_id = obs.argmax() + 1
        #    elif self.obs_type == "variation1":
        #        pass

        obs_ph_0 = obs[:self.sniff_patches]
        obs_ph_1 = obs[self.sniff_patches:self.sniff_patches * 2]
        food = int(obs[-2])
        nest = int(obs[-1])
        
        if np.any(obs_ph_0 >= self.sniff_threshold):
            obs_ph_0_id = 1
        else:
            obs_ph_0_id = 0
        
        #if np.any(obs_ph_1 >= self.sniff_threshold):
        #    obs_ph_1_id = 1
        #else:
        #    obs_ph_1_id = 0

        obs_id = (obs_ph_0_id * 2**2) + (food * 2**1) + (nest * 2**0)
        
        return obs_id
    
    def mask_actions(self, obs):
        actions_num = len(self.actions)
        actions = [i for i in range(actions_num)]
        food = bool(int(obs[-2]))

        if not food:
            actions.remove(3)              

        if np.all(obs[:self.sniff_patches] < self.sniff_threshold):
            actions.remove(1)
        
        return np.array(actions, dtype=np.uint8)
    
import pygame

BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
SKY_BLUE = (0, 127, 255)
WHITE = (255, 255, 255)
RED = (190, 0, 0)
PINK = (255, 20, 147)
PURPLE = (128, 0, 145)
GREEN = (0, 190, 0)
YELLOW = (250, 250, 0)
ORANGE = (255, 128, 0) 

class AntsVisualizer:
    def __init__(
        self,
        W_pixels,
        H_pixels,
        **kwargs
    ):
        self.fps = kwargs['FPS']
        self.shade_strength = kwargs['SHADE_STRENGTH']
        self.show_chem_text_ph_0 = kwargs['SHOW_CHEM_TEXT_PH_0']
        self.show_chem_text_ph_1 = kwargs['SHOW_CHEM_TEXT_PH_1']
        self.cluster_font_size = kwargs['CLUSTER_FONT_SIZE']
        self.chemical_font_size = kwargs['CHEMICAL_FONT_SIZE']
        self.sniff_threshold = 0.9 #kwargs['sniff_threshold']
        self.patch_size = kwargs['PATCH_SIZE']
        self.turtle_size = kwargs['TURTLE_SIZE']

        self.W_pixels = W_pixels
        self.H_pixels = H_pixels
        self.offset = self.patch_size // 2
        self.first_gui = True

        self.screen = pygame.display.set_mode((self.W_pixels, self.H_pixels))
        self.clock = pygame.time.Clock()
        pygame.font.init()
        self.cluster_font = pygame.font.SysFont("arial", self.cluster_font_size)
        self.chemical_font = pygame.font.SysFont("arial", self.chemical_font_size)
        self.ph_pos_font = pygame.font.SysFont("arial", self.chemical_font_size * 2)

        self.show_dirs_view = kwargs["show_dirs_view"]
        if self.show_dirs_view:
            self.N_DIRS = 8
            self.wiggle_patches = kwargs["wiggle_patches"]
            self.dirs = self._get_dirs()
        self.show_ph_0_view = kwargs["show_ph_0_view"]
        self.show_ph_1_view = kwargs["show_ph_1_view"]
        self.show_food_rewards = kwargs['show_food_rewards']
        self.show_nest_rewards = kwargs['show_nest_rewards']

    def _get_dirs(self):
        central = self.wiggle_patches // 2
        sliding_window = []
        
        for i in range(self.N_DIRS):
            tmp = []
            for j in range(self.wiggle_patches):
                tmp.append((i + j) % self.N_DIRS)
            sliding_window.append(tmp)
        
        sliding_window = sorted(sliding_window, key=lambda x: x[central])
        return np.array(sliding_window)

    def render(
        self,
        patches,
        food_pos_1,
        food_pos_2,
        food_pos_3,
        nest_pos,
        patches_food_1,
        patches_food_2,
        patches_food_3,
        patches_nest,
        food_rewards_grid,
        nest_rewards_grid,
        pos_to_idx,
        learners,
        fov,
        fov_dirs,
        ph_fov,
        ph_fov_dirs,
        min_coord,
        max_coord,
        actions
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # window closed -> program quits
                pygame.quit()

        if self.first_gui:
            self.first_gui = False
            pygame.init()
            pygame.display.set_caption("SLIME")

        self.screen.fill(BLACK)
        # draw patches
        for p in patches:
            if self.show_food_rewards:
                reward_val = food_rewards_grid[pos_to_idx[p]] #* 1e3
                #pygame.draw.rect(
                #    self.screen,
                #    (reward_val, reward_val, reward_val),
                #    pygame.Rect(
                #        p[0] - self.offset,
                #        p[1] - self.offset,
                #        self.patch_size,
                #        self.patch_size
                #    )
                #)
                
                text = self.chemical_font.render(str(round(reward_val, 2)), True, GREEN)
                self.screen.blit(text, text.get_rect(center=p))
            elif self.show_nest_rewards:
                reward_val = nest_rewards_grid[pos_to_idx[p]] #* 1e3
                text = self.chemical_font.render(str(round(reward_val, 2)), True, YELLOW)
                self.screen.blit(text, text.get_rect(center=p))
            else:
                if patches[p]['chemical_0'] > patches[p]['chemical_1']:
                    chem_type = "chemical_0"
                elif patches[p]['chemical_1'] > patches[p]['chemical_0']:
                    chem_type = "chemical_1"
                else:
                    chem_type = "chemical_0"
                    #idx = random.randint(0, 1)
                    #if idx == 0:
                    #    chem_type = "chemical_0"
                    #elif idx == 1:
                    #    chem_type = "chemical_1"

                chem_type = "chemical_0"
                chem = round(patches[p][chem_type]) * self.shade_strength
                pygame.draw.rect(
                    self.screen,
                    (0, chem if chem <= 255 else 255, 0) if chem_type == "chemical_0" else (chem if chem <= 255 else 255, chem if chem <= 255 else 255, 0),
                    pygame.Rect(
                        p[0] - self.offset,
                        p[1] - self.offset,
                        self.patch_size,
                        self.patch_size
                    )
                )
                if self.show_chem_text_ph_0 and (not sys.gettrace() is None or
                                            patches[p]['chemical_0'] >= self.sniff_threshold):  # if debugging show text everywhere, even 0
                    text = self.chemical_font.render(str(round(patches[p]['chemical_0'], 1)), True, GREEN)
                    self.screen.blit(text, text.get_rect(center=p))
                
                if self.show_chem_text_ph_1 and (not sys.gettrace() is None or
                                            patches[p]['chemical_1'] >= self.sniff_threshold):  # if debugging show text everywhere, even 0
                    text = self.chemical_font.render(str(round(patches[p]['chemical_1'], 1)), True, YELLOW)
                    self.screen.blit(text, text.get_rect(center=p))

            #if p == ph_pos1:
            #    text = self.ph_pos_font.render('A', True, WHITE)
            #    self.screen.blit(text, text.get_rect(center=ph_pos1))
            #elif p == ph_pos2:
            #    text = self.ph_pos_font.render('B', True, WHITE)
            #    self.screen.blit(text, text.get_rect(center=ph_pos2))
            if p in patches_food_1:
                text = self.ph_pos_font.render('F', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))
            elif p in patches_food_2:
                text = self.ph_pos_font.render('F', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))
            elif p in patches_food_3:
                text = self.ph_pos_font.render('F', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))
            elif p in patches_nest:
                text = self.ph_pos_font.render('N', True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))

        # draw learners
        for i, learner in enumerate(learners.values()):
            if learner['flags'] != [0, 0, 0] and learner['flags'] != [0, 0, 1] and actions[i] == 3:
                learner_color = BLUE
            elif learner['flags'] != [0, 0, 0] and learner['flags'] != [0, 0, 1] and actions[i] != 3:
                learner_color = RED
            elif (learner['flags'] == [0, 0, 0] or learner['flags'] == [0, 0, 1]) and (actions[i] == 2 or actions[i] == 3):
                learner_color = RED
            elif (learner['flags'] == [0, 0, 0] or learner['flags'] == [0, 0, 1]) and actions[i] == 0:
                learner_color = ORANGE
            elif (learner['flags'] == [0, 0, 0] or learner['flags'] == [0, 0, 1]) and actions[i] == 1:
                learner_color = BLUE
            else:
                learner_color = PURPLE

            pygame.draw.circle(
                self.screen,
                learner_color,
                (learner['pos'][0], learner['pos'][1]),
                self.turtle_size // 2
            )
            if learner['flags'] == [1, 0, 1]:
                text = self.cluster_font.render(str(i), True, SKY_BLUE)
                self.screen.blit(text, text.get_rect(center=learner['pos']))
            else:
                text = self.cluster_font.render(str(i), True, PINK)
                self.screen.blit(text, text.get_rect(center=learner['pos']))

            if self.show_dirs_view:
                if learner['pos'] in fov_dirs.keys():
                    i, j = np.where(fov_dirs[learner['pos']] == learner['dir'])
                    if i.size != 0 and j.size != 0:
                        new_dir = fov_dirs[learner['pos']][-1][j].item()
                        #new_pos = possible_patches[pos][new_dir]
                    else:
                        new_dir = learner['dir']
                    
                    x = np.any(
                        (fov[learner['pos']][new_dir] < min_coord[0]) | (fov[learner['pos']][new_dir] > max_coord[0]),
                        axis=-1
                    )
                    y = np.where(x == False)[0]
                    view = fov[learner['pos']][new_dir][y]
                    dirs = self.dirs[new_dir][y]
                else:
                    if len(fov[learner["pos"]].shape) > 2:
                        view = fov[learner["pos"]][learner["dir"]]
                        dirs = self.dirs[learner["dir"]]
                    else:
                        view = fov[learner["pos"]]
                        dirs = self.dirs[4]
                
                for f, d in zip(view, dirs):
                    pygame.draw.rect(
                        self.screen,
                        YELLOW,
                        pygame.Rect(
                            f[0] - self.offset,
                            f[1] - self.offset,
                            self.patch_size,
                            self.patch_size
                        )
                    )
                    text = self.cluster_font.render(str(d), True, BLACK)
                    self.screen.blit(text, text.get_rect(center=f))

            if self.show_ph_0_view or self.show_ph_1_view:
                if learner['pos'] in ph_fov_dirs.keys():
                    # Convenzione del momento:
                    # se l'agente ha una direzione perpendicolare all'ostacolo inverte la sua direzione,
                    # altrimenti viene approssimata con la posizione più vicina.
                    i, j = np.where(ph_fov_dirs[learner['pos']] == learner['dir'])
                    if i.size != 0 and j.size != 0:
                        new_dir = ph_fov_dirs[learner['pos']][-1][j].item()
                        #new_pos = possible_patches[pos][new_dir]
                    else:
                        new_dir = learner['dir']
                    
                    x = np.any(
                        (ph_fov[learner['pos']][new_dir] < min_coord[0]) | (ph_fov[learner['pos']][new_dir] > max_coord[0]),
                        axis=-1
                    )
                    y = np.where(x == False)[0]
                    ph = ph_fov[learner['pos']][new_dir][y]
                    #breakpoint()
                else:
                    if len(ph_fov[learner["pos"]].shape) > 2:
                        ph = ph_fov[learner["pos"]][learner["dir"]]
                    else:
                        ph = ph_fov[learner["pos"]]
                
                for f in ph:
                    pygame.draw.rect(
                        self.screen,
                        WHITE,
                        pygame.Rect(
                            f[0] - self.offset,
                            f[1] - self.offset,
                            self.patch_size,
                            self.patch_size
                        )
                    )
                    if self.show_ph_0_view:
                        #if patches[learner["pos"]]['chemical_0'] >= self.sniff_threshold:
                        if patches[tuple(f)]['chemical_0'] >= self.sniff_threshold:
                            text = self.chemical_font.render(
                                str(round(patches[tuple(f)]['chemical_0'], 1)),
                                True,
                                BLACK
                            )
                            self.screen.blit(text, text.get_rect(center=f))
                    elif self.show_ph_1_view:
                        if patches[learner["pos"]]['chemical_1'] >= self.sniff_threshold:
                            text = self.chemical_font.render(
                                str(round(patches[tuple(f)]['chemical_1'], 1)),
                                True,
                                BLACK
                            )
                            self.screen.blit(text, text.get_rect(center=f))

        for p in patches:
            if len(patches[p]['turtles']) > 1:
                text = self.cluster_font.render(str(len(patches[p]['turtles'])), True, WHITE)
                self.screen.blit(text, text.get_rect(center=p))

        self.clock.tick(self.fps)
        pygame.display.flip()

        return pygame.surfarray.array3d(self.screen)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

def main():
    params = {
        "learners": 40,
        "actions": [
            "random-walk",
            "move-toward-chemical-0",
            "move-toward-chemical-1",
            "move-and-drop-chemical-1",
            #"move-away-chemical-1"
        ],
        "sniff_threshold": 0.9,
        "sniff_patches": 3, 
        "diffuse_area": 0.5,
        "diffuse_radius": 0,
        "follow_mode": "det",
        #"follow_mode": "prob",
        "wiggle_patches": 3,
        "lay_area": 1,
        "lay_amount": 3.0,
        "lay_amount_first": 1, 
        "lay_amount_min": 1.0,
        "ph_decay": 0.9,
        "evaporation": 0.95,
        "obs_type": "paper",
        #"obs_type": "variation1",
        "food_quantity": 1,
        "collective_reward_weight": 0.5,
        "food_reward": 10,
        "nest_reward": 50,
        "penalty": -5,
        "max_episode_ticks": 1000,
        #"W": 69,
        #"H": 69,
        #"PATCH_SIZE": 14,
        #"TURTLE_SIZE": 14,
        "W": 31,
        "H": 31,
        "PATCH_SIZE": 20,
        "TURTLE_SIZE": 16,
    }

    params_visualizer = {
        "FPS": 15,
        "SHADE_STRENGTH": 10,
        "SHOW_CHEM_TEXT_PH_0": False,
        "SHOW_CHEM_TEXT_PH_1": False,
        "CLUSTER_FONT_SIZE": 12,
        "CHEMICAL_FONT_SIZE": 8,
        "sniff_threshold": 0.0,
        #"PATCH_SIZE": 14,
        #"TURTLE_SIZE": 14,
        "PATCH_SIZE": 20,
        "TURTLE_SIZE": 16,
        "show_dirs_view": False,
        "wiggle_patches": 3,
        "show_ph_0_view": False,
        "show_ph_1_view": False,
        "show_food_rewards": False,
        "show_nest_rewards": True 
    }

    from tqdm import tqdm

    EPISODES = 2
    SEED = 0
    np.random.seed(SEED)
    env = Ants(SEED, **params)
    env_vis = AntsVisualizer(env.W_pixels, env.H_pixels, **params_visualizer)
    ACTION_NUM = len(params["actions"])
    AGENTS_NUM = env.num_learners


    def policy(actions):
        return random.choice(actions)

    start_time = time.time()
    for ep in tqdm(range(1, EPISODES + 1), desc="Episode"):
        env.reset()
        #for tick in tqdm(range(params['episode_ticks']), desc="Tick", leave=False):
        ticks = 1
        while not env.done: 
            #breakpoint()
            actions = np.array([-1 for _ in range(AGENTS_NUM)], dtype=np.int8)
            for agent in env.agent_iter(max_iter=AGENTS_NUM):
                obs, reward, _ , _, info = env.last(agent)
                #breakpoint()
                id = env.convert_observation(obs)
                masked_actions = env.mask_actions(obs)
                action = policy(masked_actions)
                #np.random.randint(0, ACTION_NUM)
                #action = random.choice((0, 2))
                env.step(action)
                actions[int(agent)] = action
                #env.step(2)
            env_vis.render(
                env.patches,
                env.food_pos_1,
                env.food_pos_2,
                env.food_pos_3,
                env.nest_pos,
                env.patches_food_1,
                env.patches_food_2,
                env.patches_food_3,
                env.patches_nest,
                env.food_rewards_grid,
                env.nest_rewards_grid,
                env.pos_to_idx,
                env.learners,
                env.fov,
                env.fov_dirs,
                env.ph_fov,
                env.ph_fov_dirs,
                env.min_coord,
                env.max_coord,
                actions
            )
            #breakpoint()
            ticks += 1

    print("Ticks: ", ticks)
    print("Total time = ", time.time() - start_time)
    env.close()

if __name__ == "__main__":
    main()
