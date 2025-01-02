
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector
import re 
import pygame
from PIL import Image, ImageDraw
from typing import Dict
import sys
import os

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../common'))

from env_move_simple_v3 import MoveSimpleActionV3
from common.ray_view import RaySensor
import common.functions as f


class HumanMoveRayActionV3(MoveSimpleActionV3):

    #observation
    damage_radius_k = 10
    tree_radius_k = 0.5
    dist_k = 100

    #obstacles
    tree_radius = 6

    def name(self):
        add_rnd = '_RanTP' if self.target_point_rand == True else ''
        add_cnn = '_CNN' if self.observation_render == True else ''
        add_ignore = '_IgnoreObst' if self.object_ignore == True else ''
        return 'MoveRay3' + add_rnd + add_cnn + add_ignore

    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 render_time_step: float=1.,
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 object_locate:str = 'random',
                 tree_count: int = 20,
                 options = None
                ):
        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         render_time_step = render_time_step,
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         options=options,
                         )
        
        self.sensor = RaySensor(self.dist_k)
        sensor_observation = self.sensor.get_observation_box()

        self.object_locate = object_locate
        self.trees_count = tree_count
        
        num = re.findall('(\d+)', self.object_locate)
        if len(num) == 0:
            self.border_count = 1
        else:
            self.border_count = int(f.f_clamp(int(num[-1]), 1, 20))
        
        if observation_render == True:
            self.observation_space_local.low = np.concatenate((self.observation_space_local.low, sensor_observation.low))
            self.observation_space_local.high = np.concatenate((self.observation_space_local.high, sensor_observation.high))
            self.observation_space_local.shape += sensor_observation.shape
        else:
            self.observation_space_local = Box(low=np.concatenate((self.observation_space_local.low, sensor_observation.low)),
                                               high=np.concatenate((self.observation_space_local.high, sensor_observation.high)),
                                               #shape=self.observation_space_local.shape + sensor_observation.shape,
                                               dtype=np.float32
                                               )
            self.observation_space_local.low[0] = 0
            self.observation_space_local.low[1] = 0
            self.observation_space_local.low[2] = 0
            self.observation_space = self.observation_space_local

        
        self.reward_object_use = 0 # 0-ignore, 1-append, 2-rewrite
        self.reward_object_stop = 0 # штраф за сторкновение с препятствием
        self.reward_object_move = 0 # вознаграждение за попытку обойти

        self.object_ignore = object_ignore

        self.objects = []

        self.left_or_right = True
       
    def get_rewards(self)->Dict[str, float]:
        step_rews = super().get_rewards()
        step_rews['object_stop'] = self.reward_object_stop
        step_rews['object_move'] = self.reward_object_move
        return step_rews


    def reset(self, seed=None):

        observation, info = super().reset(seed=seed)

        self.reward_object_use = 0
        self.reward_object_stop = 0
        self.reward_object_move = 0 

        self.sensor.clear()

        self.objects.clear()

        position = self.dynamic.get_position()

        start_data = {'id': 0, 'type': 'start', 'pos':position, 'radius': 6, 'vis': False, 'col': (0,0,0)}
        self.objects.append(start_data)

        # target point
        final_data = {'id': 1, 'type': 'final', 'pos':self.target_point, 'radius': 6, 'vis': False, 'col': (0,0,255)}
        self.objects.append(final_data)

        if self.object_ignore == False:

            if 'wall' in self.object_locate or 'build' in self.object_locate:

                start_line = vector.obj(x = -250, y=0)
                end_line = vector.obj(x = 250, y=0)
                to_target = end_line - start_line

                k_rnd = 0.01 * random.randint(10,30)


                right_target = vector.obj(x=-to_target.y,y=to_target.x)
                right_target /= right_target.rho
                start_create = start_line + k_rnd*to_target
                to_target /= to_target.rho

                for b in range(self.border_count):
                    k_rnd = random.randint(10,44)
                    k_rnd = float(k_rnd)*0.1
                    sign = -1 if (b+1)%2 else 1
                    start_create += to_target * self.tree_radius * 6 + sign * right_target * self.tree_radius * k_rnd

                    for i in range(self.trees_count):
                        id = i + 2
                        
                        k_rnd = random.randint(20,40)
                        k_rnd = float(k_rnd)*0.1

                        sign = -1 if (i+1)%2 else 1

                        d_n = start_create + right_target * sign * (i * self.tree_radius * k_rnd )
                        if d_n.x > -self.zero.x and d_n.x < self.zero.x and d_n.y > -self.zero.y and d_n.y < self.zero.y:
                            tree_data = {'id': id, 'type': 'tree', 'pos':d_n, 'radius': self.tree_radius, 'vis': False, 'col': (0,255,0)}
                            self.objects.append(tree_data)


                if self.target_point_rand == False:
                    position.x = 0
                    position.y = 0

                    x_side = -1 if random.randint(0, 1) == 0 else 1
                    self.target_point.x = 250 * x_side
                    self.target_point.y = random.randint(-290,290)

            else:
                # trees
                for i in range(self.trees_count):
                    id = i + 2

                    d_n = vector.obj(x=0,y=0)
                    is_valide = False
                    while is_valide == False:
                        d_n = vector.obj(x = random.randint(-int(self.zero.x), int(self.zero.x)),
                                        y = random.randint(-int(self.zero.y), int(self.zero.y))
                                        )
                        d_pos = position - d_n
                        d_tp = self.target_point - d_n
                        if d_pos.rho > self.tree_radius + 1 and d_tp.rho > self.tree_radius + 1:
                            is_valide = True

                    tree_data = {'id': id, 'type': 'tree', 'pos':d_n, 'radius': self.tree_radius, 'vis': False, 'col': (0,255,0)}
                    self.objects.append(tree_data)

            self.dynamic.set_position(position)


        self.left_or_right = bool(random.randint(0, 1))

        observation[4:] = self.sensor.get_observation()

        if self.observation_render == True:
            return self.get_cnn_observation(), info
        else:
            return observation, info
        
    def step(self, action):

        if self.observation_render == True:
            observation, step_reward, terminated, truncated, info = super().step(action)

            return self.get_cnn_observation(), step_reward, terminated, truncated, info
        else:
            
            self._prev_reward_box()

            observation, step_reward, terminated, truncated, info = super().step(action)

            # убираем препятсвие
            self.dynamic.set_obstacle(vector.obj(x=100,y=100))

            if self.object_ignore == False:

                speed_bind = self.dynamic.get_speed_bind()
                position = self.dynamic.get_position()
                direction = self.dynamic.get_direction()

                self.sensor.set_move_ray_direction(speed_bind.phi)
                self.sensor.step(self.objects, position, direction)
                vec_dist_bind = self.sensor.get_move_ray()

                # препятсвие для для расчета столкновения
                self.dynamic.set_obstacle(vec_dist_bind)

            observation[4:] = self.sensor.get_observation()

            i = 0
            for obs in observation:

                if math.isnan(obs):
                    print(f'abs n:{i} is NAN')
                if math.isinf(obs):
                    print(f'abs n:{i} is INF')
                if i > 3:
                    if obs < -0.01 or obs > 1.01 :
                        obst_n = (i - 4)%3
                        obs_name = 'Density' if obst_n == 0 else ('Dist' if obst_n == 1 else 'Type')
                        #print(f'abs n:{i}, by {obs_name}, out of NORM : {obs}')
            
                obs = f.f_clamp(obs,0.,1.)
                i += 1

            return observation, step_reward, terminated, truncated, info
 
    def _prev_reward_box(self):

        self.reward_object_use = 0
        self.reward_object_stop = 0
        self.reward_object_move = 0 

        if self.object_ignore == False:

            set_move_bind = vector.obj(x=self.speed_bind_set.x, y=self.speed_bind_set.y)
            move_bind = self.dynamic.get_speed_bind()
                        
            to_target = self.target_point - self.dynamic.get_position()
            stoper_reward = 0

            for _, ray in self.sensor.rays.items():

                ray_dist, ray_dist_k = ray.get_distances()
                reward = 0.

                
                if move_bind.rho > 0.5 :
                    cos_move_obst = (ray.direct @ move_bind) / move_bind.rho
                    if cos_move_obst > 0.97:#15гр
                        speed = self.dynamic.get_speed()
                        cos_to_target = (speed @ to_target) / (speed.rho * to_target.rho)
                        #self.reward_object_move = 0.0002 * math.pow(3,cos_to_target) - 0.0001*(1./(math.sqrt(ray_dist_k)+0.1) - 1)
                        #self.reward_object_move = 0.00005 * (2*cos_to_target + 5) - 0.0001*(1./(math.sqrt(ray_dist_k)+0.25) - 1)
                        self.reward_object_move = 0.00005 * (1.5*cos_to_target + 2) - 0.0001*(1./(math.sqrt(ray_dist_k)+0.5) - 1)
                        if self.reward_object_move < 0:
                            self.reward_object_move = 0

                if ray_dist_k > 0.99:
                    ray.set_reward(reward)
                else:

                    reward = - 0.0005*(1./(math.pow(ray_dist_k, 0.25 )+0.1) - 1)
                    ray.set_reward(reward)

                    if set_move_bind.rho > 0:
                        cos_move_obst = (ray.direct @ set_move_bind) / set_move_bind.rho
                        if cos_move_obst > 0.97:#15гр
                            stoper_reward = reward


                    if reward < self.reward_object_stop:
                        self.reward_object_stop = reward

            if stoper_reward < -0.0006:
                self.reward_object_move = 0
            else:
                self.reward_object_stop = 0
              

    def calc_step_reward_box(self, set_angle_speed, set_speed, set_speed_right):

        step_reward, terminated, truncated =  super().calc_step_reward_box(set_angle_speed, set_speed, set_speed_right)

        if self.reward_object_move > 0:
            step_reward -= self.speed_step_reward
            self.speed_step_reward = 0
        return step_reward + self.reward_object_move + self.reward_object_stop, terminated, truncated

    
    def render(self):
        figures = self._get_figures(False)
        return self._get_render(figures,False)
    
    def human_render(self, figures:list=None):
        figures = self._get_figures(False)
        return super().human_render(figures)
        


    def _get_figures(self,to_observation:bool):

        black = (0, 0, 0)
        white = (255, 255, 255)

        # позиция агента
        pos = self.zero + self.dynamic.get_position()
        
        figures1 = super()._get_figures(to_observation)

        figures = []
        if self.object_ignore == False:

            # область видимости объектов  вокруг
            figures.append({'figure':'ellipse', 'senter': (pos.x,pos.y), 'radius':self.dist_k, 'in_color':black, 'out_color':white, 'width':1})

            dir_view = self.dynamic.get_direction()
            figures = figures + self.sensor.get_figures(pos, dir_view)

            for object in self.objects:
                if object['type'] == 'tree':
                    col = object['col']
                    dem = self.zero + object['pos']
                    rad = object['radius']
                    figures.append({'figure':'ellipse', 'senter': (dem.x,dem.y), 'radius':rad, 'in_color': col, 'out_color':col, 'width':0})
                elif object['type'] == 'wall':
                    col = object['col']
                    c1 = self.zero + object['c1']
                    c2 = self.zero + object['c2']
                    figures.append({'figure':'line', 'start': (c1.x,c1.y), 'end':(c2.x,c2.y), 'in_color':col, 'width':4})

                

        return figures + figures1

