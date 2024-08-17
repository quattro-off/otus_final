import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import math
import random
import vector 
import pygame

from env_simple_move import HumanMoveSimpleAction


#Caucasus terrain map
# minX -1000, minZ -1000, maxX 1000, maxZ 1000
# X, Z, Vz, Vz, course, state(0-stand, 1-crouch, 2-pron), obstacles dist,
observation_space = Box(low=np.array([-300.,-300.,-3.6,-3.6,-math.pi, 0,-300.,-300.,-300.,-300.,-300.,-300.,-300.,-300.,-300.,-300.], dtype=np.float32), high=np.array([300.,300.,3.6,3.6,math.pi, 2,300.,300.,300.,300.,300.,300.,300.,300.,300.,300.], dtype=np.float32), shape=(16,), dtype=np.float32)

#                 b_move_to_view, course move(-pi,pi), course view(-pi,pi), speed(0,3.6), state(0-stand, 1-crouch, 2-pron), action(0-idle, 1-climb)
action_space = Box(low=np.array([0.,-1,-1, 0, 0, 0], dtype=np.float32), high=np.array([1., 1, 1, 1, 2, 1], dtype=np.float32), shape=(6,), dtype=np.float32)

class HumanMoveDamageAction(HumanMoveSimpleAction):
    """непрерывные состояния, непрерывные действия"""

    damage_circle = [] #10

    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        self.observation_space = observation_space

    def step(self, action):

        observation, step_reward, terminated, truncated, info = super().step(action)
        for circle in self.damage_circle:
            observation = np.append(observation, circle['dist'])
        
        return observation, step_reward, terminated, truncated, info

    def reset(self, seed=None):

        start_observation, info = super().reset(seed=seed)
        
        self.damage_circle.clear()
        for i in range(10):
            d_n = vector.obj(x=0.,y=0.)
            d_n.x = random.randint(-self.zero.x,self.zero.x)
            d_n.y = random.randint(-self.zero.y,self.zero.y)
            rad = random.randint(30,50)
            vec_to_dem = d_n - self.position
            self.damage_circle.append({'pos': d_n, 'r' : rad, 'dist' : vec_to_dem.rho})
            start_observation[5 + 1] = vec_to_dem.rho

        return start_observation, info
      

    def simple_move(self, set_angle_dir, set_speed):

        super().simple_move(set_angle_dir, set_speed)

        for circle in self.damage_circle:
            vec_to_dem = circle['pos'] - self.position
            circle['dist'] = vec_to_dem.rho - circle['r']

    def calc_step_reward(self, set_speed):

        step_reward, terminated, truncated =  super().calc_step_reward(set_speed)

        # проверим области ущерба
        for circle in self.damage_circle:
            if circle['dist'] < 0 :
                step_reward -= 50

        return step_reward, terminated, truncated
    

    def human_render(self):
        
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)

        self.screen.fill(black)

        for circle in self.damage_circle:
            col = white if circle['dist'] > 0 else red
            dem = self.zero + circle['pos']
            pygame.draw.circle(self.screen, col, (dem.x, dem.y), circle['r'])

        target_circle = self.zero + self.target_point
        pygame.draw.circle(self.screen, blue, (target_circle.x, target_circle.y), 10)
    
        for point in self.tick_point:
            r_point = self.zero + point
            pygame.draw.circle(self.screen, green, (r_point.x, r_point.y), 2)

        # рисуем треугольник
        triangle = self.zero + self.position
        triangle_width = 10
        triangle_height = 10
        corner2 = vector.obj(x = -triangle_height, y = triangle_width/2)
        corner3 = vector.obj(x = -triangle_height, y = -triangle_width/2)
        corner2 = corner2.rotateZ(self.course)
        corner3 = corner3.rotateZ(self.course)
        corner2 += triangle
        corner3 += triangle

        triangle_points = [(triangle.x, triangle.y), (corner2.x, corner2.y), (corner3.x, corner3.y)]
        pygame.draw.polygon(self.screen, red, triangle_points)

        pygame.display.update()

        # устанавливаем частоту обновления экрана
        pygame.time.Clock().tick(60)

