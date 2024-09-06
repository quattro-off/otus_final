
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw

from env_simple_move import HumanMoveSimpleAction


# X (-300,300), Z (-300,300), Vx (-3.6,3.6), Vz (-3.6,3.6) Phi(-pi, pi) T(1,2) R(0.5-10)
# TO (0,1)
# X, Z, Vz, Vz, course ----- 5
# X, Z, R, T ------ 30  T - 1 damage 2 - stoper = 120 
# x, Z, X, Z ------ 10 speed to lower           = 40
# x, Z, X, Z ------ 10 stoper line              = 40
#                                   total       = 205
observation_space = Box(low=np.array([0, 0, 0, 0, 0,   
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,   
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,    
                                      0, 0, 0, 0,   
                                        ], dtype=np.float32), 
                        high=np.array([1, 1, 1, 1, 1,
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,  
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,   
                                       1, 1, 1, 1,  
                                        ], dtype=np.float32), shape=(45,), dtype=np.float32)

#                 course move(-pi,pi), speed(0,3.6)
action_space = Box(low=np.array([-1, 0], dtype=np.float32), high=np.array([1, 1], dtype=np.float32), shape=(2,), dtype=np.float32)

action_space_d = Discrete(5, seed=42)

observation_space_rend = Box(0, 255, (600, 600, 3), dtype=np.uint8)

class HumanMoveDamageAction(HumanMoveSimpleAction):
    """непрерывные состояния, непрерывные действия"""

    #observation
    radius_k = 10

    damage_circle_start = 5
    damage_circle_step = 4

    damage_circle_count = 10
    damage_circle = [] #10

    damage_obs = np.zeros(40, dtype=np.float32)

    def _min_then_max(self, v1:float, v2:float):
        if v1 < v2:
            return v1, v2
        else:
            return v2, v1

    def name(self):
        return "HumanMoveDamageActionCNN" if self.observation_render == True else "HumanMoveDamageAction"

    def __init__(self, continuous: bool = True, seed: int=42, render_mode: str=None, observation_render: bool = False, damage_zone: int = 10):
        super().__init__(continuous, seed, render_mode, observation_render)
        if observation_render == True:
            self.observation_space = observation_space_rend
            self.observation_space_local = observation_space
        else:
            self.observation_space = observation_space
            self.observation_space_local = observation_space
        
        self.damage_circle_count = damage_zone

    def step(self, action, is_teacher: bool = False):

        if self.observation_render == True:
            observation, step_reward, terminated, truncated, info = super().step(action, is_teacher)

            return self._get_render(True), step_reward, terminated, truncated, info
        else:
            if is_teacher == False:
                observation, step_reward, terminated, truncated, info = super().step(action, is_teacher)
                observation = np.append(observation, self.damage_obs)
        
                return observation, step_reward, terminated, truncated, info
            else:
                observation, step_reward, terminated, truncated, info, teach_observation = super().step(action, is_teacher)
                observation = np.append(observation, self.damage_obs)

                return observation, step_reward, terminated, truncated, info, teach_observation



    def reset(self, seed=None):

        start_observation, info = super().reset(seed=seed)

        _min_x, _max_x = self._min_then_max(self.position.x, 0)
        _min_y, _max_y = self._min_then_max(self.position.y, 0)
        
        self.damage_circle.clear()
        for i in range(10):
            num_x = self.damage_circle_start + i * self.damage_circle_step
            num_y = num_x + 1
            num_r = num_y + 1
            num_t = num_r + 1
            if i < self.damage_circle_count:
                d_n = vector.obj(x=random.randint(int(_min_x), int(_max_x)), y =random.randint(int(_min_y), int(_max_y)))
                rad = random.randint(0, self.radius_k)
                start_observation[num_x] = self._output_norm(self.locate_k, d_n.x)
                start_observation[num_y] = self._output_norm(self.locate_k, d_n.y)
                start_observation[num_r] = self._output_norm(self.radius_k, num_t)
                start_observation[num_t] = 1
                vec_to_dem = d_n - self.position
                self.damage_circle.append({'pos': d_n, 'r' : rad, 'dist' : vec_to_dem.rho})
            else:
                self.damage_circle.append({'pos': vector.obj(x=0.,y=0.), 'r' : 0, 'dist' : 1300})
                start_observation[num_x] = 0
                start_observation[num_y] = 0
                start_observation[num_r] = 0
                start_observation[num_t] = 1

            
            self.damage_obs[num_x-self.damage_circle_start] = float(start_observation[num_x])
            self.damage_obs[num_y-self.damage_circle_start] = float(start_observation[num_y])
            self.damage_obs[num_r-self.damage_circle_start] = float(start_observation[num_r])
            self.damage_obs[num_t-self.damage_circle_start] = float(start_observation[num_t])


        if self.observation_render == True:
            return self._get_render(True), info
        else:
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
    
    def render(self):
        #if self.is_render == False:
        #    if render_mode == None or render_mode != 'rgb_array':
        #        return None
        return self._get_render(False)

    def _get_render(self, to_observation: bool):
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        
        im = Image.new('RGB', (self.zero.x * 2, self.zero.y * 2), black)
        draw = ImageDraw.Draw(im)
        
        draw.rectangle((1, 1, self.zero.x * 2 - 1, self.zero.y * 2 - 1), outline=red, width=4)

        target_circle = self.zero + self.target_point
        draw.ellipse((target_circle.x-5, target_circle.y-5,target_circle.x+5, target_circle.y+5), fill=blue, outline=blue)

        for circle in self.damage_circle:
            col = red if to_observation == True else white if circle['dist'] > 0 else red
            dem = self.zero + circle['pos']
            draw.ellipse((dem.x-circle['r'], dem.y-circle['r'],dem.x+circle['r'], dem.y+circle['r']), fill=col, outline=col)
    
        # когда среда это картинка то траеторию не рисуем
        if to_observation == False:
            for point in self.tick_point:
                r_point = self.zero + point
                draw.point((r_point.x, r_point.y), fill=green)

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

        draw.polygon(
            xy=(
                (triangle.x, triangle.y),
                (corner2.x, corner2.y),
                (corner3.x, corner3.y)
            ), fill=red, outline=red
        )

        im_np = np.asarray(im)
        return im_np

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
            rad = float(circle['r'])
            pygame.draw.circle(self.screen, col, (dem.x, dem.y), rad)

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

