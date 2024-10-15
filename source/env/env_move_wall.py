
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw

from env_move_sector import HumanMoveSectorAction


class HumanMoveAroundWallAction(HumanMoveSectorAction):

    
    #wall_count = 20
    #wall_length = vector.obj(x=10,y=20)

    def name(self):
        return 'Wall_' + super().name() 
    
    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 wall_border:int = 5,
                 options = None
                ):
        
        self.wall_border_count = wall_border
        
        self.trees_count = 0

        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         object_ignore = object_ignore,
                         options=options,
                         )

    def reset(self, seed=None):

        self.trees_count = 0

        start_observation, info = super().reset(seed=seed)

        self.target_point.x = 250
        self.target_point.y = 0

        self.position.x = -250
        self.position.y = 0

        
        vec_dist = self.target_point - self.position
        self.time_model_optimum = vec_dist.rho / self.max_speed
        self.time_model_max = 3 * self.time_model_optimum
        
        vec_dist.rho -= self.finish_radius + 1
        

        start_observation[0] = self._output_norm(self.locate_k, vec_dist.x)   # dXt, dYt, Dt, Azt, 
        start_observation[1] = self._output_norm(self.locate_k, vec_dist.y)
        start_observation[2] = self._output_norm(self.locate_k, vec_dist.rho)
        start_observation[3] = self._output_norm(self.angle_k, self._get_course_bind(vec_dist.phi))

        
        #_min_pos_x = 0.5 * (self.position.x + self.target_point.x) - 100
        #_max_pos_x = 0.5 * (self.position.x + self.target_point.x) + 100
        #_min_pos_y = 0.5 * (self.position.y + self.target_point.y) - 100
        #_max_pos_y = 0.5 * (self.position.y + self.target_point.y) + 100

        id = self.objects[-1]['id'] + 1
        #for i in range(self.wall_count):
        #    wall_1 = vector.obj(x = random.randint(int(_min_pos_x), int(_max_pos_x)),
        #                         y = random.randint(int(_min_pos_y), int(_max_pos_y))
        #                         )
        #    point_phi = math.radians(random.randint(-180,180))
        #    wall_length = random.randint(int(self.wall_length.x), int(self.wall_length.y))
        #    wall_2 = wall_1 + vector.obj(phi=point_phi, rho=wall_length)
        #    wall_data = {'id': i+id, 'type': 'wall', 'vis': False, 'col': (1,1,0), 'c1': wall_1, 'c2': wall_2}
        #    self.objects.append(wall_data)

        if self.object_ignore == False:

            delta_y = 50
            delta_x = 50
            for i in range(4,5,2):
                rnd_y = random.randint(-200, 200)
                wall_1 = vector.obj(x= -200 + i*delta_x, y =-300)
                wall_2 = vector.obj(x= -200 + i*delta_x, y =rnd_y)
                wall_data = {'id': id + i, 'type': 'wall', 'vis': False, 'col': (1,1,0), 'c1': wall_1, 'c2': wall_2}
                self.objects.append(wall_data)
                wall_1 = vector.obj(x= -200 + i*delta_x, y =rnd_y + delta_y)
                wall_2 = vector.obj(x= -200 + i*delta_x, y =300)
                wall_data = {'id': id + i + 1, 'type': 'wall', 'vis': False, 'col': (1,1,0), 'c1': wall_1, 'c2': wall_2}
                self.objects.append(wall_data)

            id = self.objects[-1]['id'] + 1
            wall_1 = vector.obj(x= -295, y =-299)
            wall_2 = vector.obj(x= -295, y =299)
            wall_data = {'id': id, 'type': 'wall', 'vis': False, 'col': (1,1,0), 'c1': wall_1, 'c2': wall_2}
            self.objects.append(wall_data)
            wall_1 = vector.obj(x= -299, y =-295)
            wall_2 = vector.obj(x= 299, y =-295)
            wall_data = {'id': id+1, 'type': 'wall', 'vis': False, 'col': (1,1,0), 'c1': wall_1, 'c2': wall_2}
            self.objects.append(wall_data)
            wall_1 = vector.obj(x= 295, y =-299)
            wall_2 = vector.obj(x= 295, y =299)
            wall_data = {'id': id+2, 'type': 'wall', 'vis': False, 'col': (1,1,0), 'c1': wall_1, 'c2': wall_2}
            self.objects.append(wall_data)
            wall_1 = vector.obj(x= -299, y =295)
            wall_2 = vector.obj(x= 299, y =295)
            wall_data = {'id': id+3, 'type': 'wall', 'vis': False, 'col': (1,1,0), 'c1': wall_1, 'c2': wall_2}
            self.objects.append(wall_data)

            
        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info
        
    

    def calc_step_reward_box(self, set_angle_speed: float, set_speed_forward: float, set_speed_right: float):

        if self.time_model_max < self.time_model:  # время вышло
            return -5, False, True

        #ушел за границу зоны
        if self.position.x < -self.zero.x or self.position.y < -self.zero.y or self.position.x > self.zero.x or self.position.y > self.zero.y:
            return -2, False, True

        angle_10 = math.pi / 18.  # 10 градусов

        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        vec_to_finish = self.target_point - self.position
        # пришли
        if vec_to_finish.rho < self.finish_radius:
            return 5, True, False    
            

        # единичный вектор на целевую точку        
        dir_to_target_point = vec_to_finish / vec_to_finish.rho
        vec_to_finish = dir_to_target_point * (vec_to_finish.rho - self.finish_radius + 1)

        # векторное расстояние пройденное за шаг моделирования
        way_vec = self.position - self.position_prev

        # расстояние на которой приблизились к целевой точке за шаг моделирования
        # мгновенная скорость сближения
        way_step = way_vec @ dir_to_target_point

       
        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        self.stoper_step_reward = 0.

        # штраф за  нахождение на одном месте больше 10 тиков
        if len(self.tick_point) > 11:
            summ_10_step_dist = 0.
            end = len(self.tick_point) - 1
            srt = end - 11
            count_action = 0 
            for n in range( srt+1, end):
                dist_step = self.tick_point[n] - self.tick_point[n-1]
                dist_step_to_finish = dist_step @ dir_to_target_point
                summ_10_step_dist += dist_step_to_finish
                if self._equivalent_vec3(self.tick_action[n], self.tick_action[n-1], 0.01):
                    count_action += 1
            
            if abs(summ_10_step_dist) < self.speed_k and count_action > 7:
                self.stoper_step_reward = -0.01
                #print (summ_10_step_dist)
                #return -0.01, False, False
        
        # стоять не надо
        if way_vec.rho < 0.03:
            self.stoper_step_reward += -0.001


        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        ## Вознаграждение за ПОВОРОТ ( set_angle_speed )

        # угол на который надо довернуть
        delta_angle = vec_to_finish.deltaphi(self.dir_view)

        # стандартный штраф
        self.angle_step_reward = 0.001 * (0.5 * math.pi - abs(delta_angle))
                    
        # команда на разворот в противоположную сторону - шрафуем
        #if set_angle_speed != 0. and self._sign(delta_angle) != self._sign(set_angle_speed):
        #    self.angle_step_reward = -0.01
        # угловая скорость больше рассогласования по курсу
        #elif self._sign(set_angle_speed) > self._sign(delta_angle):
        #    self.angle_step_reward = -0.01
            #return angle_step_reward, False, False
        # допоплниетльный штраф за то что не смотрим на целевую точку
        #if delta_angle > 6 * angle_10: # 60 градусов
        #    angle_step_reward *= 2
        #elif vec_to_finish.rho < 100 and delta_angle > angle_10:
        #    angle_step_reward *= 4




        ## Вознаграждение за Скорость ( set_speed_forward  set_speed_right )

        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        # Скорость сближения низкая
        self.speed_step_reward = 0.001
         
        # удаляемся штраф увличивается пропорционально скорости, 
        # приближаемся награда увеличивается пропорционально скрости
        self.speed_step_reward *= way_step 


        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        ## Вознаграждение за рассогласование вектора скорости и вектора взгляда
        self.view_step_reward = 0
        # только для сближения и на большем удалении (около цели можно двигаться любым способом)
        # уменяшает награду за сближение с целевой точкой если сближение происходит боком
        #if way_step > 1 and self.speed_bind.rho > 0 and vec_to_finish.rho > 5:
        if self.speed_bind.rho > 0.5 and vec_to_finish.rho > 5:
            cos_view_to_speed = vector.obj(x=1,y=0) @ self.speed_bind / self.speed_bind.rho

            #koef_rew_view_to_speed = 1 - cos_view_to_speed
            koef_rew_view_to_speed = 1.3 - cos_view_to_speed
            d_right_reward = -0.0001 if koef_rew_view_to_speed > 0.9 and abs(self.speed_bind.y) > 0.01 else 0.0001
            self.view_step_reward = 0.0001 * koef_rew_view_to_speed + d_right_reward

        step_reward = self.angle_step_reward + self.speed_step_reward + self.view_step_reward + self.stoper_step_reward
        #print(f'REWARDS: Angle {angle_step_reward}, Speed {speed_step_reward}, View {view_step_reward}, Stoper {stoper_step_reward}')
        return step_reward, False, False