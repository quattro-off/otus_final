import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw
from typing import Any, Dict, Tuple, Union

import sys
import os

sys.path.append(os.path.abspath('./'))

from env_simple_move import HumanMoveSimpleAction

class HumanMoveSimpleAction_SAC(HumanMoveSimpleAction):

    def name(self):
        return "HumanMoveSimpleSACCNN" if self.observation_render == True else "HumanMoveSimpleSAC"

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
            #if self.time_model < 0.4*self.time_model_max:
            #    return 5, True, False    
            #else:
            #    return 2, True, False
            #if self.inside == False and abs(set_angle_speed) < 0.01 and abs(set_speed_right) < 0.01 and set_speed_forward > 0: 
            #if abs(self.angle_speed) < 0.05 and abs(self.speed_bind.y) < 0.05 and self.speed_bind.x > 0: 
                return 5, True, False    
            #else:
            #    return 2, True, False
        #    self.inside = True
        #else:
        #    self.inside = False
            

        # единичный вектор на целевую точку        
        dir_to_target_point = vec_to_finish / vec_to_finish.rho
        vec_to_finish = dir_to_target_point * (vec_to_finish.rho - self.finish_radius + 1)

        # векторное расстояние пройденное за шаг моделирования
        way_vec = self.position - self.position_prev

        # расстояние на которой приблизились к целевой точке за шаг моделирования
        # мгновенная скорость сближения
        way_step = way_vec @ dir_to_target_point

       
        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        stoper_step_reward = 0.

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
                stoper_step_reward = -0.01
                #print (summ_10_step_dist)
                #return -0.01, False, False
        
        # стоять не надо
        if way_vec.rho < 0.03:
            stoper_step_reward += -0.001


        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        ## Вознаграждение за ПОВОРОТ ( set_angle_speed )

        # угол на который надо довернуть
        delta_angle = vec_to_finish.deltaphi(self.dir_view)

        # стандартный штраф
        angle_step_reward = 0.001 * (0.5 * math.pi - abs(delta_angle))
                    
        # команда на разворот в противоположную сторону - шрафуем
        if set_angle_speed != 0. and self._sign(delta_angle) != self._sign(set_angle_speed):
            angle_step_reward = -0.01
        # угловая скорость больше рассогласования по курсу
        elif self._sign(set_angle_speed) > self._sign(delta_angle):
            angle_step_reward = -0.01
            #return angle_step_reward, False, False
        # допоплниетльный штраф за то что не смотрим на целевую точку
        #if delta_angle > 6 * angle_10: # 60 градусов
        #    angle_step_reward *= 2
        #elif vec_to_finish.rho < 100 and delta_angle > angle_10:
        #    angle_step_reward *= 4




        ## Вознаграждение за Скорость ( set_speed_forward  set_speed_right )

        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        # Скорость сближения низкая
        speed_step_reward = 0.001

        if abs(way_step) < 0.1:
            self.zero_speed_step += 1
            if self.zero_speed_step > 10: # штраф за не изменение расстояния до целевой точки (чтоб не кружил вокруг)
                speed_step_reward *= -20
                #return speed_step_reward, False, False

        else:
            self.zero_speed_step = 0
            
            # удаляемся штраф увличивается пропорционально скорости, 
            # приближаемся награда увеличивается пропорционально скрости
            speed_step_reward *= way_step 

            # усиление награды при прямом сближении
            if way_step > 0.1:
                # косинус угла меду направлением движения и направление на целевую точку
                cos_way = 1
                if way_vec.rho > 0.0001:
                    cos_way = way_step / way_vec.rho
            
                # угол смещения направления на целевую точку
                angle_rate = angle_10
                if vec_to_finish.rho < 50:
                    angle_rate = angle_10 * 0.5   # 5 градусов
                elif vec_to_finish.rho < 10:
                    angle_rate = angle_10 * 0.2   # 2 градуса
                # чем ближе тем точнее надо наводится
                if cos_way > math.cos(angle_rate):
                    speed_step_reward *= 10

        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        ## Вознаграждение за рассогласование вектора скорости и вектора взгляда
        view_step_reward = 0
        # только для сближения и на большем удалении (около цели можно двигаться любым способом)
        # уменяшает награду за сближение с целевой точкой если сближение происходит боком
        #if way_step > 1 and self.speed_bind.rho > 0 and vec_to_finish.rho > 5:
        #if self.speed_bind.rho > 0.5 and vec_to_finish.rho > 5:
        #    cos_view_to_speed = vector.obj(x=1,y=0) @ self.speed_bind / self.speed_bind.rho
        #
        #    #koef_rew_view_to_speed = 1 - cos_view_to_speed
        #    koef_rew_view_to_speed = 1.3 - cos_view_to_speed
        #    d_right_reward = -0.001 if koef_rew_view_to_speed > 0.9 and abs(self.speed_bind.y) > 0.01 else 0.001
        #    view_step_reward = 0.001 * koef_rew_view_to_speed + d_right_reward

        step_reward = angle_step_reward + speed_step_reward + view_step_reward + stoper_step_reward
        #print(f'REWARDS: Angle {angle_step_reward}, Speed {speed_step_reward}, View {view_step_reward}, Stoper {stoper_step_reward}')
        return step_reward, False, False
    
