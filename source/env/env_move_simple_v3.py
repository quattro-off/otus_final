import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from dynamic.dynamic_simple import DynamivSimple
import common.functions as f
from PIL import Image, ImageDraw
from typing import Any, Dict, Tuple, Union

import sys
import os

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath('../common'))
sys.path.append(os.path.abspath('../dynamic'))

# TO (0,1)
# Vx_bind, Vy_bind, dAzt, Dist
observation_space = Box(low=np.array([-1,-1,-1,0], dtype=np.float32), high=np.array([1,1,1,1], dtype=np.float32), shape=(4,), dtype=np.float32)

#                 speed forward(-3.6,3.6) m/s speed right(-3.6,3.6) m/s, course speed(-pi/3,pi/3) rad/s
action_space = Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), shape=(3,), dtype=np.float32)

action_space_d = Discrete(7, seed=42)

observation_space_rend = Box(0, 255, (600, 600, 1), dtype=np.uint8)

class MoveSimpleActionV3(gym.Env):



    #observation
    locate_k = 900
    speed_k = 3.6
    speed_angle_k = math.pi/3
    angle_k = math.pi

    #constants
    #time_step = 0.02
    #time_tick = 0.1
    #speed_acc = 2. #m2/s
    #angle_speed_acc = 2. #rad/s

    #min_speed = 0.44 #m/s
    #max_speed = 3.6 #m/s
    #desc_step_speed = 5#0.5 #m/s
    #desc_step_angle_speed = 0.1 #rad/s
    finish_radius = 50 #m


    def _is_norm(self, value: float)->bool:
        return False if value < 0. or value > 1. else True


    def _output_norm(self, koef: float, value: float, unsigned: bool = False):
        result = value / koef if unsigned == True else value / (2*koef) + 0.5
        #if koef != 300 and self._is_norm(result) == False:
        #    print(f'Warning: Bad normolize value: {value} by koef: {koef} - unsigned: {unsigned}')
        result = f.f_clamp(result,0.,1.)
        return result

    def _input_unnorm_vector(self, koef: float, x: float, y: float):
        return vector.obj(x=(x*2 - 1)*koef,y=(y*2 - 1)*koef)
    
    def _input_unnorm_value(self, koef: float, value: float, unsigned: bool = False):
        if unsigned == True:
            return value * koef
        else:
            return (value*2 - 1)*koef


    def name(self):
        return "HumanMoveSimpleCNN" if self.observation_render == True else "HumanMoveSimple"

    def __init__(self, 
                 dynamic = DynamivSimple(),
                 continuous: bool = True, 
                 seed: int=42, 
                 observation_render: bool = False,
                 target_point_rand:bool = False,
                 render_mode: str=None, 
                 render_time_step: float=1., 
                 options = None  #options={'finish_dist':100,'start_dist':110,'delta_phi':0}
                 ):

        self.dynamic = dynamic

        self.continuous = continuous                    # непрерывное дискретное действие
        self.observation_render = observation_render    # среда ввиде матрицы данных или картинка(оттенки серого)
        self.target_point_rand = target_point_rand      # целевая точка движения в центре или случайная
        self.options = options

        self.tick = 0

        # reward variables
        self.reward_range = (-float(600), float(600))
        self.reward = 0.
        self.angle_step_reward = 0.
        self.speed_step_reward = 0.
        self.view_step_reward = 0.
        self.stoper_step_reward = 0.
        self.zero_speed_step = 0 #счетчик малой скорости сближения с целевой точкой
        self.inside = False

        # history
        self.tick_point = []
        self.tick_action = []

        #variables for move
        self.time_model_max = 0.
        self.position_prev = vector.obj(x=0.,y=0.)
        self.target_point = vector.obj(x=100.,y=100.)
        self.speed_bind_set = vector.obj(x=0.,y=0.)

        # visualiser
        self.render_mode = render_mode
        self.time_step_human_draw = render_time_step
        self.time_next_human_draw = 0.
        self.if_render = False
        self.is_pygame = False
        self.screen = {}
        self.zero = vector.obj(x=300, y=300)


        if observation_render == True:
            self.observation_space = observation_space_rend
            self.observation_space_local = observation_space # среда для вычислений всегда параметрическая
        else:
            self.observation_space = observation_space
            self.observation_space_local = observation_space


        if continuous == True:
            self.action_space = action_space
        else:
            self.action_space = action_space_d
            self.action_space.seed(seed=seed)

        super().reset(seed=seed)
        random.seed(seed)
        self.observation_space.seed(seed)
        self.observation_space_local.seed(seed)


        if render_mode == 'human':
            self.is_render = False
            self.is_pygame = True
            pygame.init()
            self.screen = pygame.display.set_mode((self.zero.x * 2, self.zero.y * 2))
            pygame.display.set_caption("Moving")
        elif render_mode == 'rgb_array':
            self.is_pygame = False
            self.is_render = True

    def set_options(self, options = None):
        self.options = options

    def reset(self, seed: int = None):

        if seed != None:
            random.seed(seed)
            self.observation_space.seed(seed)
            self.observation_space_local.seed(seed)


        start_observation = self.observation_space_local.sample()

        self.time_model = 0.
        self.inside = False

        # целевая точка движения
        if self.target_point_rand == True:
            self.target_point.x = random.randint(-int(self.zero.x), int(self.zero.x))
            self.target_point.y = random.randint(-int(self.zero.y), int(self.zero.y))
        else:
            self.target_point.x = 0.
            self.target_point.y = 0.

        start_position = vector.obj(x=0.,y=0.)
        start_direction = vector.obj(x=0.,y=0.)

        if self.options == None:

            self.finish_radius = 5

            # начальная позиция
            start_position.x = random.randint(int(-self.zero.x), int(self.zero.x))
            start_position.y = random.randint(int(-self.zero.y), int(self.zero.y))
            start_phi = math.radians(random.randint(-180,180))
            start_direction = vector.obj(phi=start_phi,rho=1)
        else:

            self.finish_radius = self.options['finish_dist']

            # начальная позиция
            phi = random.randint(int(-180), int(180))
            start_position.rho = self.options['start_dist']
            start_position.phi = math.radians(phi)

            #начaльный курс
            dir = self.target_point - start_position
            start_dalta_phi = math.radians(random.randint(-self.options['delta_phi'], self.options['delta_phi']))
            start_direction = vector.obj(phi=dir.phi + start_dalta_phi,rho=1)

        self.dynamic.reset(start_position, start_direction)

        self.position_prev = start_position
        self.speed_bind_set = vector.obj(x=0.,y=0.)

        vec_dist = self.target_point - start_position
        self.time_model_optimum = self.dynamic.get_min_time_moving(self.target_point)
        self.time_model_max = 3 * self.time_model_optimum

        vec_dist.rho -= self.finish_radius + 1
        d_azimut = vec_dist.deltaphi(start_direction)

        start_observation[0] = 0
        start_observation[1] = 0
        start_observation[2] = self._output_norm(self.angle_k, d_azimut, True)
        start_observation[3] = self._output_norm(self.locate_k, vec_dist.rho, True)

        self.tick = 0
        self.reward = 0.
        self.angle_step_reward = 0.
        self.speed_step_reward = 0.
        self.view_step_reward = 0.
        self.stoper_step_reward = 0.
        self.tick_point.clear()
        self.tick_action.clear()

        self.time_next_human_draw = 0.
        
        info = self._get_info()

        if self.observation_render == True:
            rend = self.get_cnn_observation()
            return rend, info
        else:
            return start_observation, info

    def get_rewards(self)->Dict[str, float]:
        return {
            'angle_reward': self.angle_step_reward,
            'speed_reward': self.speed_step_reward,
            'view_reward':  self.view_step_reward,
            'stoped_reward': self.stoper_step_reward}

    def close(self):
        if self.is_pygame == True:
            pygame.quit()
        super().close()


    def _get_info(self):
        return {"tick progress: ": self.tick}
    
    def _get_length(self):
        return int(self.time_model_max)

    def step(self, action):

        set_angle_speed = 0      # action - 1(-) - 2(+)
        set_speed_forward = 0        # action - 3(-) - 4(+)
        set_speed_right = 0          # action - 5(-) - 6(+)


        if self.continuous == True:
            set_speed_forward = action[0] * self.speed_k
            set_speed_right = action[1] * self.speed_k
            set_angle_speed  = action[2] * self.speed_angle_k
        else:

            if action == 0:
                set_angle_speed = 0.
                set_speed_forward = 0.
                set_speed_right = 0.
            elif action == 1:
                #set_angle_speed = set_angle_speed - self.desc_step_angle_speed if set_angle_speed < 0. else -self.desc_step_angle_speed
                set_angle_speed = -self.speed_angle_k
                set_speed_forward = 0.
                set_speed_right = 0.
            elif action == 2:
                #set_angle_speed = set_angle_speed + self.desc_step_angle_speed if set_angle_speed > 0. else self.desc_step_angle_speed
                set_angle_speed = self.speed_angle_k
                set_speed_forward = 0.
                set_speed_right = 0.
            elif action == 3:
                set_angle_speed = 0.
                #set_speed_forward = set_speed_forward - self.desc_step_speed if set_speed_forward < 0. else -self.desc_step_speed
                set_speed_forward = -self.speed_k
                set_speed_right = 0.
            elif action == 4:
                set_angle_speed = 0.
                #set_speed_forward = set_speed_forward + self.desc_step_speed if set_speed_forward > 0. else self.desc_step_speed
                set_speed_forward = self.speed_k
                set_speed_right = 0.                
            elif action == 5:
                set_angle_speed = 0.
                set_speed_forward = 0.
                #set_speed_right = set_speed_right - self.desc_step_speed if set_speed_right < 0. else -self.desc_step_speed
                set_speed_right = -self.speed_k
            elif action == 6:
                set_angle_speed = 0.
                set_speed_forward = 0.
                #set_speed_right = set_speed_right + self.desc_step_speed if set_speed_right > 0. else self.desc_step_speed
                set_speed_right = self.speed_k

            set_angle_speed = self._clamp(set_angle_speed, -self.speed_angle_k, self.speed_angle_k)
            set_speed_forward = self._clamp(set_speed_forward, -self.speed_k, self.speed_k)
            set_speed_right = self._clamp(set_speed_right, -self.speed_k, self.speed_k)


        self.speed_bind_set.x = set_speed_forward
        self.speed_bind_set.y = set_speed_right
        
        self.position_prev = self.dynamic.get_position()
        self.dynamic.step(set_speed_forward, set_speed_right, set_angle_speed)
        self.time_model = self.dynamic.get_model_time()

        if self.tick % 10 == 0:
            t_point = self.dynamic.get_position()
            self.tick_point.append(vector.obj(x=t_point.x, y=t_point.y))
            t_action = vector.obj(x=set_angle_speed, y=set_speed_forward, z=set_speed_right)
            self.tick_action.append(t_action)

        step_reward = 0.
        terminated = False
        truncated = False

        if self.continuous == True:
            step_reward, terminated, truncated = self.calc_step_reward_box(set_angle_speed, set_speed_forward, set_speed_right)
        else:
            step_reward, terminated, truncated = self.calc_step_reward_discrete(action)

        # вознаграждение за правильное решение
        self.reward += step_reward

        vec_dist = self.target_point - self.dynamic.get_position()
        vec_dist.rho -= self.finish_radius + 1
        d_azimut = vec_dist.deltaphi(self.dynamic.get_direction())
        speed_bind = self.dynamic.get_speed_bind()
        observation = np.empty(self.observation_space_local.shape, dtype=np.float32)
        observation[0] = self._output_norm(self.speed_k, speed_bind.x, True) 
        observation[1] = self._output_norm(self.speed_k, speed_bind.y, True)
        observation[2] = self._output_norm(self.angle_k, d_azimut, True)
        observation[3] = self._output_norm(self.locate_k, vec_dist.rho, True)
        
        for obs in observation: 
            if math.isnan(obs) or math.isinf(obs):
                print(self.target_point, self.dynamic.get_position(), self.dynamic.get_direction())

        info = self._get_info()
        self.tick += 1

        if self.is_pygame == True:
            if self.time_model > self.time_next_human_draw:
                self.time_next_human_draw = self.time_model + self.time_step_human_draw
                self.human_render(self._get_figures(False))

        if self.observation_render == True:
            return self.get_cnn_observation(), step_reward, terminated, truncated, info
        else:
            return observation, step_reward, terminated, truncated, info



    def teach_action(self):
        return self.dynamic.teach_action(self.target_point, self.continuous)



    def calc_step_reward_box(self, set_angle_speed: float, set_speed_forward: float, set_speed_right: float):

        if self.time_model_max < self.time_model:  # время вышло
            return -3, False, True

        position = self.dynamic.get_position()
        dir_view = self.dynamic.get_direction()
        angle_speed = self.dynamic.get_angle_speed()
        speed_bind = self.dynamic.get_speed_bind()

        #ушел за границу зоны
        if position.x < -self.zero.x or position.y < -self.zero.y or position.x > self.zero.x or position.y > self.zero.y:
            return -1, False, True


        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        vec_to_finish = self.target_point - position
        # пришли
        if vec_to_finish.rho < self.finish_radius:

            time_fine = self.time_model_optimum / self.time_model
            return time_fine, True, False
            
        angle_10 = math.pi / 18.  # 10 градусов

        # единичный вектор на целевую точку        
        dir_to_target_point = vec_to_finish / vec_to_finish.rho
        vec_to_finish = dir_to_target_point * (vec_to_finish.rho - self.finish_radius + 1)

        # векторное расстояние пройденное за шаг моделирования
        way_vec = position - self.position_prev

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
                if f.f_equivalent_vec3(self.tick_action[n], self.tick_action[n-1], 0.01):
                    count_action += 1
            
            if abs(summ_10_step_dist) < self.speed_k and count_action > 7:
                self.stoper_step_reward = -0.001
                #print (summ_10_step_dist)
                #return -0.01, False, False
        
        # стоять не надо
        #if way_vec.rho < 0.03:
        #    self.stoper_step_reward += -0.001


        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        ## Вознаграждение за ПОВОРОТ ( set_angle_speed )

        # косинус угола на который надо довернуть
        cos_delta_angle = dir_view @ dir_to_target_point
        
        if vec_to_finish.rho > self.finish_radius + 5:
            self.angle_step_reward = 0.0002 * ( 2*(cos_delta_angle) ** 3 - 4*(cos_delta_angle) * angle_speed**2 - 1)
                    

        ## Вознаграждение за Скорость ( set_speed_forward  set_speed_right )

        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        # Скорость сближения низкая

        if way_step < 0:
            self.speed_step_reward = -0.0007 * (abs(way_step)**(1/3))
        else:
            self.speed_step_reward = 0.01 * way_step**3

        #self.speed_step_reward = 0.001 * way_step

        #self.speed_step_reward = 0.001
        #if abs(way_step) < 0.1:
        #    self.zero_speed_step += 1
        #    if self.zero_speed_step > 10: # штраф за не изменение расстояния до целевой точки (чтоб не кружил вокруг)
        #        self.speed_step_reward = -0.002
        #    else:
        #        self.speed_step_reward *= way_step 
        #
        #else:
        #
        #    self.zero_speed_step = 0
        #    
        #    # удаляемся штраф увличивается пропорционально скорости, 
        #    # приближаемся награда увеличивается пропорционально скрости
        #    self.speed_step_reward *= way_step 
        #
        #    # усиление награды при прямом сближении
        #    if way_step > 0.1:
        #        # косинус угла меду направлением движения и направление на целевую точку
        #        cos_way = 1
        #        if way_vec.rho > 0.0001:
        #            cos_way = way_step / way_vec.rho
        #    
        #        # угол смещения направления на целевую точку
        #        angle_rate = angle_10
        #        if vec_to_finish.rho < 50:
        #            angle_rate = angle_10 * 0.5   # 5 градусов
        #        elif vec_to_finish.rho < 10:
        #            angle_rate = angle_10 * 0.2   # 2 градуса
        #        # чем ближе тем точнее надо наводится
        #        if cos_way > math.cos(angle_rate):
        #            self.speed_step_reward *= 2
        #    else:
        #        self.speed_step_reward *= 1.2

        #!!!!!!!!!!!!!!!!!!!!!!!!!!----REWARD-----
        ## Вознаграждение за рассогласование вектора скорости и вектора взгляда
        self.view_step_reward = 0
        # только для сближения и на большем удалении (около цели можно двигаться любым способом)
        # уменяшает награду за сближение с целевой точкой если сближение происходит боком
        if speed_bind.rho > 0.5 and vec_to_finish.rho > self.finish_radius + 5:
            cos_view_to_speed = vector.obj(x=1,y=0) @ speed_bind / speed_bind.rho
            self.view_step_reward = 0.0002 * (cos_view_to_speed**3 - 0.7 * math.sqrt(abs(speed_bind.y)) )

        step_reward = self.angle_step_reward + self.speed_step_reward + self.view_step_reward + self.stoper_step_reward
        #print(f'REWARDS: Angle {angle_step_reward}, Speed {speed_step_reward}, View {view_step_reward}, Stoper {stoper_step_reward}')
        return step_reward, False, False
    
    def calc_step_reward_discrete(self, action):

        if self.time_model_max < self.time_model:  # время вышло
            return -5, False, True

        position = self.dynamic.get_position()

        #ушел за границу зоны
        if position.x < -self.zero.x or position.y < -self.zero.y or position.x > self.zero.x or position.y > self.zero.y:
            return -2, False, True
        
        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        vec_to_finish = self.target_point - position
        if vec_to_finish.rho < self.finish_radius:
            if self.time_model < 0.4*self.time_model_max:
                return 5, True, False    
            else:
                return 2, True, False
        
        nomove_reward = 0.

        # единичный вектор на целевую точку        
        dir_to_target_point = vec_to_finish / vec_to_finish.rho
        vec_to_finish = dir_to_target_point * (vec_to_finish.rho - self.finish_radius + 1)
        # векторное расстояние пройденное за шаг моделирования
        way_vec = position - self.position_prev

        if way_vec.rho < 0.03:
            nomove_reward = -0.001

        # расстояние на которой приблизились к целевой точке за шаг моделирования
        # мгновенная скорость сближения
        way_step = way_vec @ dir_to_target_point
        
        # угол на который надо довернуть
        delta_angle = self._get_course_bind(vec_to_finish.phi)
        
        angle_step_reward = 0

        if action == 0:
            angle_step_reward = -0.01
        elif action == 1 or action == 2:
            angle_step_reward = -0.001

            # команда на разворот в противоположную сторону - шрафуем
            if delta_angle > 0 and action == 1:
                angle_step_reward *= 10
            elif delta_angle < 0 and action == 2:
                angle_step_reward *= 10
            #elif action == 2 and delta_angle > 0.01 :
            #    angle_step_reward *= (math.pi - delta_angle)
            #elif action == 1 and delta_angle < -0.01 :
            #    angle_step_reward *= (math.pi + delta_angle)
            #if way_step < 1:
            #    angle_step_reward *= 10
            
        elif action == 3:
            angle_step_reward = -0.01
        elif action == 4:
            angle_step_reward = 0.001
        elif action == 5:
            angle_step_reward = -0.01
        elif action == 6:
            angle_step_reward = -0.01

        # усиление награды при прямом сближении
        move_reward = 0.001
        if way_step > 1:

            # косинус угла меду направлением движения и направление на целевую точку
            cos_way = 1
            if way_vec.rho > 0.0001:
                cos_way = way_step / way_vec.rho
        
            # угол смещения направления на целевую точку
            angle_rate = math.pi / 18.  # 10 градусов
            if vec_to_finish.rho < 50:
                angle_rate *= 0.5   # 5 градусов
            elif vec_to_finish.rho < 10:
                angle_rate *= 0.2   # 2 градуса
            # чем ближе тем точнее надо наводится
            if cos_way > math.cos(angle_rate):
                move_reward *= 10
        else:
            move_reward *= way_step


        return angle_step_reward + move_reward + nomove_reward, False, False

   
    def _get_figures(self,to_observation:bool):

        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)

        dir_view = self.dynamic.get_direction()
        speed_local = self.dynamic.get_speed()

        # позиция агента
        pos = self.zero + self.dynamic.get_position()
        
        figures = []
        
        figures.append({'figure':'rectangle', 'start': (1,1), 'end': (self.zero.x * 2 - 1,self.zero.x * 2 - 1), 'out_color':red, 'width':4})

        target_circle = self.zero + self.target_point
        figures.append({'figure':'ellipse', 'senter': (target_circle.x,target_circle.y), 'radius':self.finish_radius, 'in_color': blue, 'out_color':blue, 'width':0})
        
        
        # когда среда это картинка то траеторию не рисуем
        if to_observation == False:
            for point in self.tick_point:
                r_point = self.zero + point
                figures.append({'figure':'point', 'senter':(r_point.x, r_point.y), 'radius':2, 'out_color': green, 'width':0})

        # рисуем треугольник
        triangle = pos + 5 * dir_view
        triangle_width = 10
        triangle_height = 10
        corner2 = vector.obj(x = -triangle_height, y = triangle_width/2)
        corner3 = vector.obj(x = -triangle_height, y = -triangle_width/2)
        corner2 = corner2.rotateZ(dir_view.phi)
        corner3 = corner3.rotateZ(dir_view.phi)
        corner2 += triangle
        corner3 += triangle
        base = pos - 2 * dir_view

        figures.append({'figure':'polygon', 'points':[(triangle.x, triangle.y),
                                                      (corner2.x, corner2.y),
                                                      (base.x, base.y),
                                                      (corner3.x, corner3.y)]
                                                                            , 'in_color': red, 'out_color':red, 'width':0})
        

        if speed_local.rho > 0:
            dir_move = speed_local/speed_local.rho
            srt = pos + 10 * dir_move
            end = srt + 10 * speed_local
            figures.append({'figure':'line', 'start': (srt.x,srt.y), 'end':(end.x,end.y), 'in_color':red, 'width':1})

            set_speed = vector.obj(phi=dir_view.phi + self.speed_bind_set.phi,rho=self.speed_bind_set.rho)
            dir_move = set_speed/set_speed.rho
            srt = pos + 10 * dir_move
            end = srt + 4 * set_speed
            figures.append({'figure':'line', 'start': (srt.x,srt.y), 'end':(end.x,end.y), 'in_color':green, 'width':1})


        return figures
    
    def render(self):
        #if self.is_render == False:
        #    if render_mode == None or render_mode != 'rgb_array':
        #        return None
        return self._get_render(self._get_figures(False), False)
    
    def get_cnn_observation(self):
        return self._get_render(self._get_figures(True), True)
    
    def _get_render(self, figures:list, to_observation: bool):

        black = (0, 0, 0)
        
        im = Image.new('RGB', (self.zero.x * 2, self.zero.y * 2), black)
        draw = ImageDraw.Draw(im)

        for figure in figures:
            if figure['figure'] == 'rectangle':
                draw.rectangle((figure['start'][0], figure['start'][1], figure['end'][0], figure['end'][1]), outline=figure['out_color'], width=figure['width'])
            elif figure['figure'] == 'ellipse':
                draw.ellipse((figure['senter'][0]-figure['radius'], 
                            figure['senter'][1]-figure['radius'],
                            figure['senter'][0]+figure['radius'], 
                            figure['senter'][1]+figure['radius']), fill=figure['in_color'], outline=figure['out_color'])
            elif figure['figure'] == 'point':
                draw.point((figure['senter'][0], figure['senter'][1]), fill=figure['out_color'])
            elif figure['figure'] == 'polygon':
                draw.polygon(xy=(tuple(figure['points'])), fill=figure['in_color'], outline=figure['out_color'])
            elif figure['figure'] == 'line':
                draw.line(xy=((figure['start'][0], figure['start'][1]), (figure['end'][0], figure['end'][1])),fill=figure['in_color'])


        if to_observation == True:
            im = im.convert('L')
            im_np = np.asarray([im])
            im_np = im_np.transpose((1, 2, 0))
        else:
            im_np = np.asarray(im)
        return im_np

    def human_render(self, figures:list):
    
        black = (0, 0, 0)
        self.screen.fill(black)

        for figure in figures:
            if figure['figure'] == 'ellipse' or figure['figure'] == 'point':
                pygame.draw.circle(self.screen, figure['out_color'], figure['senter'], figure['radius'], figure['width'])
            elif figure['figure'] == 'polygon':
                pygame.draw.polygon(self.screen, figure['out_color'], figure['points'], figure['width'])
            elif figure['figure'] == 'line':
                pygame.draw.line(self.screen, figure['in_color'], list(figure['start']), list(figure['end']), figure['width'])

        pygame.display.update()
        # устанавливаем частоту обновления экрана
        pygame.time.Clock().tick(60)


