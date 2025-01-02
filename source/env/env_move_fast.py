import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw
from typing import Any, Dict, Tuple, Union


# TO (0,1)
# Vx_bind, Vy_bind, dAzt, Dist
observation_space = Box(low=np.array([-1,-1,-1,0], dtype=np.float32), high=np.array([1,1,1,1], dtype=np.float32), shape=(4,), dtype=np.float32)

#                 speed forward(-3.6,3.6) m/s speed right(-3.6,3.6) m/s, course speed(-pi/3,pi/3) rad/s
action_space = Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), shape=(3,), dtype=np.float32)

action_space_d = Discrete(7, seed=42)

observation_space_rend = Box(0, 255, (600, 600, 1), dtype=np.uint8)

class MoveFastAction(gym.Env):



    #observation
    locate_k = 900
    speed_k = 3.6
    speed_angle_k = math.pi/3
    angle_k = math.pi

    #constants
    time_step = 0.02
    time_tick = 0.1
    speed_acc = 2. #m2/s
    angle_speed_acc = 2. #rad/s

    min_speed = 0.44 #m/s
    max_speed = 3.6 #m/s
    desc_step_speed = 5#0.5 #m/s
    desc_step_angle_speed = 0.1 #rad/s
    finish_radius = 50 #m


    def _is_norm(self, value: float)->bool:
        return False if value < 0. or value > 1. else True


    def _output_norm(self, koef: float, value: float, unsigned: bool = False):
        result = value / koef if unsigned == True else value / (2*koef) + 0.5
        #if koef != 300 and self._is_norm(result) == False:
        #    print(f'Warning: Bad normolize value: {value} by koef: {koef} - unsigned: {unsigned}')
        result = self._clamp(result,0.,1.)
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
                 continuous: bool = True, 
                 seed: int=42, 
                 observation_render: bool = False,
                 target_point_rand:bool = False,
                 render_mode: str=None, 
                 render_time_step: float=1., 
                 options = None  #options={'finish_dist':100,'start_dist':110,'delta_phi':0}
                 ):

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
        self.time_model = 0.
        self.time_model_max = 0.
        self.position_prev = vector.obj(x=0.,y=0.)
        self.position = vector.obj(x=0.,y=0.)
        self.speed_local =vector.obj(x=0.,y=0.) # в локальной системме координат 
        self.speed_bind = vector.obj(x=0.,y=0.) # в связанной системме координат
        self.angle_speed = 0.
        self.dir_view = vector.obj(x=1.,y=0.)
        self.desc_angle_speed = 0
        self.target_point = vector.obj(x=100.,y=100.)
        self.vec_dist_to_obctacle_bind = vector.obj(x=100.,y=100.)

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

        if self.options == None:

            self.finish_radius = 5

            # начальная позиция
            self.position.x = random.randint(int(-self.zero.x), int(self.zero.x))
            self.position.y = random.randint(int(-self.zero.y), int(self.zero.y))
            #начальная скорость
            start_speed = math.radians(random.randint(-180,180))
            self.speed_bind = vector.obj(phi=start_speed,rho=random.randint(1,3))
            self.speed_bind_set = vector.obj(phi=self.speed_bind.phi,rho=self.speed_bind.rho)
            start_phi = math.radians(random.randint(-180,180))
            self.dir_view = vector.obj(phi=start_phi,rho=1)
        else:

            self.finish_radius = self.options['finish_dist']

            # начальная позиция
            phi = random.randint(int(-180), int(180))
            self.position.rho = self.options['start_dist']
            self.position.phi = math.radians(phi)

            #начальная скорость
            self.speed_bind=vector.obj(x=0,y=0)
            self.speed_bind_set = vector.obj(x=0,y=0)

            #начaльный курс
            dir = self.target_point - self.position
            start_dalta_phi = math.radians(random.randint(-self.options['delta_phi'], self.options['delta_phi']))
            self.dir_view = vector.obj(phi=dir.phi + start_dalta_phi,rho=1)

        self.vec_dist_to_obctacle_bind = vector.obj(x=100.,y=100.)

        vec_dist = self.target_point - self.position
        self.time_model_optimum = vec_dist.rho / self.max_speed
        self.time_model_max = 3 * self.time_model_optimum

        vec_dist.rho -= self.finish_radius + 1
        d_azimut = vec_dist.deltaphi(self.dir_view)

        start_observation[0] = self._output_norm(self.speed_k, self.speed_bind.x, True)  
        start_observation[1] = self._output_norm(self.speed_k, self.speed_bind.y, True)
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
            rend = self._get_render(True)
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

        set_angle_speed = self.angle_speed      # action - 1(-) - 2(+)
        set_speed_forward = self.speed_bind.x        # action - 3(-) - 4(+)
        set_speed_right = self.speed_bind.y          # action - 5(-) - 6(+)


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
        
        self.simple_move(set_angle_speed, set_speed_forward, set_speed_right)

        step_reward = 0.
        terminated = False
        truncated = False

        if self.continuous == True:
            step_reward, terminated, truncated = self.calc_step_reward_box(set_angle_speed, set_speed_forward, set_speed_right)
        else:
            step_reward, terminated, truncated = self.calc_step_reward_discrete(action)

        # вознаграждение за правильное решение
        self.reward += step_reward

        vec_dist = self.target_point - self.position
        vec_dist.rho -= self.finish_radius + 1
        d_azimut = vec_dist.deltaphi(self.dir_view)
        observation = np.empty(self.observation_space_local.shape, dtype=np.float32)
        observation[0] = self._output_norm(self.speed_k, self.speed_bind.x, True) 
        observation[1] = self._output_norm(self.speed_k, self.speed_bind.y, True)
        observation[2] = self._output_norm(self.angle_k, d_azimut, True)
        observation[3] = self._output_norm(self.locate_k, vec_dist.rho, True)
        
        for obs in observation: 
            if math.isnan(obs) or math.isinf(obs):
                print(self.target_point, self.position, self.dir_view )

        info = self._get_info()
        self.tick += 1

        if self.is_pygame == True:
            if self.time_model > self.time_next_human_draw:
                self.time_next_human_draw = self.time_model + self.time_step_human_draw
                self.human_render()

        if self.observation_render == True:
            return self._get_render(True), step_reward, terminated, truncated, info
        else:
            return observation, step_reward, terminated, truncated, info


    def simple_move(self, set_angle_speed: float, set_speed_forward: float, set_speed_right: float):

        self.position_prev.x = self.position.x
        self.position_prev.y = self.position.y

        # заданный вектор и скорость движения
        #set_dir = vector.obj(rho=1, phi=set_angle_dir)

        new_speed = vector.obj(x=self.speed_bind.x, y=self.speed_bind.y)
        new_position = vector.obj(x=self.position.x, y = self.position.y)
        new_angle_speed = self.angle_speed

        final_time = self.time_model + self.time_tick
        while self.time_model < final_time:
            self.time_model += self.time_step


            # угловая скорость вправо - влево
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            angle_speed_sign = 1.
            if self._equivalent(set_angle_speed, self.angle_speed) == True:
                angle_speed_sign = 0
            elif set_angle_speed < self.angle_speed:
                angle_speed_sign = -1.
            # меняем угловую скорость
            new_angle_speed += angle_speed_sign * self.angle_speed_acc * self.time_step
            new_angle_speed = self._clamp(new_angle_speed, -self.speed_angle_k, self.speed_angle_k)
            # доворачивает текщий вектор движения в сторону заданного на угол за шаг
            # угол на который можно развернутся за шаг
            angle_step = new_angle_speed * self.time_step
            self.dir_view = self.dir_view.rotateZ(angle_step)
            dir_view_right = vector.obj(x=-self.dir_view.y,y=self.dir_view.x)

 
            # скорость вперед - назад
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_forward_sign = 1.
            if self._equivalent(set_speed_forward, new_speed.x) == True:
                speed_forward_sign = 0
            elif set_speed_forward < new_speed.x:
                speed_forward_sign = -1.


            # скорость вправо - влево
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_right_sign = 1.
            if self._equivalent(set_speed_right, new_speed.y) == True:
                speed_right_sign = 0
            elif set_speed_right < new_speed.y:
                speed_right_sign = -1.

            
            # меняем скорость
            new_speed.x += speed_forward_sign * self.speed_acc * self.time_step
            new_speed.x = self._clamp(new_speed.x, -self.speed_k, self.speed_k)
            new_speed.y += speed_right_sign * self.speed_acc * self.time_step
            new_speed.y = self._clamp(new_speed.y, -self.speed_k, self.speed_k)

            if new_speed.rho > self.speed_k:
                new_speed /= new_speed.rho
                new_speed *= self.speed_k

            # проверяем препятсвия по пути
            if self.vec_dist_to_obctacle_bind.rho < 2 * self.speed_k * self.time_tick:
                cos_obst = new_speed @ self.vec_dist_to_obctacle_bind
                if cos_obst > 0:
                    right_obst = vector.obj(x=-self.vec_dist_to_obctacle_bind.y, y=self.vec_dist_to_obctacle_bind.x)
                    right_speed = new_speed @ right_obst / right_obst.rho
                    new_speed = right_speed * right_obst / right_obst.rho


            # переводим скорость в локальную системму координат 
            speed = new_speed.x * self.dir_view + new_speed.y * dir_view_right

            # движение в мертвой области нет
            if -self.min_speed > speed.rho or speed.rho > self.min_speed:
                new_position += speed * self.time_step


        if self.dir_view.phi > math.pi or self.dir_view.phi < -math.pi:
            print(set_angle_speed)
            print(self.dir_view)

        self.speed_local = new_speed.x * self.dir_view + new_speed.y * dir_view_right

        self.speed_bind.x = new_speed.x
        self.speed_bind.y = new_speed.y

        self.position.x = new_position.x
        self.position.y = new_position.y

        self.angle_speed = new_angle_speed

        if self.tick % 10 == 0:
            t_point = vector.obj(x=self.position.x, y = self.position.y)
            self.tick_point.append(t_point)
            t_action = vector.obj(x=set_angle_speed, y=set_speed_forward, z=set_speed_right)
            self.tick_action.append(t_action)

    def teach_action(self):

        t_action = []
            
        teach_set_dir = self.target_point - self.position

        delta_angle = self._get_course_bind(teach_set_dir.phi)

        if self.continuous == True:
            teach_set_speed = 1
            if teach_set_dir.rho < 2:
                teach_set_speed /= 7

            delta_angle = self._clamp(delta_angle, -self.speed_angle_k, self.speed_angle_k)
            teach_set_angle_speed = delta_angle / self.speed_angle_k

            t_action = np.array([teach_set_speed, 0, teach_set_angle_speed])

        else:

            if abs(delta_angle) > 5 * math.pi / 180 :
                if self.speed_bind.x > 1:
                    t_action = 3
                else:
                    t_action = 2 if delta_angle >= 0 else 1
            else:
                if teach_set_dir.rho < 1:
                    t_action = 3 if self.speed_bind.x > 0 else 4
                elif teach_set_dir.rho < 4:
                    t_action = 3 if self.speed_bind.x >= 0.5 else 4
                else:
                    t_action = 4


        return t_action


    def calc_step_reward_box(self, set_angle_speed: float, set_speed_forward: float, set_speed_right: float):

        if self.time_model_max < self.time_model:  # время вышло
            return -3, False, True


        #ушел за границу зоны
        if self.position.x < -self.zero.x or self.position.y < -self.zero.y or self.position.x > self.zero.x or self.position.y > self.zero.y:
            return -1, False, True


        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        vec_to_finish = self.target_point - self.position
        # пришли
        if vec_to_finish.rho < self.finish_radius:

            time_fine = self.time_model_optimum / self.time_model
            return time_fine, True, False
           

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
        # Скорость сближения низкая
            
        # удаляемся штраф увличивается пропорционально скорости, 
        # приближаемся награда увеличивается пропорционально скрости
        self.speed_step_reward = 0.001 * way_step 

        step_reward = self.speed_step_reward + self.stoper_step_reward
        #print(f'REWARDS: Speed {speed_step_reward}, Stoper {stoper_step_reward}')
        return step_reward, False, False
    
    def calc_step_reward_discrete(self, action):

        if self.time_model_max < self.time_model:  # время вышло
            return -5, False, True

        #ушел за границу зоны
        if self.position.x < -self.zero.x or self.position.y < -self.zero.y or self.position.x > self.zero.x or self.position.y > self.zero.y:
            return -2, False, True
        
        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        vec_to_finish = self.target_point - self.position
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
        way_vec = self.position - self.position_prev

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
        draw.ellipse((target_circle.x-self.finish_radius, 
                      target_circle.y-self.finish_radius,
                      target_circle.x+self.finish_radius, 
                      target_circle.y+self.finish_radius), fill=blue, outline=blue)
    
        # когда среда это картинка то траеторию не рисуем
        if to_observation == False:
            for point in self.tick_point:
                r_point = self.zero + point
                draw.point((r_point.x, r_point.y), fill=green)

        # рисуем треугольник
        pos = self.zero + self.position
        triangle = pos + 5 * self.dir_view
        triangle_width = 10
        triangle_height = 10
        corner2 = vector.obj(x = -triangle_height, y = triangle_width/2)
        corner3 = vector.obj(x = -triangle_height, y = -triangle_width/2)
        corner2 = corner2.rotateZ(self.dir_view.phi)
        corner3 = corner3.rotateZ(self.dir_view.phi)
        corner2 += triangle
        corner3 += triangle
        base = pos - 2 * self.dir_view

        draw.polygon(
            xy=(
                (triangle.x, triangle.y),
                (corner2.x, corner2.y),
                (base.x, base.y),
                (corner3.x, corner3.y)
            ), fill=red, outline=red
        )

        # вектор скорости
        if self.speed_local.rho > 0:
            dir_move = self.speed_local/self.speed_local.rho
            srt = pos + 10 * dir_move
            end = srt + 5 * self.speed_local
            draw.line(xy=((srt.x,srt.y), (end.x,end.y)),fill=red)


        if to_observation == True:
            im = im.convert('L')
            im_np = np.asarray([im])
            im_np = im_np.transpose((1, 2, 0))
        else:
            im_np = np.asarray(im)
        return im_np
    
    def render(self):
        #if self.is_render == False:
        #    if render_mode == None or render_mode != 'rgb_array':
        #        return None
        return self._get_render(False)

    def human_render(self):
    
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)

        self.screen.fill(black)

        target_circle = self.zero + self.target_point
        pygame.draw.circle(self.screen, blue, (target_circle.x, target_circle.y), self.finish_radius)
    
        for point in self.tick_point:
            r_point = self.zero + point
            pygame.draw.circle(self.screen, green, (r_point.x, r_point.y), 2)

        # рисуем треугольник
        pos = self.zero + self.position
        triangle = pos + 5 * self.dir_view
        triangle_width = 10
        triangle_height = 10
        corner2 = vector.obj(x = -triangle_height, y = triangle_width/2)
        corner3 = vector.obj(x = -triangle_height, y = -triangle_width/2)
        corner2 = corner2.rotateZ(self.dir_view.phi)
        corner3 = corner3.rotateZ(self.dir_view.phi)
        corner2 += triangle
        corner3 += triangle
        base = pos - 2 * self.dir_view

        triangle_points = [(triangle.x, triangle.y),
                           (corner2.x, corner2.y),
                           (base.x, base.y),
                           (corner3.x, corner3.y)]
        pygame.draw.polygon(self.screen, red, triangle_points)

        # вектор скорости
        if self.speed_local.rho > 0:
            dir_move = self.speed_local/self.speed_local.rho
            srt = pos + 10 * dir_move
            end = srt + 5 * self.speed_local
            pygame.draw.aaline(self.screen, red, [srt.x,srt.y], [end.x,end.y])

        pygame.display.update()

        # устанавливаем частоту обновления экрана
        pygame.time.Clock().tick(60)


    def _clamp(self, value: float, min: float, max: float):
        if value > max:
            value = max
        elif value < min:
            value = min
        return value
    
    def _death(self, value: float, min: float, max: float):
        if min < value and value < max:
            value = 0
        return value
    
    def _get_course_bind(self, phi:float, cicle:bool = False)->float:
        if cicle == True:
            phi = phi if phi >= 0 else phi + 2 * math.pi
            view_phi = self.dir_view.phi if self.dir_view.phi >= 0 else self.dir_view.phi + 2 * math.pi
            delta_angle = phi - view_phi
            if delta_angle < 0:
                delta_angle += 2 * math.pi

            if delta_angle < 0 or delta_angle > 2 * math.pi:
                print(delta_angle)
        else:
            delta_angle = phi - self.dir_view.phi
            sign = -1 if delta_angle >= 0 else 1
            if abs(delta_angle) > math.pi:
                delta_angle = sign * (2 * math.pi - abs(delta_angle))
        return delta_angle
    
    def _equivalent(self, value1:float, value2:float, epsilon:float = 0.0001)->bool:
        return True if abs(value1 - value2) < epsilon else False
        
    def _equivalent_vec3(self, value1:vector, value2:vector, epsilon:float = 0.0001)->bool:
        if abs(value1.x - value2.x) < epsilon and abs(value1.y - value2.y) < epsilon and abs(value1.z - value2.z) < epsilon:
            return True
        else:
            return False
        
    def _min_then_max(self, v1:float, v2:float):
        if v1 < v2:
            return v1, v2
        else:
            return v2, v1
        
    def _sign(self, value:float)->float:
        if value < 0.:
            return -1
        else:
            return 1

    def _sign_or_zero(self, value:float)->float:
        if self._equivalent(value, 0.) == True:
            return 0.
        else:
            return self._sign(value)