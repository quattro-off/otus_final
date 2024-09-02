import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
import copy
from PIL import Image, ImageDraw


#Caucasus terrain map
# minX -1000, minZ -1000, maxX 1000, maxZ 1000
# X, Z, Vz, Vz, course
observation_space = Box(low=np.array([-300.,-300.,-3.6,-3.6,-math.pi], dtype=np.float32), high=np.array([300.,300.,3.6,3.6,math.pi], dtype=np.float32), shape=(5,), dtype=np.float32)

#                 course move(-pi,pi), speed(0,3.6)
action_space = Box(low=np.array([-1, 0], dtype=np.float32), high=np.array([1, 1], dtype=np.float32), shape=(2,), dtype=np.float32)

action_space_d = Discrete(5, seed=42)

class HumanMoveSimpleAction(gym.Env):
    """непрерывные состояния, непрерывные действия"""

    continuous = True
    observation_render = False
    reward = 0.
    tick = 0

    tick_point = []


    #constants
    time_step = 0.02
    time_tick = 1.
    speed_acc = 2. #m2/s
    angle_speed = 2. #rad/s
    min_speed = 0.44 #m/s
    max_speed = 3.6 #m/s
    desc_step_speed = 0.5 #m/s
    desc_step_angle_speed = 0.1 #rad/s
    finish_radius = 3 #m

    #variable
    time_model = 0.
    time_model_max = 0.
    position_prev = vector.obj(x=0.,y=0.)
    position = vector.obj(x=0.,y=0.)
    v_speed = vector.obj(x=0.,y=0.)
    speed = 0.
    dir_move = vector.obj(x=1.,y=0.)
    course = 0.
    desc_angle_speed = 0

    target_point = vector.obj(x=100.,y=100.)

    if_render = False
    is_pygame = False
    screen = {}
    zero = vector.obj(x=300, y=300)

    def name(self):
        return "HumanMoveSimple"

    def __init__(self, continuous: bool = True, seed: int=42, render_mode: str=None, observation_render: bool = False):
        self.render_mode = render_mode
        self.reward_range = (-float(600), float(600))
        self.continuous = continuous
        self.observation_render = observation_render
        self.observation_space = observation_space
        if continuous == True:
            self.action_space = action_space
        else:
            self.action_space = action_space_d
            self.action_space.seed(seed=seed)

        super().reset(seed=seed)
        random.seed(seed)
        self.observation_space.seed(seed)

        if render_mode == 'human':
            self.is_render = False
            self.is_pygame = True
            pygame.init()
            self.screen = pygame.display.set_mode((self.zero.x * 2, self.zero.y * 2))
            pygame.display.set_caption("Moving")
        elif render_mode == 'rgb_array':
            self.is_pygame = False
            self.is_render = True

    def reset(self, seed: int = None, options=None):

        if seed != None:
            random.seed(seed)
            self.observation_space.seed(seed)

        start_observation = self.observation_space.sample()

        self.time_model = 0.
        self.position = vector.obj(x=start_observation[0],y=start_observation[1])
        self.v_speed = vector.obj(x=start_observation[2],y=start_observation[3])
        self.speed = self.v_speed.rho
        self.dir_move = vector.obj(x=self.v_speed.x/self.v_speed.rho,y=self.v_speed.y/self.v_speed.rho)
        self.course = self.dir_move.phi

        self.target_point.x = 0.
        self.target_point.y = 0.

        dist = self.position.rho
        self.time_model_max = 5 * dist / self.max_speed

        self.tick = 0
        self.reward = 0.
        self.tick_point.clear()
        
        info = self._get_info()

        if self.observation_render == True:
            rend = self._get_render()
            return rend, info
        else:
            return start_observation, info


    def close(self):
        if self.is_pygame == True:
            pygame.quit()
        super().close()


    def _get_info(self):
        return {"tick progress: ": self.tick}
    
    def _get_length(self):
        return int(self.time_model_max)

    def step(self, action, is_teacher: bool = False):

        teach_observation = self.teach_move() if is_teacher == True else None

        set_angle_dir   = self.dir_move.phi         # action - 1(-) - 2(+)
        set_speed       = self.speed                # action - 3(-) - 4(+)

        if self.continuous == True:
            set_angle_dir   = action[0] * math.pi
            set_speed       = action[1] * 3.6
        else:

            curr_angle_speed = 0
            if action == 1:
                curr_angle_speed = -self.desc_step_angle_speed
            elif action == 2:
                curr_angle_speed = self.desc_step_angle_speed
            elif action == 3:
                set_speed = self.speed - self.desc_step_speed
            elif action == 4:
                set_speed = self.speed + self.desc_step_speed

            curr_angle_speed = self._clamp(curr_angle_speed, -self.angle_speed, self.angle_speed)
            set_angle_dir += curr_angle_speed

        self.simple_move(set_angle_dir, set_speed)
        step_reward, terminated, truncated = self.calc_step_reward(set_speed)

        # вознаграждение за правильное решение
        self.reward += step_reward

        observation = np.array([self.position.x, self.position.y, self.v_speed.x, self.v_speed.y, self.course])

        info = self._get_info()
        self.tick += 1

        if self.is_pygame == True:
            self.human_render()

        if self.observation_render == True:
            return self._get_render()
        else:
            if is_teacher == False:
                return observation, step_reward, terminated, truncated, info
            else:
                return observation, step_reward, terminated, truncated, info, teach_observation

    def simple_move(self, set_angle_dir: float, set_speed: float):

        self.position_prev.x = self.position.x
        self.position_prev.y = self.position.y

        # заданный вектор и скорость движения
        set_dir = vector.obj(rho=1, phi=set_angle_dir)

        new_speed = self.speed
        new_position = vector.obj(x=self.position.x, y = self.position.y)

        # угол на который можно развернутся за шаг
        angle_step = self.angle_speed * self.time_step
        cos_angle_step = math.cos(angle_step)

        final_time = self.time_model + self.time_tick
        while self.time_model < final_time:
            self.time_model += self.time_step

            # косинус угла между текущим и заданным курсами
            cos_rotate_angle = self.dir_move @ set_dir

            # когда разница между текущим курсом и заданным курсом дольше гула поворота за шаг
            if cos_rotate_angle < cos_angle_step:
                self_right = vector.obj(x=-self.dir_move.y, y=self.dir_move.x)
                cos_side_rotate = set_dir @ self_right
                angle_sign = -1 if cos_side_rotate < 0 else 1
                # доворачивает текщий вектор движения в сторону заданного на угол за шаг
                self.dir_move = self.dir_move.rotateZ(angle_sign * angle_step)
            else:
                self.dir_move = set_dir

            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_sign = 1.
            if set_speed < new_speed:
                speed_sign = -1.
            # меняем скорость
            new_speed += speed_sign * self.speed_acc * self.time_step
            new_speed = self._clamp(new_speed, 0, self.max_speed)

            # движеие вне мертвой области 
            if -self.min_speed > new_speed or new_speed > self.min_speed:
                new_position += new_speed * self.time_step * self.dir_move

        self.course = self.dir_move.phi            

        # мертвая область
        new_speed = self._death(new_speed, -self.min_speed, self.min_speed)
        self.v_speed.x = self.dir_move.x * new_speed
        self.v_speed.y = self.dir_move.y * new_speed
        self.speed = new_speed

        self.position.x = new_position.x
        self.position.y = new_position.y

        t_point = vector.obj(x=self.position.x, y = self.position.y)
        self.tick_point.append(t_point)


    def teach_move(self):

        # заданный вектор и скорость движения
        teach_set_dir = self.target_point - self.position
        teach_set_speed = 3.6
        if teach_set_dir.rho < 2:
            teach_set_speed = 0.5
        dist_to_point = 1/teach_set_dir.rho
        teach_set_dir = teach_set_dir * dist_to_point

        # текущая позиция
        teach_speed = self.speed
        teach_v_speed = vector.obj(x=self.v_speed.x, y = self.v_speed.y)
        teach_position = vector.obj(x=self.position.x, y = self.position.y)
        teach_dir_move = self.dir_move

        # угол на который можно развернутся за шаг
        angle_step = self.angle_speed * self.time_step
        cos_angle_step = math.cos(angle_step)

        start_time = self.time_model
        final_time = self.time_model + self.time_tick
        while start_time < final_time:
            start_time += self.time_step

            # косинус угла между текущим и заданным курсами
            cos_rotate_angle = teach_dir_move @ teach_set_dir

            # когда разница между текущим курсом и заданным курсом дольше гула поворота за шаг
            if cos_rotate_angle < cos_angle_step:
                self_right = vector.obj(x=-teach_dir_move.y, y=teach_dir_move.x)
                cos_side_rotate = teach_set_dir @ self_right
                if cos_side_rotate < 0:
                    angle_step *= -1
                # доворачивает текщий вектор движения в сторону заданного на угол за шаг
                teach_dir_move = teach_dir_move.rotateZ(angle_step)
            else:
                teach_dir_move = teach_set_dir

            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_sign = 1.
            if teach_set_speed < teach_speed:
                speed_sign = -1.
            # меняем скорость
            teach_speed += speed_sign * self.speed_acc * self.time_step
            teach_speed = self._clamp(teach_speed, -self.max_speed, self.max_speed)

            # движеие вне мертвой области 
            if -self.min_speed > teach_speed or teach_speed > self.min_speed:
                teach_position += teach_speed * self.time_step * teach_set_dir

        # мертвая область
        teach_speed = self._death(teach_speed, -self.min_speed, self.min_speed)
        teach_v_speed.x = teach_dir_move.x * teach_speed
        teach_v_speed.y = teach_dir_move.y * teach_speed

        return np.array([teach_position.x, teach_position.y, teach_v_speed.x, teach_v_speed.y, teach_dir_move.phi])

    def teach_action(self, state):

        t_action = {}
            
        position = vector.obj(x=state[0], y = state[1])
        teach_set_dir = -position

        if self.continuous == True:
            teach_set_speed = 1
            if teach_set_dir.rho < 2:
                teach_set_speed /= 7
            course = teach_set_dir.phi / math.pi 
            t_action = np.array([course, teach_set_speed])
        else:
            delta_angle = teach_set_dir.phi - self.course
            sign = -1 if delta_angle >= 0 else 1
            if abs(delta_angle) > math.pi:
                delta_angle = sign * (2 * math.pi - abs(delta_angle))

            if abs(delta_angle) > 5 * math.pi / 180 :
                if self.speed > 1:
                    t_action = 3
                else:
                    t_action = 2 if delta_angle >= 0 else 1
            else:
                if teach_set_dir.rho < 1:
                    t_action = 3 if self.speed > 0 else 4
                elif teach_set_dir.rho < 4:
                    t_action = 3 if self.speed >= 0.5 else 4
                else:
                    t_action = 4


        return t_action
    


    def calc_step_reward(self, set_speed:float):

        if self.time_model_max < self.time_model:  # время вышло
            return -100, False, True


        #ушел за границу зоны
        if self.position.x < -self.zero.x or self.position.y < -self.zero.y or self.position.x > self.zero.x or self.position.y > self.zero.y:
            return -100, False, True

        step_reward = 1

        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        vec_to_finish = self.target_point - self.position

        # расстояние до целевой точки
        dist_to_finish = vec_to_finish.rho
    
        if dist_to_finish < self.finish_radius: # пришли
            return 100, False, True

        # единичный вектор на целевую точку        
        dist_to_finish = 1./dist_to_finish
        dir_to_target_point = vec_to_finish * dist_to_finish

        # векторное расстояние пройденное за шаг моделирования
        way_vec = self.position - self.position_prev

        # расстояние на которой приблизились к целевой точке за шаг моделирования
        way_step = way_vec @ dir_to_target_point

        # косинус угла меду направлением движения и направление на целевую точку
        cos_way = 0
        if way_vec.rho > 0.0001:
            cos_way = way_step / way_vec.rho

        # коэффициент усиления награды в случае прямого движения на целевую точку
        k_way = 1.
        # угол смещения направления на целевую точку
        angle_rate = math.pi / 18.  # 10 градусов
        if dist_to_finish < 50:
            angle_rate *= 0.5   # 5 градусов
        elif dist_to_finish < 10:
            angle_rate *= 0.2   # 2 градуса
        # чем ближе тем точнее надо наводиться
        if cos_way > math.cos(angle_rate):
            k_way = 2.

        if way_step > -1 and way_step < 1:
            step_reward = -1
        else:
            delta_step = 0.5
            if way_step < 0:
                delta_step = -2
            step_reward = k_way * way_step + delta_step

        if set_speed > 0.5:
            if self.speed < 0.5: # при заданной скорости вдруг остановка
                step_reward = -10
            else:
                step_reward + self.speed
        else: #просто стоять тоже нельзя
            step_reward = -10

        return step_reward, False, False

    def _get_render(self):
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
    
    def render(self):
        #if self.is_render == False:
        #    if render_mode == None or render_mode != 'rgb_array':
        #        return None
        return self._get_render()

    def human_render(self):
    
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)

        self.screen.fill(black)

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