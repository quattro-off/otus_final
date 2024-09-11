import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw


# dXt (-300,300), dXt (-300,300) 
# Dt t (0, 300)
# Azt t (-pi, pi)

# TO (0,1)
# dXt, dYt, Dt, Azt, 
observation_space = Box(low=np.array([0,0,0,0], dtype=np.float32), high=np.array([1,1,1,1], dtype=np.float32), shape=(4,), dtype=np.float32)

#                 course speed(-pi/3,pi/3) rad/s, speed forward(-3.6,3.6) m/s speed right(-3.6,3.6) m/s
action_space = Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), shape=(3,), dtype=np.float32)

action_space_d = Discrete(7, seed=42)

observation_space_rend = Box(0, 255, (600, 600, 1), dtype=np.uint8)

class HumanMoveSimpleAction(gym.Env):
    """непрерывные состояния, непрерывные действия"""

    continuous = True               # непрерывноеб дискретное действие
    observation_render = False      # среда ввиде матрицы данных или картинка(оттенки серого)
    target_point_rand = False       # целевая точка движения в центре или случайная
    observation_space_local = []    # среда ля вычистений всегда параметрическая
    reward = 0.
    tick = 0

    tick_point = []

    #observation
    locate_k = 300
    speed_k = 3.6
    speed_angle_k = math.pi/3
    angle_k = math.pi

    def _is_norm(self, value: float)->bool:
        return False if value < 0. or value > 1. else True


    def _output_norm(self, koef: float, value: float, unsigned: bool = False):
        result = value / koef if unsigned == True else value / (2*koef) + 0.5
        if koef != 300 and self._is_norm(result) == False:
            print(f'Warning: Bad normolize value: {value} by koef: {koef}')
        return result

    def _input_unnorm_vector(self, koef: float, x: float, y: float):
        return vector.obj(x=(x*2 - 1)*koef,y=(y*2 - 1)*koef)
    
    def _input_unnorm_value(self, koef: float, value: float, unsigned: bool = False):
        if unsigned == True:
            return value * koef
        else:
            return (value*2 - 1)*koef


    #constants
    time_step = 0.02
    time_tick = 1.
    speed_acc = 2. #m2/s
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
    speed_local =vector.obj(x=0.,y=0.) # в локальной системме координат 
    speed_bind = vector.obj(x=0.,y=0.) # в связанной системме координат
    angle_speed = 0.
    
    dir_view = vector.obj(x=1.,y=0.)
    desc_angle_speed = 0

    target_point = vector.obj(x=100.,y=100.)

    if_render = False
    is_pygame = False
    screen = {}
    zero = vector.obj(x=300, y=300)

    def name(self):
        return "HumanMoveSimpleCNN" if self.observation_render == True else "HumanMoveSimple"

    def __init__(self, 
                 continuous: bool = True, 
                 seed: int=42, 
                 render_mode: str=None, 
                 observation_render: bool = False,
                 target_point_rand:bool = False
                 ):
        self.render_mode = render_mode
        self.reward_range = (-float(600), float(600))
        self.continuous = continuous
        self.observation_render = observation_render
        self.target_point_rand = target_point_rand

        if observation_render == True:
            self.observation_space = observation_space_rend
            self.observation_space_local = observation_space
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

    def reset(self, seed: int = None, options=None):

        if seed != None:
            random.seed(seed)
            self.observation_space.seed(seed)
            self.observation_space_local.seed(seed)

        start_observation = self.observation_space_local.sample()

        self.time_model = 0.
        # начальная позиция
        self.position.x = random.randint(-int(self.zero.x), int(self.zero.x))
        self.position.y = random.randint(-int(self.zero.y), int(self.zero.y))
        #начальная скорость
        self.speed_bind = vector.obj(x=random.randint(-3,3),y=random.randint(-3,3))

        #начaльный курс
        if self.speed_bind.rho == 0:
            self.dir_view = vector.obj(x=1,y=0)
        else:
            self.dir_view = vector.obj(x=self.speed_bind.x/self.speed_bind.rho,y=self.speed_bind.y/self.speed_bind.rho)

        # целевая точка движения
        if self.target_point_rand == True:
            self.target_point.x = random.randint(-int(self.zero.x), int(self.zero.x))
            self.target_point.y = random.randint(-int(self.zero.y), int(self.zero.y))
        else:
            self.target_point.x = 0.
            self.target_point.y = 0.

        vec_dist = self.target_point - self.position
        dist = vec_dist.rho
        self.time_model_max = 5 * dist / self.max_speed

        start_observation[0] = self._output_norm(self.locate_k, vec_dist.x)   # dXt, dYt, Dt, Azt, 
        start_observation[1] = self._output_norm(self.locate_k, vec_dist.y)
        start_observation[2] = self._output_norm(self.locate_k, vec_dist.rho)
        start_observation[3] = self._output_norm(self.angle_k, self._get_course_bind(vec_dist.phi))

        self.tick = 0
        self.reward = 0.
        self.tick_point.clear()
        
        info = self._get_info()

        if self.observation_render == True:
            rend = self._get_render(True)
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

    def step(self, action):

        set_angle_speed = self.angle_speed      # action - 1(-) - 2(+)
        set_speed_forward = self.speed_bind.x        # action - 3(-) - 4(+)
        set_speed_right = self.speed_bind.y          # action - 5(-) - 6(+)


        if self.continuous == True:
            set_angle_speed  = action[0] * self.speed_angle_k
            set_speed_forward = action[1] * self.speed_k
            set_speed_right = action[2] * self.speed_k
        else:

            if action == 0:
                set_angle_speed = 0.
                set_speed_forward = 0.
                set_speed_right = 0.
            elif action == 1:
                set_angle_speed = set_angle_speed - self.desc_step_angle_speed if set_angle_speed < 0. else -self.desc_step_angle_speed
                set_speed_forward = 0.
                set_speed_right = 0.
            elif action == 2:
                set_angle_speed = set_angle_speed + self.desc_step_angle_speed if set_angle_speed > 0. else self.desc_step_angle_speed
                set_speed_forward = 0.
                set_speed_right = 0.
            elif action == 3:
                set_angle_speed = 0.
                set_speed_forward = set_speed_forward - self.desc_step_speed if set_speed_forward < 0. else -self.desc_step_speed
                set_speed_right = 0.
            elif action == 4:
                set_angle_speed = 0.
                set_speed_forward = set_speed_forward + self.desc_step_speed if set_speed_forward > 0. else self.desc_step_speed
                set_speed_right = 0.                
            elif action == 5:
                set_angle_speed = 0.
                set_speed_forward = 0.
                set_speed_right = set_speed_right - self.desc_step_speed if set_speed_right < 0. else -self.desc_step_speed
            elif action == 6:
                set_angle_speed = 0.
                set_speed_forward = 0.
                set_speed_right = set_speed_right + self.desc_step_speed if set_speed_right > 0. else self.desc_step_speed

            set_angle_speed = self._clamp(set_angle_speed, -self.speed_angle_k, self.speed_angle_k)
            set_speed_forward = self._clamp(set_speed_forward, -self.speed_k, self.speed_k)
            set_speed_right = self._clamp(set_speed_right, -self.speed_k, self.speed_k)

        self.simple_move(set_angle_speed, set_speed_forward, set_speed_right)
        step_reward, terminated, truncated = self.calc_step_reward(set_angle_speed, set_speed_forward, set_speed_right)

        # вознаграждение за правильное решение
        self.reward += step_reward

        vec_dist = self.target_point - self.position
        observation = np.empty(self.observation_space_local.shape, dtype=np.float32)
        observation[0] = self._output_norm(self.locate_k, vec_dist.x)   # dXt, dYt, Dt, Azt, 
        observation[1] = self._output_norm(self.locate_k, vec_dist.y)
        observation[2] = self._output_norm(self.locate_k, vec_dist.rho)
        observation[3] = self._output_norm(self.angle_k, self._get_course_bind(vec_dist.phi))
        
        for obs in observation: 
            if math.isnan(obs) or math.isinf(obs):
                print(self.target_point, self.position, self.dir_view )

        info = self._get_info()
        self.tick += 1

        if self.is_pygame == True:
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

        # угол на который можно развернутся за шаг
        angle_step = set_angle_speed * self.time_step

        final_time = self.time_model + self.time_tick
        while self.time_model < final_time:
            self.time_model += self.time_step

            # доворачивает текщий вектор движения в сторону заданного на угол за шаг
            self.dir_view = self.dir_view.rotateZ(angle_step)
            dir_view_right = vector.obj(x=-self.dir_view.y,y=self.dir_view.x)
 
            # скорость вперед - назад
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_sign = 1.
            if self._equivalent(set_speed_forward, new_speed.x) == True:
                speed_sign = 0
            elif set_speed_forward < new_speed.x:
                speed_sign = -1.
            # меняем скорость
            new_speed.x += speed_sign * self.speed_acc * self.time_step
            new_speed.x = self._clamp(new_speed.x, -self.speed_k, self.speed_k)

            # скорость вправо - влево
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_sign = 1.
            if self._equivalent(set_speed_right, new_speed.y) == True:
                speed_sign = 0
            elif set_speed_right < new_speed.y:
                speed_sign = -1.
            # меняем скорость
            new_speed.y += speed_sign * self.speed_acc * self.time_step
            new_speed.y = self._clamp(new_speed.y, -self.speed_k, self.speed_k)

            if new_speed.rho > self.speed_k:
                new_speed /= new_speed.rho
                new_speed *= self.speed_k

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

        t_point = vector.obj(x=self.position.x, y = self.position.y)
        self.tick_point.append(t_point)

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

            t_action = np.array([teach_set_angle_speed, teach_set_speed, 0])

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
    


    def calc_step_reward(self, set_angle_speed:float, set_speed_forward:float, set_speed_right:float):

        if self.time_model_max < self.time_model:  # время вышло
            return -1, False, True


        #ушел за границу зоны
        if self.position.x < -self.zero.x or self.position.y < -self.zero.y or self.position.x > self.zero.x or self.position.y > self.zero.y:
            return -2, False, True

        step_reward = -0.001

        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        vec_to_finish = self.target_point - self.position
        
        # угол на который надо довернуть
        delta_angle = self._get_course_bind(vec_to_finish.phi)
        
        sign_angle = 1 if delta_angle >= 0 else -1
        sign_set_angle_speed = 1 if set_angle_speed >= 0 else -1
        # команда на разворот в противоположную сторону - шрафуем
        if set_angle_speed != 0. and sign_angle != sign_set_angle_speed:
            return -0.002, False, False

        # расстояние до целевой точки
        dist_to_finish = vec_to_finish.rho
    
        if dist_to_finish < self.finish_radius: # пришли
            return 3, False, True

        # единичный вектор на целевую точку        
        dir_to_target_point = vec_to_finish / dist_to_finish

        # векторное расстояние пройденное за шаг моделирования
        way_vec = self.position - self.position_prev

        # расстояние на которой приблизились к целевой точке за шаг моделирования
        way_step = way_vec @ dir_to_target_point

        # косинус угла меду направлением движения и направление на целевую точку
        cos_way = 0
        if way_vec.rho > 0.0001:
            cos_way = way_step / way_vec.rho

        if way_step > 1 and set_speed_forward > set_speed_right:
            # поощряем сближение - дожно быть меньше штрафа за прочие действия (чтобы не зацикливал траекторию)
            step_reward = 0.01
            # угол смещения направления на целевую точку
            angle_rate = math.pi / 18.  # 10 градусов
            if dist_to_finish < 50:
                angle_rate *= 0.5   # 5 градусов
            elif dist_to_finish < 10:
                angle_rate *= 0.2   # 2 градуса
            # чем ближе тем точнее надо наводиться
            if cos_way > math.cos(angle_rate):
                step_reward = 0.05

            if self.speed_bind.rho > 0.5: 
                step_reward *= self.speed_bind.rho
        else:
            if set_speed_forward < 0:
                step_reward = -0.002

        return step_reward, False, False


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
        corner2 = corner2.rotateZ(self.dir_view.phi)
        corner3 = corner3.rotateZ(self.dir_view.phi)
        corner2 += triangle
        corner3 += triangle

        draw.polygon(
            xy=(
                (triangle.x, triangle.y),
                (corner2.x, corner2.y),
                (corner3.x, corner3.y)
            ), fill=red, outline=red
        )

        if self.speed_local.rho > 0:
            dir_move = self.speed_local/self.speed_local.rho
            srt = triangle + 10 * dir_move
            end = srt + 10 * self.speed_local
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
        pygame.draw.circle(self.screen, blue, (target_circle.x, target_circle.y), 10)
    
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

        triangle_points = [(triangle.x, triangle.y), (corner2.x, corner2.y), (corner3.x, corner3.y)]
        pygame.draw.polygon(self.screen, red, triangle_points)

        if self.speed_local.rho > 0:
            dir_move = self.speed_local/self.speed_local.rho
            srt = pos + 10 * dir_move
            end = srt + self.speed_local
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
    
    def _get_course_bind(self, phi:float)->float:
        delta_angle = phi - self.dir_view.phi
        sign = -1 if delta_angle >= 0 else 1
        if abs(delta_angle) > math.pi:
            delta_angle = sign * (2 * math.pi - abs(delta_angle))
        return delta_angle
    
    def _equivalent(self, value1:float, value2:float, epsilon:float = 0.0001)->bool:
        return True if abs(value1 - value2) < epsilon else False
        
        
    def _min_then_max(self, v1:float, v2:float):
        if v1 < v2:
            return v1, v2
        else:
            return v2, v1
        
    def _sign_or_zero(self, value:float)->float:
        if self._equivalent(value, 0.) == True:
            return 0.
        elif value < 0.:
            return -1
        else:
            return 1