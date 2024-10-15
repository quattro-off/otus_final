
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw
from typing import Dict
import sys
import os

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../common'))

from env_simple_move import HumanMoveSimpleAction
from sector_view import SectorView

# dXt (-300,300), dXt (-300,300) 
# Dt t (0, 300)
# Azt t (-pi, pi)
# dXt, dYt, Dt, Azt, 

# setor info Agent Bind location
# density (0, 100%)
# distance (0, 300)
# type (0,10) 0-free 1 - target_point, 2-obstacle, 3-damage area

# NOrmalize (0,1)

observation_space = Box(low=np.array([0, 0, 0, 0, # dXt, dYt, Dt, Azt, 
                                      0, 0, 0,  #  1    345  15 Deg -  density, distance, type
                                      0, 0, 0,  #  2     15  45 Deg 
                                      0, 0, 0,  #  3     45  75 Deg 
                                      0, 0, 0,  #  4     75 105 Deg
                                      0, 0, 0,  #  5    105 135 Deg
                                      0, 0, 0,  #  6    135 165 Deg 
                                      0, 0, 0,  #  7    165 195 Deg 
                                      0, 0, 0,  #  8    195 225 Deg 
                                      0, 0, 0,  #  9    225 255 Deg 
                                      0, 0, 0,  # 10    255 285 Deg 
                                      0, 0, 0,  # 11    285 315 Deg
                                      0, 0, 0,  # 12    315 345 Deg
                                        ], dtype=np.float32), 
                        high=np.array([1, 1, 1, 1,                                      
                                       1, 1, 1, #  1    345  15 Deg -  density, distance, type
                                       1, 1, 1, #  2     15  45 Deg 
                                       1, 1, 1, #  3     45  75 Deg 
                                       1, 1, 1, #  4     75 105 Deg
                                       1, 1, 1, #  5    105 135 Deg
                                       1, 1, 1, #  6    135 165 Deg 
                                       1, 1, 1, #  7    165 195 Deg 
                                       1, 1, 1, #  8    195 225 Deg 
                                       1, 1, 1, #  9    225 255 Deg 
                                       1, 1, 1, # 10    255 285 Deg 
                                       1, 1, 1, # 11    285 315 Deg
                                       1, 1, 1, # 12    315 345 Deg
                                        ], dtype=np.float32), shape=(40,), dtype=np.float32)


#                  speed forward(-3.6,3.6) m/s course speed(-pi/3,pi/3) rad/s, speed right(-3.6,3.6) m/s
action_space = Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), shape=(3,), dtype=np.float32)

action_space_d = Discrete(7, seed=42)

observation_space_rend = Box(0, 255, (600, 600, 1), dtype=np.uint8)

class HumanMoveSectorAction(HumanMoveSimpleAction):

    #observation
    damage_radius_k = 10
    tree_radius_k = 0.5
    dist_k = 150

    #obstacles
    tree_radius = 6

    def name(self):
        add_rnd = '_RanTP' if self.target_point_rand == True else ''
        add_cnn = '_CNN' if self.observation_render == True else ''
        add_ignore = '_IgnoreObst' if self.object_ignore == True else ''
        return 'MoveSector' + add_rnd + add_cnn + add_ignore

    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 tree_count: int = 20,
                 options = None
                ):
        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         options=options,
                         )
        
        self.trees_count = tree_count
        
        if observation_render == True:
            self.observation_space = observation_space_rend
            self.observation_space_local = observation_space
        else:
            self.observation_space = observation_space
            self.observation_space_local = observation_space
        
        self.reward_object_use = 0 # 0-ignore, 1-append, 2-rewrite
        self.reward_object_stop = 0 # штраф за сторкновение с препятствием
        self.reward_object_move = 0 # вознаграждение за попытку обойти

        self.object_ignore = object_ignore

        self.objects = []

        self.sectors = {}
        #Az,Ev,D,T   
        self.sectors[0  ] = SectorView(sector_phi=0, max_dist=150)
        self.sectors[30 ] = SectorView(sector_phi=30, max_dist=150)
        self.sectors[60 ] = SectorView(sector_phi=60, max_dist=150)
        self.sectors[90 ] = SectorView(sector_phi=90, max_dist=150)
        self.sectors[120] = SectorView(sector_phi=120, max_dist=150)
        self.sectors[150] = SectorView(sector_phi=150, max_dist=150)
        self.sectors[180] = SectorView(sector_phi=180, max_dist=150)
        self.sectors[210] = SectorView(sector_phi=210, max_dist=150)
        self.sectors[240] = SectorView(sector_phi=240, max_dist=150)
        self.sectors[270] = SectorView(sector_phi=270, max_dist=150)
        self.sectors[300] = SectorView(sector_phi=300, max_dist=150)
        self.sectors[330] = SectorView(sector_phi=330, max_dist=150)
        
    def get_rewards(self)->Dict[str, float]:
        step_rews = super().get_rewards()
        step_rews['object_stop'] = self.reward_object_stop
        step_rews['object_move'] = self.reward_object_move
        return step_rews


    def reset(self, seed=None):

        start_observation, info = super().reset(seed=seed)

        self.reward_object_use = 0
        self.reward_object_stop = 0
        self.reward_object_move = 0 

        for sector in self.sectors.values():
            sector.clear()

        self.objects.clear()

        start_data = {'id': 0, 'type': 'start', 'pos':self.position, 'radius': 6, 'vis': False, 'col': (0,0,0)}
        self.objects.append(start_data)

        # target point
        final_data = {'id': 1, 'type': 'final', 'pos':self.target_point, 'radius': 6, 'vis': False, 'col': (0,0,255)}
        self._check_object_on_sector(final_data)
        self.objects.append(final_data)

        if self.object_ignore == False:

            _min_pos_x = 0.5 * (self.position.x + self.target_point.x) - 100
            _max_pos_x = 0.5 * (self.position.x + self.target_point.x) + 100
            _min_pos_y = 0.5 * (self.position.y + self.target_point.y) - 100
            _max_pos_y = 0.5 * (self.position.y + self.target_point.y) + 100

            # trees
            for i in range(self.trees_count):
                id = i + 2

            
                d_n = vector.obj(x=0,y=0)
                is_valide = False
                while is_valide == False:
                    d_n = vector.obj(x = random.randint(int(_min_pos_x), int(_max_pos_x)),
                                    y = random.randint(int(_min_pos_y), int(_max_pos_y))
                                    )
                    d_pos = self.position - d_n
                    d_tp = self.target_point - d_n
                    if d_pos.rho > self.tree_radius + 1 and d_tp.rho > self.tree_radius + 1:
                        is_valide = True

                tree_data = {'id': id, 'type': 'tree', 'pos':d_n, 'radius': self.tree_radius, 'vis': False, 'col': (0,255,0)}
                self._check_object_on_sector(tree_data)
                self.objects.append(tree_data)

        i = 4
        for sector in self.sectors.values():
            s_o = sector.get_observation()
            start_observation[i]   = s_o[0]
            start_observation[i+1] = s_o[1]
            start_observation[i+2] = s_o[2]
            i = i + 3


        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info
        

    def _check_object_on_sector(self, object):

        if object['id'] == 0:
            return
        
        if object['type'] != 'tree' and object['type'] != 'wall':
            return

        object['vis'] = False
        object['col'] = (0,255,0)

        b_set_to_view = False
        dist_left = 0
        dist_right = 0
        bind_phi_left_side = 0
        bind_phi_right_side = 0

        if object['type'] == 'tree':
            vec_dir = object['pos'] - self.position

            if vec_dir.rho < self.dist_k:

                d_angle_view = math.atan2(object['radius'], vec_dir.rho)
                bind_phi_left_side = math.degrees(self._get_course_bind(vec_dir.phi - d_angle_view, True))
                bind_phi_right_side = math.degrees(self._get_course_bind(vec_dir.phi + d_angle_view, True))
                dist_left = vec_dir.rho - object['radius']
                dist_right = vec_dir.rho - object['radius']

                b_set_to_view = True

        elif object['type'] == 'wall':
            vec_dir_1 = object['c1'] - self.position
            vec_dir_2 = object['c2'] - self.position


            #if vec_dir_1.rho < self.dist_k or vec_dir_2.rho < self.dist_k:
            angle = vec_dir_1.deltaphi(vec_dir_2)
            if angle > 0:
                vec_dir_1, vec_dir_2 = vec_dir_2, vec_dir_1

            bind_phi_left_side = math.degrees(self._get_course_bind(vec_dir_1.phi, True))
            bind_phi_right_side = math.degrees(self._get_course_bind(vec_dir_2.phi, True))
            dist_left = vec_dir_1.rho
            dist_right = vec_dir_2.rho

            b_set_to_view = True

        if b_set_to_view:
            for sector in self.sectors.values():
                if sector.add_obstacle(object['id'], object['type'], bind_phi_left_side, dist_left, bind_phi_right_side, dist_right):
                    object['col'] = (255,255,2)


    def step(self, action):

        if self.observation_render == True:
            observation, step_reward, terminated, truncated, info = super().step(action)

            return self._get_render(True), step_reward, terminated, truncated, info
        else:

            observation, step_reward, terminated, truncated, info = super().step(action)

            if self.object_ignore == False:

                for sector in self.sectors.values():
                        sector.clear()

                for obj in self.objects:
                        self._check_object_on_sector(obj)

            i = 4
            for sector in self.sectors.values():
                s_o = sector.get_observation()
                observation[i]   = s_o[0]
                observation[i+1] = s_o[1]
                observation[i+2] = s_o[2]
                i = i + 3

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
            
                obs = self._clamp(obs,0.,1.)
                i += 1

            return observation, step_reward, terminated, truncated, info
 

    def simple_move(self, set_angle_dir, set_speed_forward, set_speed_right):
        
        self._prev_reward_box(set_angle_dir, set_speed_forward, set_speed_right)

        super().simple_move(set_angle_dir, set_speed_forward, set_speed_right)


    def _prev_reward_box(self, set_angle_dir, set_speed_forward, set_speed_right):

        self.reward_object_use = 0
        self.reward_object_stop = 0
        self.reward_object_move = 0 

        if self.object_ignore == False:

            direct_along_sector = 0             # сектор на целевую точку
            move_along_sector = 0               # сектор движения
            density_move_along_sector = 0       # плотность препятствий в секторе движения
            num_along_sector = -1
            max_density_sector = 0              # сектор с максимальной плотностью
            min_density_sector = 0              # сектор с максимальной плотностью
            max_density = 0.                    # максимальная плотность из всех секторов
            min_density = 1.                    # минимальная плотность из всех секторов
            count_sector_with_70_density = 0.   # количество секторов с большой плотностью
            keys = []

            set_move_local = vector.obj(x=set_speed_forward, y=set_speed_right)
            #set_move_local = vector.obj(x=self.speed_bind.x, y=self.speed_bind.y)

            # вектор на целевую точку в связанной ССК
            direct_to_point = self.target_point - self.position
            view_phi_bind = direct_to_point.deltaphi(self.dir_view)
            dir_view_bind = vector.obj(phi=view_phi_bind,rho=1)


            for key, sector in self.sectors.items():

                keys.append(key)

                density = sector.get_density()

                # данные об препятствиях
                if density > max_density:
                    max_density = density
                    max_density_sector = key
                if density < min_density:
                    min_density = density
                    min_density_sector = key

                if density > 0.7:
                    count_sector_with_70_density += 1

                # сектор направления на целевую точку                
                is_direct_along_sector, _ = sector.is_vector_in_sector(dir_view_bind)
                if is_direct_along_sector:
                    direct_along_sector = key

                # движемся вдоль сектора
                is_move_along_sector, _ = sector.is_vector_in_sector(set_move_local)

                # сектор чист - проверяем следующий
                if self._equivalent(density, 0.):
                    continue

                if is_move_along_sector == False:
                    continue


                # сектор движения
                move_along_sector = key
                density_move_along_sector = density

                
                # данные о препятствии в сенкторе
                is_find, obj_id, vec_dist_bind = sector.get_near_obstacle()

                # в секторе движения штаф зависит отплотности препятствий и расстоянии до препятствий
                if is_find == False:
                    print(f'WARNING!!! In sector({key}) with density({density}) is not trees')

                object = None
                for o in self.objects:
                    if o['id'] == obj_id:
                        object = o
                        break

                if object == None:
                    print(f'ERROR!!! Object {obj_id} is not found')

                dist = vec_dist_bind.rho


                self.reward_object_stop = -0.0001 * density 
                    
                if dist > 2*self.speed_k:

                    self.reward_object_use = 1
                    koef_dist = 1. if self.dist_k < dist else 10 * (2. - dist/self.dist_k)
                    self.reward_object_stop *= koef_dist

                else:
                        
                    self.reward_object_use = 1
                    self.reward_object_stop *= 10

                    cos_to_obj = (vec_dist_bind @ set_move_local) / (vec_dist_bind.rho * set_move_local.rho)

                    if cos_to_obj > 0:#по направлению движения есть препятствие
                        self.reward_object_use = 2
                        self.reward_object_stop -= 0.001 * density
                        if dist < 1: 
                            self.reward_object_stop -= 0.001
                            set_speed_forward = 0.

                            if dist < 0.5:
                                self.reward_object_stop -= 0.001
                                self.speed_local.x = 0
                                self.speed_local.y = 0
                                self.speed_bind.x = 0
                                self.speed_bind.y = 0

                            object['col'] = (255,0,255)

            if self.reward_object_use == 1:
                # добавить вознагражения если происходит обход препятсвий
                s_len = len(self.sectors)
                s_half = int(0.5*s_len)
                
                d_s = 180/s_half
                num = int(direct_along_sector/d_s)
                keys_left_right = []  #сортировка секторов начиная от сектора на целевуюточку по одному вправо и влево
                keys_left_right.append(keys[num])
                for i in range(1,s_half+1):
                    n_s_next = num + i
                    if n_s_next >= s_len:
                        n_s_next = s_len - n_s_next
                    keys_left_right.append(keys[n_s_next])
                    if len(keys) == len(keys_left_right):
                        break
                    n_s_prev = num - i
                    if n_s_prev < 0:
                        n_s_prev = s_len + n_s_prev
                    keys_left_right.append(keys[n_s_prev])
                    if len(keys) == len(keys_left_right):
                        break

                if len(keys) != len(keys_left_right):
                    print(f'WARNING!!! error sort sectors key')


                num_move = 0
                for k_s in keys_left_right:
                    # когда обходим препятсвие награда задается за сочетание минимизации угла отклонения от цели
                    #  и минимизации запоненности сектора
                    if move_along_sector == k_s:
                        sector = self.sectors[k_s]
                        density = sector.get_density()
                        self.reward_object_move = 0.0003 * (s_len - num_move) * (1-density)
                        break
                    num_move += 1

            # целевая точка ближе препятствия - игнорируем препятствие
            max_d_sector = self.sectors[direct_along_sector]
            is_find, obj_id, vec_dist_bind = max_d_sector.get_near_obstacle()
            if is_find:
                if vec_dist_bind.rho > direct_to_point.rho:
                    #print('INFO!!! target point is near')
                    self.reward_object_use = 0
                    self.reward_object_stop = 0
                    self.reward_object_move = 0 
            else:
                self.reward_object_use = 0
                self.reward_object_stop = 0
                self.reward_object_move = 0                 

    def calc_step_reward_box(self, set_angle_speed, set_speed, set_speed_right):

        step_reward, terminated, truncated =  super().calc_step_reward_box(set_angle_speed, set_speed, set_speed_right)

        #if self.time_model_max < self.time_model:  # время вышло
        #    return -5, False, True

        #ушел за границу зоны
        #if self.position.x < -self.zero.x or self.position.y < -self.zero.y or self.position.x > self.zero.x or self.position.y > self.zero.y:
        #    return -2, False, True

        # вектор от текущей позиции до целевой точки (целевая точка всегла в центре)
        #vec_to_finish = self.target_point - self.position
        # пришли
        #if vec_to_finish.rho < self.finish_radius:
        #    return 5, True, False    

        #step_reward = -0.001


        if self.reward_object_use == 1:#append
            self.view_step_reward = 0.
            self.angle_step_reward = 0.
            step_reward += self.reward_object_stop + self.reward_object_move
        elif self.reward_object_use == 2:#rewrite
            self.speed_step_reward = 0.
            self.view_step_reward = 0.
            self.stoper_step_reward = 0.
            step_reward = self.reward_object_stop + self.reward_object_move

        return step_reward, terminated, truncated
        #return step_reward, False, False
    
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
        draw.ellipse((
                    target_circle.x-self.finish_radius, 
                    target_circle.y-self.finish_radius,
                    target_circle.x+self.finish_radius,  
                    target_circle.y+self.finish_radius), 
            fill=blue, outline=blue)
        
        if self.object_ignore == False:
            i = 0
            for sector in self.sectors.values():
                col = (0,20+20*i,200)
                i += 1
                is_find, obj_id, vec_to = sector.get_near_obstacle()
                if is_find:
                    vec_to = vec_to.rotateZ(self.dir_view.phi)
                    v1 = self.zero + self.position
                    v2 = self.zero + vec_to + self.position
                    draw.line(xy=((v1.x,v1.y), (v2.x,v2.y)),fill=col, width=2)

            for object in self.objects:
                if object['type'] == 'tree':
                    col = object['col']
                    dem = self.zero + object['pos']
                    rad = object['radius']
                    draw.ellipse((dem.x-rad, dem.y-rad,dem.x+rad, dem.y+rad), fill=col, outline=col)
                elif object['type'] == 'wall':
                    col = object['col']
                    c1 = self.zero + object['c1']
                    c2 = self.zero + object['c2']
                    draw.line(xy=((c1.x,c1.y), (c2.x,c2.y)),fill=col, width=4)             

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

        if self.speed_local.rho > 0:
            dir_move = self.speed_local/self.speed_local.rho
            srt = pos + 10 * dir_move
            end = srt + 10 * self.speed_local
            draw.line(xy=((srt.x,srt.y), (end.x,end.y)),fill=red)

            set_speed = vector.obj(phi=self.dir_view.phi + self.speed_bind_set.phi,rho=self.speed_bind_set.rho)
            dir_move = set_speed/set_speed.rho
            srt = pos + 10 * dir_move
            end = srt + 4 * set_speed
            draw.line(xy=((srt.x,srt.y), (end.x,end.y)),fill=green)


        # область видимости объектов  вокруг
        area = self.zero + self.position
        draw.ellipse((
                    area.x-self.dist_k, 
                    area.y-self.dist_k,
                    area.x+self.dist_k,  
                    area.y+self.dist_k), 
            outline=white, width=2)
        
        start_phi = 15
        delta_phi = 30
        for i in range(len(self.sectors)):
            lt0 = vector.obj(phi=self.dir_view.phi - math.radians(start_phi + i * delta_phi), rho=0.8*self.dist_k)
            lt1 = vector.obj(phi=self.dir_view.phi - math.radians(start_phi + i * delta_phi), rho=self.dist_k)
            srt = area + lt0
            end = area + lt1
            draw.line(xy=((srt.x,srt.y), (end.x,end.y)),fill=red)

        im_np = np.asarray(im)
        return im_np

    def human_render(self):
        
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)
        pure =(255,0,255)

        self.screen.fill(black)

        if self.object_ignore == False:
            i = 0
            for sector in self.sectors.values():
                col = (0,20+20*i,200)
                i += 1
                is_find, obj_id, vec_to = sector.get_near_obstacle()
                if is_find:
                    vec_to = vec_to.rotateZ(self.dir_view.phi)
                    v1 = self.zero + self.position
                    v2 = self.zero + vec_to + self.position
                    pygame.draw.line(self.screen, col, [v1.x,v1.y], [v2.x,v2.y], 2)

            for object in self.objects:
                if object['type'] == 'tree':
                    col = object['col']
                    dem = self.zero + object['pos']
                    rad = object['radius']
                    pygame.draw.circle(self.screen, col, (dem.x, dem.y), rad)
                elif object['type'] == 'wall':
                    col = object['col']
                    c1 = self.zero + object['c1']
                    c2 = self.zero + object['c2']
                    pygame.draw.line(self.screen, col, [c1.x,c1.y], [c2.x,c2.y],4)

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

        if self.speed_local.rho > 0:
            dir_move = self.speed_local/self.speed_local.rho
            srt = pos + 10 * dir_move
            end = srt + 4 * self.speed_local
            pygame.draw.aaline(self.screen, red, [srt.x,srt.y], [end.x,end.y])

            set_speed = vector.obj(phi=self.dir_view.phi + self.speed_bind_set.phi,rho=self.speed_bind_set.rho)
            dir_move = set_speed/set_speed.rho
            srt = pos + 10 * dir_move
            end = srt + 4 * set_speed
            pygame.draw.aaline(self.screen, green, [srt.x,srt.y], [end.x,end.y])



        # область видимости объектов  вокруг
        area = self.zero + self.position
        pygame.draw.circle(self.screen, white, (area.x, area.y), self.dist_k, 2)

        start_phi = 15
        delta_phi = 30
        for i in range(len(self.sectors)):
            lt0 = vector.obj(phi=self.dir_view.phi - math.radians(start_phi + i * delta_phi), rho=0.8*self.dist_k)
            lt1 = vector.obj(phi=self.dir_view.phi - math.radians(start_phi + i * delta_phi), rho=self.dist_k)
            srt = area + lt0
            end = area + lt1
            pygame.draw.aaline(self.screen, red, [srt.x,srt.y], [end.x,end.y])

        pygame.display.update()

        # устанавливаем частоту обновления экрана
        pygame.time.Clock().tick(60)

