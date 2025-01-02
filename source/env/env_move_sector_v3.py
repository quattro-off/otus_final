
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import re
from typing import Dict
import sys
import os

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

from env_move_simple_v3 import MoveSimpleActionV3
from common.sector_view import SectorView
import common.functions as f


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

class HumanMoveSectorActionV3(MoveSimpleActionV3):

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
        
        self.object_locate = object_locate
        self.trees_count = tree_count
        
        num = re.findall('(\d+)', self.object_locate)
        if len(num) == 0:
            self.border_count = 1
        else:
            self.border_count = int(f.f_clamp(int(num[-1]), 1, 20))
        
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

        self.left_or_right = True
        
    def get_rewards(self)->Dict[str, float]:
        step_rews = super().get_rewards()
        step_rews['object_stop'] = self.reward_object_stop
        step_rews['object_move'] = self.reward_object_move
        return step_rews


    def reset(self, seed=None):

        start_observation, info = super().reset(seed=seed)

        position = self.dynamic.get_position()

        self.reward_object_use = 0
        self.reward_object_stop = 0
        self.reward_object_move = 0 

        for sector in self.sectors.values():
            sector.clear()

        self.objects.clear()

        start_data = {'id': 0, 'type': 'start', 'pos':position, 'radius': 6, 'vis': False, 'col': (0,0,0)}
        self.objects.append(start_data)

        # target point
        final_data = {'id': 1, 'type': 'final', 'pos':self.target_point, 'radius': 6, 'vis': False, 'col': (0,0,255)}
        self._check_object_on_sector(final_data)
        self.objects.append(final_data)

        if self.object_ignore == False:
            self.target_point.x = 250
            self.target_point.y = 0
            position.x = -250
            position.y = 0
            to_target = self.target_point - position

            self.time_model_optimum = self.dynamic.get_min_time_moving(self.target_point)
            self.time_model_max = 3 * self.time_model_optimum
            


            if 'wall' in self.object_locate or 'build' in self.object_locate:


                right_target = vector.obj(x=-to_target.y,y=to_target.x)
                right_target /= right_target.rho
                start_create = position + 0.3*to_target
                to_target /= to_target.rho

                for b in range(self.border_count):
                    sign = -1 if (b+1)%2 else 1
                    start_create += to_target * self.tree_radius * 6 + sign * right_target * self.tree_radius * 4

                    for i in range(self.trees_count):
                        id = i + 2
                        
                        sign = -1 if (i+1)%2 else 1

                        d_n = start_create + right_target * sign * (i * self.tree_radius * 4 )
                        if d_n.x > -self.zero.x and d_n.x < self.zero.x and d_n.y > -self.zero.y and d_n.y < self.zero.y:
                            tree_data = {'id': id, 'type': 'tree', 'pos':d_n, 'radius': self.tree_radius, 'vis': False, 'col': (0,255,0)}
                            self._check_object_on_sector(tree_data)
                            self.objects.append(tree_data)


            else:
                # trees
                for i in range(self.trees_count):
                    id = i + 2

                    d_n = vector.obj(x=0,y=0)
                    is_valide = False
                    while is_valide == False:
                        d_n = vector.obj(x = random.randint(int(self.zero.x), int(self.zero.x)),
                                        y = random.randint(int(self.zero.y), int(self.zero.y))
                                        )
                        d_pos = position - d_n
                        d_tp = self.target_point - d_n
                        if d_pos.rho > self.tree_radius + 1 and d_tp.rho > self.tree_radius + 1:
                            is_valide = True

                    tree_data = {'id': id, 'type': 'tree', 'pos':d_n, 'radius': self.tree_radius, 'vis': False, 'col': (0,255,0)}
                    self._check_object_on_sector(tree_data)
                    self.objects.append(tree_data)

            position.y = random.randint(-250,250)
            self.target_point.y = random.randint(-250,250)

            self.dynamic.set_position(position)

        i = 4
        for sector in self.sectors.values():
            s_o = sector.get_observation()
            start_observation[i]   = s_o[0]
            start_observation[i+1] = s_o[1]
            start_observation[i+2] = s_o[2]
            i = i + 3


        self.left_or_right = bool(random.randint(0, 1))

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

        position = self.dynamic.get_position()

        if object['type'] == 'tree':
            vec_dir = object['pos'] - position

            if vec_dir.rho < self.dist_k:

                d_angle_view = math.atan2(object['radius'], vec_dir.rho)
                shift_dir = vector.obj(phi=vec_dir.phi - d_angle_view,rho=1)
                bind_phi_left_side = math.degrees(self.dynamic.get_delta_angle_direction(shift_dir))
                shift_dir = vector.obj(phi=vec_dir.phi + d_angle_view,rho=1)
                bind_phi_right_side = math.degrees(self.dynamic.get_delta_angle_direction(shift_dir))
                dist_left = vec_dir.rho - object['radius']
                dist_right = vec_dir.rho - object['radius']

                b_set_to_view = True

        elif object['type'] == 'wall':

            obj_line =  object['c1'] - object['c2']
            to_line_vec = vector.obj(x=-obj_line.y, y=obj_line.x)
            to_line_vec.rho = 1

            vec_dir_1 = object['c1'] - position
            vec_dir_2 = object['c2'] - position

            dist_to_line = to_line_vec @ vec_dir_1
            if vec_dir_1.rho < self.dist_k or vec_dir_2.rho < self.dist_k or dist_to_line < self.dist_k:

                angle = vec_dir_1.deltaphi(vec_dir_2)
                if angle > 0:
                    vec_dir_1, vec_dir_2 = vec_dir_2, vec_dir_1

                bind_phi_left_side = math.degrees(self.dynamic.get_delta_angle_direction(vec_dir_1))
                bind_phi_right_side = math.degrees(self.dynamic.get_delta_angle_direction(vec_dir_2))
                dist_left = vec_dir_1.rho
                dist_right = vec_dir_2.rho

                b_set_to_view = True

        if b_set_to_view:
            for sector in self.sectors.values():
                if sector.add_obstacle(object['id'], object['type'], bind_phi_left_side, dist_left, bind_phi_right_side, dist_right):
                    object['col'] = (255,255,0)


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
            
                obs = f.f_clamp(obs,0.,1.)
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

            # убираем препятсвие
            self.vec_dist_to_obctacle_bind.x = 100
            self.vec_dist_to_obctacle_bind.y = 100

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

            #set_move_local = vector.obj(x=set_speed_forward, y=set_speed_right)
            set_move_local = vector.obj(x=self.speed_bind.x, y=self.speed_bind.y)

            # вектор на целевую точку в связанной ССК
            direct_to_point = self.target_point - self.position
            view_phi_bind = direct_to_point.deltaphi(self.dir_view)
            dir_view_bind = vector.obj(phi=view_phi_bind,rho=1)
            move_d_phi_bind = direct_to_point.deltaphi(self.speed_local)

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

                # препятсвие для для расчета столкновения
                self.vec_dist_to_obctacle_bind.x = vec_dist_bind.x
                self.vec_dist_to_obctacle_bind.y = vec_dist_bind.y

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


                self.reward_object_use = 1 if density > 0.5 else 0
                self.reward_object_stop = -0.0001 * density 
                    
                if dist < 2 *  self.speed_k:

                    self.reward_object_use = 1
                        
                    koef_dist = 1. if self.dist_k < dist else 10 * (1. - dist/self.dist_k)
                    self.reward_object_stop *= koef_dist

                    cos_to_obj = (vec_dist_bind @ set_move_local) / (vec_dist_bind.rho * set_move_local.rho)

                    if cos_to_obj > 0:#по направлению движения есть препятствие
                        self.reward_object_use = 2
                        self.reward_object_stop *= 1.4
                        if dist < 1: 
                            self.reward_object_stop *= 1.4

                            if dist < 0.5:
                                self.reward_object_stop *= 1.4


                            object['col'] = (255,0,255)
                        else:
                            object['col'] = (255,255,0)

            if True:#self.reward_object_use > 0:
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
                    n_s_prev = num - i
                    if n_s_prev < 0:
                        n_s_prev = s_len + n_s_prev

                    if self.left_or_right:
                        keys_left_right.append(keys[n_s_prev])
                        if len(keys) == len(keys_left_right):
                            break
                        keys_left_right.append(keys[n_s_next])
                        if len(keys) == len(keys_left_right):
                            break
                    else:
                        keys_left_right.append(keys[n_s_next])
                        if len(keys) == len(keys_left_right):
                            break

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
                        k_dist = sector.get_distance()
                        #self.reward_object_move = 0.00002 * (s_len - num_move + 1) * (0.5 - density)
                        
                        self.reward_object_move = 0.001 * ((1.1*math.pi - abs(move_d_phi_bind))/math.pi) * (0.2 - density) 
                        self.reward_object_move += -0.0001*(1./(math.sqrt(k_dist)+0.1) - 1)

                        #if self.reward_object_move < 0:
                        #    self.reward_object_move *= 3
                        break
                    num_move += 1

                #print(f'move sector: {move_along_sector}, rStop {self.reward_object_stop}, rMove {self.reward_object_move}')

            # целевая точка ближе препятствия - игнорируем препятствие
            #if direct_along_sector == move_along_sector:
            #    max_d_sector = self.sectors[direct_along_sector]
            #    is_find, obj_id, vec_dist_bind = max_d_sector.get_near_obstacle()
            #    if is_find:
            #        if vec_dist_bind.rho > direct_to_point.rho:
            #            #print('INFO!!! target point is near')
            #            self.reward_object_use = 0
            #            self.reward_object_stop = 0
            #            self.reward_object_move = 0 
              

    def calc_step_reward_box(self, set_angle_speed, set_speed, set_speed_right):

        step_reward, terminated, truncated =  super().calc_step_reward_box(set_angle_speed, set_speed, set_speed_right)

        return step_reward + self.reward_object_move, terminated, truncated
    

    def render(self):
        figures = self._get_figures(False)
        return self._get_render(figures,False)
    
    def human_render(self, figures:list=None):
        figures = self._get_figures(False)
        return super().human_render(figures)
        


    def _get_figures(self,to_observation:bool):

        black = (0, 0, 0)
        white = (255, 255, 255)
        red= (255,0,0)

        position = self.dynamic.get_position()
        dir_view = self.dynamic.get_direction()

        # позиция агента
        pos = self.zero + position
        
        figures1 = super()._get_figures(to_observation)

        figures = []
        if self.object_ignore == False:

            # область видимости объектов  вокруг
            figures.append({'figure':'ellipse', 'senter': (pos.x,pos.y), 'radius':self.dist_k, 'in_color':black, 'out_color':white, 'width':1})

            i = 0
            for sector in self.sectors.values():
                col = (0,20+20*i,200)
                i += 1
                is_find, obj_id, vec_to = sector.get_near_obstacle()
                if is_find:
                    vec_to = vec_to.rotateZ(dir_view.phi)
                    v1 = pos
                    v2 = pos + vec_to
                    figures.append({'figure':'line', 'start': (v1.x,v1.y), 'end':(v2.x,v2.y), 'in_color':col, 'width':2})

                end_sector_phi = sector.sector_phi + sector.delta_phi
                lt0 = vector.obj(phi=dir_view.phi + math.radians(end_sector_phi), rho=0.8*self.dist_k)
                lt1 = vector.obj(phi=dir_view.phi + math.radians(end_sector_phi), rho=self.dist_k)
                srt = pos + lt0
                end = pos + lt1
                figures.append({'figure':'line', 'start': (srt.x,srt.y), 'end':(end.x,end.y), 'in_color':red, 'width':1})

                density = vector.obj(phi=dir_view.phi + math.radians(sector.sector_phi), rho=sector.get_distance(True) + 10)
                density += pos
                rad = sector.get_density()*10
                figures.append({'figure':'ellipse', 'senter': (density.x,density.y), 'radius':rad, 'in_color':white, 'out_color':white, 'width':0})

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