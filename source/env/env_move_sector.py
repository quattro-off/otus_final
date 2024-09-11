
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw

from env_simple_move import HumanMoveSimpleAction


# dXt (-300,300), dXt (-300,300) 
# Dt t (0, 300)
# Azt t (-pi, pi)
# dXt, dYt, Dt, Azt, 

# setor info Agent Bind location
# azimut (-pi, pi)
# evalation (-pi/2, pi/2)
# dist (0, 300)
# type (0,10) 0-free 1 - target_point, 2-obstacle, 3-damage area

# NOrmalize (0,1)

observation_space = Box(low=np.array([0, 0, 0, 0, # dXt, dYt, Dt, Azt, 
                                      0, 0, 0, 0,  #  1    345  15 Deg - Az,Ev,D,T   
                                      0, 0, 0, 0,  #  2     15  45 Deg 
                                      0, 0, 0, 0,  #  3     45  75 Deg 
                                      0, 0, 0, 0,  #  4     75 105 Deg
                                      0, 0, 0, 0,  #  5    105 135 Deg
                                      0, 0, 0, 0,  #  6    135 165 Deg 
                                      0, 0, 0, 0,  #  7    165 195 Deg 
                                      0, 0, 0, 0,  #  8    195 225 Deg 
                                      0, 0, 0, 0,  #  9    225 255 Deg 
                                      0, 0, 0, 0,  # 10    255 285 Deg 
                                      0, 0, 0, 0,  # 11    285 315 Deg
                                      0, 0, 0, 0,  # 12    315 345 Deg
                                        ], dtype=np.float32), 
                        high=np.array([1, 1, 1, 1,
                                      1, 1, 1, 1,  #  1    345  15 Deg - Az,Ev,D,T   
                                      1, 1, 1, 1,  #  2     15  45 Deg 
                                      1, 1, 1, 1,  #  3     45  75 Deg 
                                      1, 1, 1, 1,  #  4     75 105 Deg
                                      1, 1, 1, 1,  #  5    105 135 Deg
                                      1, 1, 1, 1,  #  6    135 165 Deg 
                                      1, 1, 1, 1,  #  7    165 195 Deg 
                                      1, 1, 1, 1,  #  8    195 225 Deg 
                                      1, 1, 1, 1,  #  9    225 255 Deg 
                                      1, 1, 1, 1,  # 10    255 285 Deg 
                                      1, 1, 1, 1,  # 11    285 315 Deg
                                      1, 1, 1, 1,  # 12    315 345 Deg
                                        ], dtype=np.float32), shape=(52,), dtype=np.float32)


#                 course speed(-pi/3,pi/3) rad/s, speed forward(-3.6,3.6) m/s speed right(-3.6,3.6) m/s
action_space = Box(low=np.array([-1, -1, -1], dtype=np.float32), high=np.array([1, 1, 1], dtype=np.float32), shape=(3,), dtype=np.float32)

action_space_d = Discrete(7, seed=42)

observation_space_rend = Box(0, 255, (600, 600, 1), dtype=np.uint8)

class HumanMoveSectorAction(HumanMoveSimpleAction):
    """непрерывные состояния, непрерывные действия"""

    object_ignore = False

    #observation
    damage_radius_k = 10
    tree_radius_k = 0.5
    dist_k = 50

    tree_radius = 6
    trees_count = 60
    objects = []

    sectors = {#Az,Ev,D,T   
          0: [0,0.,0.,1.,0.],
         30: [0,0.,0.,1.,0.],
         60: [0,0.,0.,1.,0.],
         90: [0,0.,0.,1.,0.],
        120: [0,0.,0.,1.,0.],
        150: [0,0.,0.,1.,0.],
        180: [0,0.,0.,1.,0.],
        210: [0,0.,0.,1.,0.],
        240: [0,0.,0.,1.,0.],
        270: [0,0.,0.,1.,0.],
        300: [0,0.,0.,1.,0.],
        330: [0,0.,0.,1.,0.],
    }



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
                 object_ignore:bool = False
                ):
        super().__init__(continuous, seed, render_mode, observation_render,target_point_rand)
        if observation_render == True:
            self.observation_space = observation_space_rend
            self.observation_space_local = observation_space
        else:
            self.observation_space = observation_space
            self.observation_space_local = observation_space
        
        self.object_ignore = object_ignore

    def reset(self, seed=None):

        start_observation, info = super().reset(seed=seed)

        for k in self.sectors.keys():
            self.sectors[k]  = [0,0.,0.,1.,0.]

        self.objects.clear()

        start_data = {'id': 0, 'type': 'start', 'pos':self.position, 'radius': 6, 'vis': False, 'col': (0,0,0)}
        self.objects.append(start_data)

        # target point
        final_data = {'id': 1, 'type': 'final', 'pos':self.target_point, 'radius': 6, 'vis': False, 'col': (0,0,255)}
        self._check_object_on_sector(final_data)
        self.objects.append(final_data)

        # trees
        for i in range(self.trees_count):
            id = i + 2

            _min_pos_x = 0.5 * (self.position.x + self.target_point.x) - 100
            _max_pos_x = 0.5 * (self.position.x + self.target_point.x) + 100
            _min_pos_y = 0.5 * (self.position.y + self.target_point.y) - 100
            _max_pos_y = 0.5 * (self.position.y + self.target_point.y) + 100
            
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
        for k in self.sectors.keys():
            start_observation[i]   = self.sectors[k][1]
            start_observation[i+1] = self.sectors[k][2]
            start_observation[i+2] = self.sectors[k][3]
            start_observation[i+3] = self.sectors[k][4]
            i = i + 4


        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info
        

    def _check_object_on_sector(self, object):

        object['vis'] = False
        object['col'] = (0,255,0)
        vec_dir = object['pos'] - self.position

        if vec_dir.rho < self.dist_k:

            delta_angle = vec_dir.phi - self.dir_view.phi

            sign = -1 if delta_angle >= 0 else 1
            if abs(delta_angle) > math.pi:
                delta_angle = math.degrees( sign * (2 * math.pi - abs(delta_angle)) )
            else:
                delta_angle = math.degrees(delta_angle)
            if delta_angle < 0:
                delta_angle = 360 + delta_angle
            num_s = 30 * int(delta_angle // 30)
            if num_s == 360:
                num_s = 0
            d_num = delta_angle % 30
            if d_num  > 15:
                num_s = 0 if num_s == 330 else num_s + 30
                d_num -= 15

            if num_s > 330 or num_s < 0:
                print(f'Error: Course: {self.dir_view.phi} Tree Phi: {vec_dir.phi} Res Angle: {delta_angle} Sector: {num_s}')

            norm_dist = self._output_norm(self.dist_k, vec_dir.rho, True)
            if norm_dist < self.sectors[num_s][3]:
                if self.sectors[num_s][0] > 0:
                    self.objects[self.sectors[num_s][0]]['vis'] = False
                    self.objects[self.sectors[num_s][0]]['col'] = (0,255,0)

                object['vis'] = True

                #Az,Ev,D,T  
                self.sectors[num_s][0] = object['id']
                self.sectors[num_s][1] = self._output_norm(30,d_num,True)
                self.sectors[num_s][2] = 0.
                self.sectors[num_s][3] = norm_dist
                if object['type'] == 'final':
                    object['col'] = (0,0,255)
                    self.sectors[num_s][4] = 0.
                elif object['type'] == 'tree':
                    object['col'] = (255,0,255)
                    self.sectors[num_s][4] = 0.1


      

    def step(self, action):

        if self.observation_render == True:
            observation, step_reward, terminated, truncated, info = super().step(action)

            return self._get_render(True), step_reward, terminated, truncated, info
        else:

            observation, step_reward, terminated, truncated, info = super().step(action)

            for k in self.sectors.keys():
                self.sectors[k]  = [0,0.,0.,1.,0.]

            for tree in self.objects:
                self._check_object_on_sector(tree)

            i = 4
            for k in self.sectors.keys():
                observation[i]   = self.sectors[k][1]
                observation[i+1] = self.sectors[k][2]
                observation[i+2] = self.sectors[k][3]
                observation[i+3] = self.sectors[k][4]
                i = i + 4
    
            i = 0
            for obs in observation:
                if math.isnan(obs):
                    print(f'abs n:{i} is NAN')
                if math.isinf(obs):
                    print(f'abs n:{i} is INF')
                if i >3:
                    if obs < -0.01 or obs > 1.01 :
                        print(f'abs n:{i} out of NORM : {obs}')
                    
                obs = self._clamp(obs,0.,1.)
                i += 1

            return observation, step_reward, terminated, truncated, info
 

    def simple_move(self, set_angle_dir, set_speed_forward, set_speed_right):

        if self.object_ignore == False:

            move_forward_local = self._sign_or_zero(set_speed_forward) * self.dir_view
            move_right_local = self._sign_or_zero(set_speed_right) * vector.obj(x=-self.dir_view.y, y=self.dir_view.x)

            for sector in self.sectors.values():
                if sector[0] == 0:
                    continue

                object = self.objects[sector[0]]
                if object['type'] == 'tree':
                    vec_dist = object['pos'] - self.position
                    dist = vec_dist.rho
                    if dist < object['radius'] + self.speed_local.rho:
                        vec_dist /= vec_dist.rho

                        id = -1
                        if move_forward_local.rho > 0:
                            cos_forward_to_tree = vec_dist @ move_forward_local
                            if cos_forward_to_tree > 0: #по направлению движения есть препятствие
                                set_speed_forward = 0.
                                id = object['id']

                        if move_right_local.rho > 0:
                            cos_right_to_tree = vec_dist @ move_right_local
                            if cos_right_to_tree > 0: # по направлению бокового движения есть препятствие
                                set_speed_right = 0.
                                id = object['id']

                        if id > 0:
                            #print(f'Info: Stoper Object: tree_{id} dist {dist}')
                            break

        super().simple_move(set_angle_dir, set_speed_forward, set_speed_right)


    def calc_step_reward(self, set_angle_speed, set_speed, set_speed_right):

        step_reward, terminated, truncated =  super().calc_step_reward(set_angle_speed, set_speed, set_speed_right)

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

        for object in self.objects:
            if object['type'] == 'tree':
                col = object['col']
                dem = self.zero + object['pos']
                rad = object['radius']
                draw.ellipse((dem.x-rad, dem.y-rad,dem.x+rad, dem.y+rad), fill=col, outline=col)

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

        im_np = np.asarray(im)
        return im_np

    def human_render(self):
        
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)

        self.screen.fill(black)

        # область видимости объектов  вокруг
        area = self.zero + self.position
        pygame.draw.circle(self.screen, white, (area.x, area.y), self.dist_k, 2)

        #lt = vector.obj(phi=self.dir_view.phi - math.radians(15), rho=self.dist_k)
        #srt = area + lt
        #end = area - lt
        #pygame.draw.aaline(self.screen, white, [srt.x,srt.y], [end.x,end.y])
        #rt = vector.obj(phi=self.dir_view.phi + math.radians(15), rho=self.dist_k)
        #srt = area + rt
        #end = area - rt
        #pygame.draw.aaline(self.screen, white, [srt.x,srt.y], [end.x,end.y])


        for object in self.objects:
            if object['type'] == 'tree':
                col = object['col']
                dem = self.zero + object['pos']
                rad = object['radius']
                pygame.draw.circle(self.screen, col, (dem.x, dem.y), rad)

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
            end = srt + 4 * self.speed_local
            pygame.draw.aaline(self.screen, red, [srt.x,srt.y], [end.x,end.y])

        pygame.display.update()

        # устанавливаем частоту обновления экрана
        pygame.time.Clock().tick(60)

