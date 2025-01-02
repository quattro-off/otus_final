
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw

from env_move_sector import HumanMoveSectorAction
from env_move_sector_v2 import HumanMoveSectorActionV2
from env_move_sector_v3 import HumanMoveSectorActionV3
from env_move_ray_v3 import HumanMoveRayActionV3
from env_move_fast_sector import HumanMoveFastSectorAction

spiral = [[0,0],
                  [1,0],
                  [1,1],
                  [0,1],
                  [-1,1],
                  [-1,0],
                  [-1,-1],
                  [0,-1],
                  [1,-1],
                  [2,-1],
                  [2,0],
                  [2,1],
                  [2,2],
                  [1,2],
                  [0,2],
                  [-1,2],
                  [-2,2],
                  [-2,1],
                  [-2,0],
                  [-2,-1],
                  [-2,-2],
                  [-1,-2],
                  [0,-2],
                  [1,-2],
                  [2,-2],
                  [3,-2]
                  ]


def gen_walls(id:int ,wall_count:int, objects:dict, exclude_point:list):
    
    dl = int(len(spiral) / wall_count)
    dl = 1 if dl == 0 else dl
    for i in range(wall_count):
        side_min = 20
        side_max = 50
        corners = random.randint(3,6)
        senter = vector.obj(x=0,y=0)
        to_close = random.randint(0,1)
        ii = (i * dl) % len(spiral)
        xy = spiral[ii]
        for _ in range(10):
            senter.x=random.randint(100 * xy[0] - 50, 100 * xy[0] + 50)
            senter.y=random.randint(100 * xy[1] - 50, 100 * xy[1] + 50)
            
            is_correct = True

            for point in exclude_point:
                to_point = point - senter
                if to_point.rho < side_max:
                    is_correct = False
            if is_correct:
                break
        
        if senter.rho == 0:
            print('WARNING!!! not found place for wall')
            continue
        corner_point = []
        corner_angle = []
        for _ in range(corners):
            corner_angle.append(random.randint(0,360))
        corner_angle.sort()
        for a in corner_angle:
            point = vector.obj(phi = math.radians(a), rho=random.randint(side_min, side_max))
            corner_point.append(point + senter)
        for i in range(corners-1):
            wall_data = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': corner_point[i], 'c2': corner_point[i+1]}
            objects.append(wall_data)
            id += 1
        
        if to_close == 1:
            wall_data = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': corner_point[0], 'c2': corner_point[corners-1]}
            objects.append(wall_data)
            id += 1

def gen_builds(id:int, border_count:int, wall_count:int, objects:dict, place_size:vector, exclude_point:list):
    
    side_min = 20
    side_max = 20

    to_target = vector.obj(x=1,y=0)
    right_target = vector.obj(x=0,y=1)
    start_create = vector.obj(x=-250,y=0)

    for b in range(border_count):
        sign = -1 if (b+1)%2 else 1
        start_create += to_target * side_max * 2 + sign * right_target * side_max 

        for i in range(wall_count):
            radius = random.randint(side_min, side_max)
            start_angle = random.randint(0,360)
            first_side = random.randint(0,90)

            sign = -1 if (i+1)%2 else 1
            d_n = start_create + right_target * sign * (i * side_max * 1.5 )

            is_correct = True

            for point in exclude_point:
                to_point = point - d_n
                if to_point.rho < side_max:
                    is_correct = False
            if not is_correct:
                continue

            corner_1 = d_n + vector.obj(phi = math.radians(start_angle), rho=radius)
            corner_2 = d_n + vector.obj(phi = math.radians(start_angle + first_side), rho=radius)
            corner_3 = d_n + vector.obj(phi = math.radians(start_angle + 180), rho=radius)
            corner_4 = d_n + vector.obj(phi = math.radians(start_angle + 180 + first_side), rho=radius)

            if d_n.x > -place_size.x and d_n.x < place_size.x and d_n.y > -place_size.y and d_n.y < place_size.y:
                wall_data_1 = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': corner_1, 'c2': corner_2}
                objects.append(wall_data_1)
                id += 1
                wall_data_2 = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': corner_2, 'c2': corner_3}
                objects.append(wall_data_2)
                id += 1
                wall_data_3 = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': corner_3, 'c2': corner_4}
                objects.append(wall_data_3)
                id += 1
                wall_data_4 = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': corner_4, 'c2': corner_1}
                objects.append(wall_data_4)
                id += 1

def gen_lines(id:int, border_count:int, wall_count:int, objects:dict, place_size:vector, line_angle: int):
    
    side_min = 20
    side_max = 20

    to_target = vector.obj(x=1,y=0)
    right_target = vector.obj(x=0,y=1)
    start_create = vector.obj(x=-200,y=0)

    if line_angle < 0:
        k_shift = 1.5
    else:
        k_shift = 0.5 + math.sin(math.radians(line_angle))

    for b in range(border_count):

        for i in range(wall_count):
            radius = random.randint(side_min, side_max)
            if line_angle < 0:
                start_angle = random.randint(0,360)
            else:
                start_angle = line_angle #180#random.randint(0,360)

            sign = -1 if (i+1)%2 else 1
            d_n = start_create + right_target * sign * (i * side_max * k_shift)

            corner_1 = d_n + vector.obj(phi = math.radians(start_angle), rho=radius)
            corner_2 = d_n + vector.obj(phi = math.radians(start_angle + 180), rho=radius)

            if (d_n.x > -place_size.x and d_n.x < place_size.x) or (d_n.y > -place_size.y and d_n.y < place_size.y):
                wall_data_1 = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': corner_1, 'c2': corner_2}
                objects.append(wall_data_1)
                id += 1

        sign = -1 if (b+1)%2 else 1
        start_create += to_target * side_max * 4.5 + sign * right_target * side_max * k_shift

class HumanMoveAroundWallAction(HumanMoveSectorAction):


    def name(self):
        return 'Wall_' + super().name() 
    
    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 render_time_step: float=1.,
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 tree_count: int = 0,
                 wall_border:int = 5,
                 options = None
                ):
        
        self.wall_border_count = wall_border

        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         render_time_step = render_time_step,
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         object_ignore = object_ignore,
                         tree_count = tree_count,
                         options=options,
                         )

    def reset(self, seed=None):


        start_observation, info = super().reset(seed=seed)

        self.target_point.x = 250
        self.target_point.y = 250
        self.position.x = -250
        self.position.y = -250
        vec_dist = self.target_point - self.position
        self.time_model_optimum = vec_dist.rho / self.max_speed
        self.time_model_max = 3 * self.time_model_optimum
        vec_dist.rho -= self.finish_radius + 1
        start_observation[0] = self._output_norm(self.locate_k, vec_dist.x)   # dXt, dYt, Dt, Azt, 
        start_observation[1] = self._output_norm(self.locate_k, vec_dist.y)
        start_observation[2] = self._output_norm(self.locate_k, vec_dist.rho, True)
        start_observation[3] = self._output_norm(self.angle_k, self._get_course_bind(vec_dist.phi))

        if self.object_ignore == False:
            id = self.objects[-1]['id'] + 1
            gen_walls(id,self.wall_border_count, self.objects, [self.position, self.target_point])


            #delta_y = 50
            #delta_x = 50
            #for i in range(4,5,2):
            #    rnd_y = random.randint(-200, 200)
            #    wall_1 = vector.obj(x= -200 + i*delta_x, y =-300)
            #    wall_2 = vector.obj(x= -200 + i*delta_x, y =rnd_y)
            #    wall_data = {'id': id + i, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #    self.objects.append(wall_data)
            #    wall_1 = vector.obj(x= -200 + i*delta_x, y =rnd_y + delta_y)
            #    wall_2 = vector.obj(x= -200 + i*delta_x, y =300)
            #    wall_data = {'id': id + i + 1, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #    self.objects.append(wall_data)

            #id = self.objects[-1]['id'] + 1
            #wall_1 = vector.obj(x= -295, y =-299)
            #wall_2 = vector.obj(x= -295, y =299)
            #wall_data = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)
            #wall_1 = vector.obj(x= -299, y =-295)
            #wall_2 = vector.obj(x= 299, y =-295)
            #wall_data = {'id': id+1, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)
            #wall_1 = vector.obj(x= 295, y =-299)
            #wall_2 = vector.obj(x= 295, y =299)
            #wall_data = {'id': id+2, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)
            #wall_1 = vector.obj(x= -299, y =295)
            #wall_2 = vector.obj(x= 299, y =295)
            #wall_data = {'id': id+3, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)

            
        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info



class HumanMoveAroundWallActionV2(HumanMoveSectorActionV2):

    def name(self):
        return 'Wall_' + super().name() 
    
    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 render_time_step: float=1.,
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 tree_count: int = 0,
                 wall_border:int = 5,
                 options = None
                ):
        
        self.wall_border_count = wall_border

        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         render_time_step = render_time_step,
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         object_ignore = object_ignore,
                         tree_count = tree_count,
                         options=options,
                         )

    def reset(self, seed=None):


        start_observation, info = super().reset(seed=seed)

        self.target_point.x = 250
        self.target_point.y = 250
        self.position.x = -250
        self.position.y = -250
        vec_dist = self.target_point - self.position
        self.time_model_optimum = vec_dist.rho / self.max_speed
        self.time_model_max = 3 * self.time_model_optimum
        vec_dist.rho -= self.finish_radius + 1
        start_observation[0] = self._output_norm(self.locate_k, vec_dist.x)   # dXt, dYt, Dt, Azt, 
        start_observation[1] = self._output_norm(self.locate_k, vec_dist.y)
        start_observation[2] = self._output_norm(self.locate_k, vec_dist.rho, True)
        start_observation[3] = self._output_norm(self.angle_k, self._get_course_bind(vec_dist.phi))


        if self.object_ignore == False:
            id = self.objects[-1]['id'] + 1
            gen_walls(id,self.wall_border_count, self.objects, [self.position, self.target_point])
            
        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info



class HumanMoveAroundWallActionV3(HumanMoveSectorActionV3):

    def name(self):
        return 'Wall_' + super().name() 
    
    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 render_time_step: float=1.,
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 object_locate:str = 'random',
                 tree_count: int = 0,
                 wall_border:int = 5,
                 line_angle:int = 0,
                 options = None
                ):
        
        self.line_angle = line_angle
        self.wall_border_count = wall_border

        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         render_time_step = render_time_step,
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         object_ignore = object_ignore,
                         object_locate = object_locate,
                         tree_count = tree_count,
                         options=options,
                         )

    def reset(self, seed=None):


        start_observation, info = super().reset(seed=seed)


        if self.object_ignore == False:
            id = self.objects[-1]['id'] + 1

            wall_1 = vector.obj(x= -299, y =-295)
            wall_2 = vector.obj(x= 299, y =-295)
            wall_data = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            id += 1
            self.objects.append(wall_data)
            wall_1 = vector.obj(x= -299, y = 295)
            wall_2 = vector.obj(x= 295, y = 295)
            wall_data = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            id += 1
            self.objects.append(wall_data)

            if 'build' in self.object_locate:
                gen_builds(id, self.border_count, self.wall_border_count, self.objects, self.zero)
            elif 'wall' in self.object_locate:
                gen_lines(id, self.border_count, self.wall_border_count, self.objects, self.zero, self.line_angle)
            else:
                gen_walls(id,self.wall_border_count, self.objects, [self.dynamic.get_position(), self.target_point])


            
            
        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info
        

class HumanMoveRayAroundWallActionV3(HumanMoveRayActionV3):

    def name(self):
        return 'Wall_' + super().name() 
    
    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 render_time_step: float=1.,
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 object_locate:str = 'random',
                 tree_count: int = 0,
                 wall_border:int = 5,
                 line_angle:int = 0,
                 options = None
                ):
        
        self.line_angle = line_angle
        self.wall_border_count = wall_border

        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         render_time_step = render_time_step,
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         object_ignore = object_ignore,
                         object_locate = object_locate,
                         tree_count = tree_count,
                         options=options,
                         )

    def reset(self, seed=None):


        start_observation, info = super().reset(seed=seed)


        if self.object_ignore == False:
            id = self.objects[-1]['id'] + 1
            
           
            if 'build' in self.object_locate:
                gen_builds(id, self.border_count, self.wall_border_count, self.objects, self.zero, [self.dynamic.get_position(), self.target_point])
            elif 'wall' in self.object_locate:
                gen_lines(id, self.border_count, self.wall_border_count, self.objects, self.zero, self.line_angle)
            else:
                gen_walls(id,self.wall_border_count, self.objects, [self.position, self.target_point])

            #self.target_point.x = -250
            #self.target_point.y = 0
            
            
        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info
        

class HumanMoveFastAroundWallAction(HumanMoveFastSectorAction):

    
    #wall_count = 20
    #wall_length = vector.obj(x=10,y=20)

    def name(self):
        return 'Wall_' + super().name() 
    
    def __init__(self, 
                 continuous: bool = True,
                 seed: int=42, 
                 render_mode: str=None, 
                 render_time_step: float=1.,
                 observation_render: bool = False, 
                 target_point_rand:bool = False,
                 object_ignore:bool = False,
                 tree_count: int = 0,
                 wall_border:int = 5,
                 options = None
                ):
        
        self.wall_border_count = wall_border

        super().__init__(continuous = continuous, 
                         seed = seed, 
                         render_mode = render_mode, 
                         render_time_step = render_time_step,
                         observation_render = observation_render,
                         target_point_rand = target_point_rand,
                         object_ignore = object_ignore,
                         tree_count = tree_count,
                         options=options,
                         )

    def reset(self, seed=None):


        start_observation, info = super().reset(seed=seed)

        self.target_point.x = 250
        self.target_point.y = 0
        self.position.x = -250
        self.position.y = -0
        vec_dist = self.target_point - self.position
        self.time_model_optimum = vec_dist.rho / self.max_speed
        self.time_model_max = 3 * self.time_model_optimum
        vec_dist.rho -= self.finish_radius + 1
        start_observation[0] = self._output_norm(self.locate_k, vec_dist.x)   # dXt, dYt, Dt, Azt, 
        start_observation[1] = self._output_norm(self.locate_k, vec_dist.y)
        start_observation[2] = self._output_norm(self.locate_k, vec_dist.rho, True)
        start_observation[3] = self._output_norm(self.angle_k, self._get_course_bind(vec_dist.phi))

        if self.object_ignore == False:
            id = self.objects[-1]['id'] + 1
            gen_walls(id,self.wall_border_count, self.objects, [self.position, self.target_point])

            #delta_y = 50
            #delta_x = 50
            #for i in range(0,1,2):
            #    rnd_y = random.randint(-250, -200)
            #    wall_1 = vector.obj(x= -200 + i*delta_x, y =-300)
            #    wall_2 = vector.obj(x= -100 + i*delta_x, y =rnd_y)
            #    wall_data = {'id': id + i, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #    self.objects.append(wall_data)
            #    wall_1 = vector.obj(x= -200 , y =-0)
            #    wall_2 = vector.obj(x= 0, y =20)
            #    wall_data = {'id': id + i + 1, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #    self.objects.append(wall_data)
            
            #id = self.objects[-1]['id'] + 1

            #wall_1 = vector.obj(x= -295, y =-299)
            #wall_2 = vector.obj(x= -295, y =299)
            #wall_data = {'id': id, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)
            #wall_1 = vector.obj(x= -299, y =-295)
            #wall_2 = vector.obj(x= 299, y =-295)
            #wall_data = {'id': id+1, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)
            #wall_1 = vector.obj(x= 295, y =-299)
            #wall_2 = vector.obj(x= 295, y =299)
            #wall_data = {'id': id+2, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)
            #wall_1 = vector.obj(x= -299, y =295)
            #wall_2 = vector.obj(x= 299, y =295)
            #wall_data = {'id': id+3, 'type': 'wall', 'vis': False, 'col': (0,1,0), 'c1': wall_1, 'c2': wall_2}
            #self.objects.append(wall_data)
            
        if self.observation_render == True:
            return self._get_render(True), info
        else:
            return start_observation, info
