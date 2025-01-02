from gymnasium.spaces import Box
import numpy as np
import math
import vector 


observation_space = Box(low=np.array([-1, 0, #  1     0  Deg -  azimut, distance
                                      -1, 0, #  2     30 Deg 
                                      -1, 0, #  3     60 Deg 
                                      -1, 0, #  4     90 Deg
                                      -1, 0, #  5    120 Deg
                                      -1, 0, #  6    150 Deg 
                                      -1, 0, #  7    -30 Deg 
                                      -1, 0, #  8    -60 Deg 
                                      -1, 0, #  9    -90 Deg 
                                      -1, 0, # 10   -120 Deg 
                                      -1, 0, # 11   -150 Deg
                                      -1, 0, # 12   -180 Deg
                                        ], dtype=np.float32), 
                        high=np.array([1, 1, #  1     0  Deg -  azimut, distance
                                       1, 1, #  2     30 Deg 
                                       1, 1, #  3     60 Deg 
                                       1, 1, #  4     90 Deg
                                       1, 1, #  5    120 Deg
                                       1, 1, #  6    150 Deg 
                                       1, 1, #  7    -30 Deg 
                                       1, 1, #  8    -60 Deg 
                                       1, 1, #  9    -90 Deg 
                                       1, 1, # 10   -120 Deg 
                                       1, 1, # 11   -150 Deg
                                       1, 1, # 12   -180 Deg
                                        ], dtype=np.float32), shape=(24,), dtype=np.float32)

class RayView():


    def __init__(self, 
                sector_phi:float = 0,
                max_dist: float = 50
                ):
        self.sector_phi = sector_phi
        self.direct = vector.obj(rho=1,phi=math.radians(self.sector_phi) )
        self.point = vector.obj(x=0.,y=0.)
        self.max_dist = max_dist
        self.current_dist = max_dist
        self.observation = [float(self.direct.phi/math.pi),1.]# azimut, distance
        self.obstacles = {}# 'type', 'bind_phi_left', 'bind_phi_right', 'vec_dist'}
        self.reward = 0

    def set_reward(self, reward:float):
        self.reward = reward
    def get_reward(self):
        return self.reward

    def get_observation(self):
        
        if len(self.obstacles) > 0:
            near = next(iter(self.obstacles))
            obst = self.obstacles[int(near)]
            self.observation[1] = float(obst['vec_dist'].rho/self.max_dist)
        else:
            self.observation[1] = 1.
        return self.observation
    
    def get_ray(self, absolute:bool=False):
        if len(self.obstacles) > 0:
            near = next(iter(self.obstacles))
            obst = self.obstacles[int(near)]
            if absolute:
                return vector.obj(phi=obst['vec_dist'].phi, rho=obst['vec_dist'].rho)
            else:
                return vector.obj(phi=obst['vec_dist'].phi, rho=obst['vec_dist'].rho/self.max_dist)
        else:
            if absolute:
                return vector.obj(phi=self.direct.phi, rho=self.max_dist)
            else:
                return vector.obj(phi=self.direct.phi, rho=self.direct.rho)
    
    def get_distance(self, absolute:bool=False):
        return self.max_dist * self.observation[1] if absolute else self.observation[1]
    
    def get_distances(self):
        return self.max_dist * self.observation[1], self.observation[1]

    def size(self):
        return len(self.obstacles)

    def clear(self):
        self.observation[1] = 1.
        self.obstacles.clear()

    def get_obstacles(self):
        obst = []
        for id in self.obstacles.keys():
            obst.append(id)
        return obst
    
    def get_near_obstacle(self, type:str = None):
        id_near = -1
        b_find = False
        min_dist = vector.obj(x=1000,y=0)
        # 'type', 'vec_dist'}
        for id, obst in self.obstacles.items():
            if type != None and type != obst['type']:
                continue

            if min_dist.rho > obst['vec_dist'].rho:
                min_dist.rho = obst['vec_dist'].rho
                min_dist.phi = obst['vec_dist'].phi
                b_find = True
                id_near = id

        return b_find, id_near, min_dist
    
    def add_obstacle(self, id: int, type:str, phi_left: float, dist_left: float, phi_right: float, dist_right: float):

        vec_left = vector.obj(rho=dist_left, phi=math.radians(phi_left),z=0)
        vec_right = vector.obj(rho=dist_right, phi=math.radians(phi_right),z=0)

        b_cross, cross_point = self.vec_to_cross_point(vec_left, vec_right, vector.obj(phi=self.direct.phi, rho=self.max_dist,z=0))

        if b_cross and cross_point.rho < self.max_dist:
           
            self.obstacles[id] = {'type': type, 'vec_dist': cross_point}
            def min_dist(data):
                return data[1]['vec_dist'].rho
            vs = sorted(self.obstacles.items(), key=min_dist)
            self.obstacles = dict(vs)

        return b_cross

    def vec_to_cross_point(self, v11, v12, v22):

        cross_point = vector.obj(x=0,y=0)

        v21 = vector.obj(x=self.point.x,y=0,z=self.point.y)

        vector_for_crossed = v12 - v11
        vector_cross = v22 - v21
        
        prod1 = vector_cross.cross(v11-v21)
        prod2 = vector_cross.cross(v12-v21)

        if self._sign(prod1.z) == self._sign(prod2.z) :
            #print(f'WARNING: vectors is not crossed')
            return False, cross_point
        
        if prod1.z == 0 or prod2.z == 0: # Отсекаем также и пограничные случаи
            #print(f'WARNING: vectors is parallel')
            return False, cross_point

        prod1 = vector_for_crossed.cross(v21-v11)
        prod2 = vector_for_crossed.cross(v22-v11)

        if self._sign(prod1.z) == self._sign(prod2.z) :
            #print(f'WARNING: vectors is not crossed')
            return False, cross_point

        if prod1.z == 0 or prod2.z == 0: # Отсекаем также и пограничные случаи
            #print(f'WARNING: vectors is parallel')
            return False, cross_point

        cross_point.x = vector_cross.x * abs(prod1.z)/abs(prod2.z-prod1.z)
        cross_point.y = vector_cross.y * abs(prod1.z)/abs(prod2.z-prod1.z)
        
        return True, cross_point
        
    def _sign(self, value:float)->float:
        if value < 0.:
            return -1
        else:
            return 1

class RaySensor():
    
    def __init__(self, 
                dist_k:float = 50,
                ):

        self.dist_k = dist_k

        self.rays = {}
        #Az,Ev,D,T   
        self.rays[0  ] = RayView(sector_phi=0,   max_dist=self.dist_k)
        self.rays[30 ] = RayView(sector_phi=30,  max_dist=self.dist_k)
        self.rays[60 ] = RayView(sector_phi=60,  max_dist=self.dist_k)
        self.rays[90 ] = RayView(sector_phi=90,  max_dist=self.dist_k)
        self.rays[120] = RayView(sector_phi=120, max_dist=self.dist_k)
        self.rays[150] = RayView(sector_phi=150, max_dist=self.dist_k)
        self.rays[180] = RayView(sector_phi=-30, max_dist=self.dist_k)
        self.rays[210] = RayView(sector_phi=-60, max_dist=self.dist_k)
        self.rays[240] = RayView(sector_phi=-90, max_dist=self.dist_k)
        self.rays[270] = RayView(sector_phi=-120, max_dist=self.dist_k)
        self.rays[300] = RayView(sector_phi=-150, max_dist=self.dist_k)
        self.rays[330] = RayView(sector_phi=-180, max_dist=self.dist_k)

        self.observation = np.zeros(24, dtype=np.float32)

        self.move_ray = RayView(sector_phi=0,   max_dist=self.dist_k)
        self.move_ray_left = RayView(sector_phi=0,   max_dist=self.dist_k)
        self.move_ray_right = RayView(sector_phi=0,   max_dist=self.dist_k)
    
    def get_observation_box(self):
        return observation_space

    def clear(self):
        self.move_ray.clear()
        self.move_ray_left.clear()
        self.move_ray_right.clear()
        for ray in self.rays.values():
            ray.clear()

    def check_object_on_sector(self, object, position, direction):

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
            vec_dir = object['pos'] - position

            if vec_dir.rho < self.dist_k:

                d_angle_view = math.atan2(object['radius'], vec_dir.rho)
                shift_dir = vector.obj(phi=vec_dir.phi - d_angle_view,rho=1)
                bind_phi_left_side = math.degrees(shift_dir.deltaphi(direction))
                shift_dir = vector.obj(phi=vec_dir.phi + d_angle_view,rho=1)
                bind_phi_right_side = math.degrees(shift_dir.deltaphi(direction))
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
            if vec_dir_1.rho < self.dist_k or vec_dir_2.rho < self.dist_k or (dist_to_line < self.dist_k and obj_line.rho > self.dist_k):
            
                angle = vec_dir_1.deltaphi(vec_dir_2)
                if angle > 0:
                    vec_dir_1, vec_dir_2 = vec_dir_2, vec_dir_1

                bind_phi_left_side = math.degrees(vec_dir_1.deltaphi(direction))
                bind_phi_right_side = math.degrees(vec_dir_2.deltaphi(direction))
                dist_left = vec_dir_1.rho
                dist_right = vec_dir_2.rho

                b_set_to_view = True

        if b_set_to_view:

            self.move_ray.add_obstacle(object['id'], object['type'], bind_phi_left_side, dist_left, bind_phi_right_side, dist_right)

            for ray in self.rays.values():
                if ray.add_obstacle(object['id'], object['type'], bind_phi_left_side, dist_left, bind_phi_right_side, dist_right):
                    object['col'] = (255,255,0)


    def step(self, objects:dict, position:vector, direction:vector):
        self.clear()

        for obj in objects:
            self.check_object_on_sector(obj, position, direction)

    def get_observation(self):
        i = 0
        for ray in self.rays.values():
            s_o = ray.get_observation()
            self.observation[i]   = s_o[0]
            self.observation[i+1] = s_o[1]
            i = i + 2
        return self.observation
    
    def get_figures(self, position:vector, direction:vector):
        figures = []
        for ray in self.rays.values():
            vec_ray = ray.get_ray(True)
            vec_ray = vec_ray.rotateZ(direction.phi)
            v1 = position
            v2 = position + vec_ray
            col = (0,150,150)
            figures.append({'figure':'line', 'start': (v1.x,v1.y), 'end':(v2.x,v2.y), 'in_color':col, 'width':2})

        return figures

    def set_move_ray_direction(self, phi:float):
        self.move_ray.sector_phi = phi
        self.move_ray.direct.phi = phi

    def get_move_ray(self):
        return self.move_ray.get_ray(True)