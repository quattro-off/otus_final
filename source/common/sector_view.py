
import numpy as np
import math
import vector 

class SectorView():


    def __init__(self, 
                sector_phi:float = 0,
                delta_phi: float = 15,
                max_dist: float = 50
                ):
        self.sector_phi = sector_phi
        self.delta_phi = delta_phi
        self.direct = vector.obj(rho=1,phi=math.radians(self.sector_phi) )
        self.max_dist = max_dist
        self.observation = [0.,1.,0.]# density, distance, type
        self.obstacles = {}# 'type', 'bind_phi_left', 'bind_phi_right', 'vec_dist'}
        self.mem_obst_id = []

    def is_vector_in_sector(self, vec:vector):
        cos_with_sector = (vec @ self.direct)/(vec.rho * self.direct.rho)
        return (True if cos_with_sector > math.cos(math.radians(self.delta_phi)) else False), cos_with_sector

    def get_observation(self):
        return self.observation
    
    def get_density(self):
        return self.observation[0]
    
    def size(self):
        return len(self.obstacles)

    def clear(self):
        self.observation = [0.,1.,0.]
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
        # 'type', 'bind_phi_left', 'bind_phi_right', 'vec_dist'}
        for id, obst in self.obstacles.items():
            if type != None and type != obst['type']:
                continue

            dist = obst['vec_dist'].rho

            if min_dist.rho > dist:
                min_dist.rho = dist
                min_dist.phi = obst['vec_dist'].phi
                b_find = True
                id_near = id

        return b_find, id_near, min_dist
    
    def add_obstacle(self, id: int, type:str, phi_left: float, dist_left: float, phi_right: float, dist_right: float):

        vec_left = vector.obj(rho=dist_left, phi=math.radians(phi_left))
        vec_right = vector.obj(rho=dist_right, phi=math.radians(phi_right))

        
        delta_rad = math.radians(self.delta_phi)
        vec_sector_left = self.direct.rotateZ(-delta_rad)
        vec_sector_right = self.direct.rotateZ(delta_rad)

        d_obs_phi = vec_left.deltaphi(vec_right)
        if d_obs_phi > 0:
            print(f'Error phi: phi left: {phi_left}, phi right: {phi_right}, delts: {d_obs_phi} -  by id: {id}, sector: {self.sector_phi}')
            return False
       
    
        # <0 - правее  >0 - левее относительно сетора
        d_phi_left_by_sector_left = vec_sector_left.deltaphi(vec_left)
        d_phi_right_by_sector_left = vec_sector_left.deltaphi(vec_right)
        d_phi_right_by_sector_right = vec_sector_right.deltaphi(vec_right)
        d_phi_left_by_sector_right = vec_sector_right.deltaphi(vec_left)


        left = 0
        right = 2 * self.delta_phi
        left_dir = vector.obj(phi=0,rho=1000)
        right_dir = vector.obj(phi=0,rho=1000)
        b_cross_sector = True
        if d_phi_left_by_sector_left >= 0:      # левый край препятствия левее левого края сектора
            if d_phi_right_by_sector_left >=0:  # правый край препятствия левее левого края сектора
                b_cross_sector = False          # препятвие вне сектора
            else:
                left = 0
                b_cross, left_dir = self.vec_to_cross_point(vec_left, vec_right, vec_sector_left)
                if b_cross == False:
                    left_dir.x = vec_left.x
                    left_dir.y = vec_left.y
                if d_phi_right_by_sector_right >= 0:# правый край препятствия левее правого края сектора
                     # сектора пересекаются 
                    right = self.to_int(2 * self.delta_phi - math.degrees(d_phi_right_by_sector_right))
                    right_dir.x = vec_right.x
                    right_dir.y = vec_right.y
                else:                               # правый край препятствия правее правого края сектора
                    # сектора пересекаются 
                    right = 2 * self.delta_phi
                    b_cross, right_dir = self.vec_to_cross_point(vec_left, vec_right, vec_sector_right)
                    if b_cross == False:
                        right_dir.x = vec_right.x
                        right_dir.y = vec_right.y
        else:                                   # левый край препятствия правее левого края сектора
            if d_phi_left_by_sector_right > 0: # левый край препятствия левее правого края сектора
                left = self.to_int(math.degrees(-d_phi_left_by_sector_left))
                left_dir.x = vec_left.x
                left_dir.y = vec_left.y
                if d_phi_right_by_sector_right >= 0:    # правый край препятствия левее правого края сектора
                    # сектора пересекаются 
                    right = self.to_int(2 * self.delta_phi - math.degrees(d_phi_right_by_sector_right) )
                    right_dir.x = vec_right.x
                    right_dir.y = vec_right.y
                else:                                   # правый край препятствия правее правого края сектора
                    # сектора пересекаются 
                    right = 2 * self.delta_phi    
                    b_cross, right_dir = self.vec_to_cross_point(vec_left, vec_right, vec_sector_right)  
                    if b_cross == False:
                        right_dir.x = vec_right.x
                        right_dir.y = vec_right.y
            else:
                b_cross_sector = False          # препятвие вне сектора

        if left > right or left < 0 or right > 2 * self.delta_phi:
            print(f'Error phi: phi left: {left}, phi right: {right} - by id: {id}, sector: {self.sector_phi}')
            return False

        if b_cross_sector:

            # рассотяние до препятствия 
            min_dist_vec_to_obst = vector.obj(x=1000,y=0)
            # вектор между точками
            vec_between_point = left_dir - right_dir
            vec_between_point /= vec_between_point.rho
            # определим можноли провести на отрезок перепендикуляр
            proection_left = vec_between_point @ left_dir
            proection_right = vec_between_point @ right_dir
            if self._sign(proection_left) == self._sign(proection_right):
                #нельзя
                min_dist_vec_to_obst = left_dir if left_dir.rho < right_dir.rho else right_dir
            else:   # можно 
                vec_between_point = vec_between_point.rotateZ(0.5*math.pi)
                min_dist_to_obst = vec_between_point @ right_dir
                min_dist_vec_to_obst = vec_between_point * abs(min_dist_to_obst)

            #is_mem = True if id in self.mem_obst_id else False
            is_mem = False

            if min_dist_vec_to_obst.rho < self.max_dist or is_mem == True:
                self.obstacles[id] = {'type': type, 'bind_phi_left': left, 'bind_phi_right': right, 'vec_dist': min_dist_vec_to_obst}

                if id not in self.mem_obst_id:
                    self.mem_obst_id.append(id)

                obst_type = ''
                dist = 1000
                line = np.zeros(int(2 * self.delta_phi), dtype=np.int8)
                for k, obst in self.obstacles.items():
                    n_start = int(obst['bind_phi_left'])
                    n_end = int(obst['bind_phi_right'])
                    for n in range(n_start, n_end):
                        line[n] = 1

                    if dist > obst['vec_dist'].rho:
                        dist = obst['vec_dist'].rho
                        obst_type = obst['type']
                
                density = 0
                for p in line:
                    if p == 1:
                        density += 1

                self.observation[0] = density / float(line.size)
                self.observation[1] = dist / self.max_dist
                if obst_type == 'tree' or obst_type == 'build':
                    self.observation[2] = 0.1
                else:
                    self.observation[2] = 0.2

        return b_cross_sector

    def vec_to_cross_point(self, crossed_start, crossed_end, vector_end):

        cross_point = vector.obj(x=0,y=0)

        vector_end.rho = 2000

        v11 = vector.obj(x=crossed_start.x,y=crossed_start.y,z=0)
        v12 = vector.obj(x=crossed_end.x,y=crossed_end.y,z=0)
        v21 = vector.obj(x=0,y=0,z=0)
        v22 = vector.obj(x=vector_end.x,y=vector_end.y,z=0)

        vector_for_crossed = v12 - v11
        vector_cross = v22 - v21
        
        prod1 = vector_for_crossed.cross(v21-v11)
        prod2 = vector_for_crossed.cross(v22-v11)

        if self._sign(prod1.z) == self._sign(prod2.z) :
            #print(f'WARNING: vectors is not crossed')
            return False, cross_point

        if prod1.z == 0 or prod2.z == 0: # Отсекаем также и пограничные случаи
            #print(f'WARNING: vectors is parallel')
            return False, cross_point

        prod1 = vector_cross.cross(v11-v21)
        prod2 = vector_cross.cross(v12-v21)

        if self._sign(prod1.z) == self._sign(prod2.z) :
            #print(f'WARNING: vectors is not crossed')
            return False, cross_point

        if prod1.z == 0 or prod2.z == 0: # Отсекаем также и пограничные случаи
            #print(f'WARNING: vectors is parallel')
            return False, cross_point

        cross_point.x = crossed_start.x + vector_for_crossed.x * abs(prod1.z)/abs(prod2.z-prod1.z)
        cross_point.y = crossed_start.y + vector_for_crossed.y * abs(prod1.z)/abs(prod2.z-prod1.z)
        
        return True, cross_point


    def to_int(self, value:float):
        if value > 0:
            return int(value + 0.5)
        else:
            return int(value - 0.5)
        
    def _sign(self, value:float)->float:
        if value < 0.:
            return -1
        else:
            return 1
