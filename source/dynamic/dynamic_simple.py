import numpy as np
import math
import vector 
import common.functions as f
#from typing import Any, Dict, Tuple, Union

import sys
import os

sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../common'))


class DynamivSimple():


    #constants
    time_step = 0.02
    time_tick = 0.1
    speed_acc = 2. #m2/s
    angle_speed_acc = 2. #rad/s


    def name(self):
        return "DynamivSimple" 

    def __init__(self
                 , speed_max:float = 3.6
                 , speed_min:float = 0.4
                 , angle_speed_max:float = math.pi/3
                 ):

        self.tick = 0

        #variables for move
        self.time_model = 0.

        self.speed_local =vector.obj(x=0.,y=0.) # в локальной системме координат 
        self.speed_bind = vector.obj(x=0.,y=0.) # в связанной системме координат
        self.angle_speed = 0.


        self.position = vector.obj(x=0.,y=0.)
        self.dir_view = vector.obj(x=1.,y=0.)
        self.vec_dist_to_obctacle_bind = vector.obj(x=100.,y=100.)

        self.speed_min = speed_min
        self.speed_max = speed_max   
        self.angle_speed_max = angle_speed_max

    def get_model_time(self):
        return self.time_model
    
    def get_position(self):
        return vector.obj(x=self.position.x,y=self.position.y)

    def get_direction(self):
        return vector.obj(x=self.dir_view.x,y=self.dir_view.y)
    
    def get_speed(self):
        return vector.obj(x=self.speed_local.x,y=self.speed_local.y)
    
    def get_speed_bind(self):
        return vector.obj(x=self.speed_bind.x,y=self.speed_bind.y)
    
    def get_angle_speed(self):
        return self.angle_speed 

    def get_min_time_moving(self, to_position:vector):
        dist = to_position - self.position
        return dist.rho / self.speed_max
    
    def get_delta_angle_direction(self, to_direction:vector):
        return to_direction.deltaphi(self.dir_view)

    def set_position(self, position:float):
        self.position.x = position.x
        self.position.y = position.y

    def reset(self, start_position:vector, start_direction:vector):

        self.tick = 0
        self.time_model = 0.

        self.position = start_position
        self.dir_view = start_direction
                
        self.speed_local =vector.obj(x=0.,y=0.) # в локальной системме координат 
        self.speed_bind = vector.obj(x=0.,y=0.) # в связанной системме координат

        self.angle_speed = 0.

        self.dir_view = vector.obj(x=1.,y=0.)
        self.vec_dist_to_obctacle_bind = vector.obj(x=100.,y=100.)

    def set_obstacle(self, vec_to_obstacle:vector):
        self.vec_dist_to_obctacle_bind.x = vec_to_obstacle.x
        self.vec_dist_to_obctacle_bind.y = vec_to_obstacle.y


    def step(self, set_speed_forward:float, set_speed_right:float, set_angle_speed:float):

       

        new_speed = vector.obj(x=self.speed_bind.x, y=self.speed_bind.y)
        new_position = vector.obj(x=self.position.x, y = self.position.y)
        new_angle_speed = self.angle_speed
        dir_view_right = vector.obj(x=-self.dir_view.y,y=self.dir_view.x)

        final_time = self.time_model + self.time_tick
        while self.time_model < final_time:
            self.time_model += self.time_step


            # угловая скорость вправо - влево
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            angle_speed_sign = 1.
            if f.f_equivalent(set_angle_speed, self.angle_speed) == True:
                angle_speed_sign = 0
            elif set_angle_speed < self.angle_speed:
                angle_speed_sign = -1.
            # меняем угловую скорость
            new_angle_speed += angle_speed_sign * self.angle_speed_acc * self.time_step
            new_angle_speed = f.f_clamp(new_angle_speed, -self.angle_speed_max, self.angle_speed_max)
            # доворачивает текщий вектор движения в сторону заданного на угол за шаг
            # угол на который можно развернутся за шаг
            angle_step = new_angle_speed * self.time_step
            self.dir_view = self.dir_view.rotateZ(angle_step)
            dir_view_right = vector.obj(x=-self.dir_view.y,y=self.dir_view.x)

 
            # скорость вперед - назад
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_forward_sign = 1.
            if f.f_equivalent(set_speed_forward, new_speed.x) == True:
                speed_forward_sign = 0
            elif set_speed_forward < new_speed.x:
                speed_forward_sign = -1.


            # скорость вправо - влево
            # по разности текущей и заданной скорости выбираем тормозить или разгоняться
            speed_right_sign = 1.
            if f.f_equivalent(set_speed_right, new_speed.y) == True:
                speed_right_sign = 0
            elif set_speed_right < new_speed.y:
                speed_right_sign = -1.

            
            # меняем скорость
            new_speed.x += speed_forward_sign * self.speed_acc * self.time_step
            new_speed.x = f.f_clamp(new_speed.x, -self.speed_max, self.speed_max)
            new_speed.y += speed_right_sign * self.speed_acc * self.time_step
            new_speed.y = f.f_clamp(new_speed.y, -self.speed_max, self.speed_max)

            if new_speed.rho > self.speed_max:
                new_speed /= new_speed.rho
                new_speed *= self.speed_max

            # проверяем препятсвия по пути
            min_dist_to_obst = 2 * self.speed_max * self.time_tick
            if self.vec_dist_to_obctacle_bind.rho < min_dist_to_obst:
                cos_obst = new_speed @ self.vec_dist_to_obctacle_bind
                if cos_obst > 0:
                    #if self.vec_dist_to_obctacle_bind.rho < 0.5:
                    #    new_speed.x = 0
                    #    new_speed.y = 0
                    #else:
                        right_obst = vector.obj(x=-self.vec_dist_to_obctacle_bind.y, y=self.vec_dist_to_obctacle_bind.x)
                        right_speed = new_speed @ right_obst / right_obst.rho
                        new_speed = right_speed * right_obst / right_obst.rho

            #if self.vec_dist_to_obctacle_bind.rho < 0.5:
            #    print(self.vec_dist_to_obctacle_bind.rho, new_speed)


            # переводим скорость в локальную системму координат 
            speed = new_speed.x * self.dir_view + new_speed.y * dir_view_right

            # движение в мертвой области нет
            if -self.speed_min > speed.rho or speed.rho > self.speed_min:
                new_position += speed * self.time_step


        if self.dir_view.phi > math.pi or self.dir_view.phi < -math.pi:
            print(set_angle_speed)
            print(self.dir_view)

        self.speed_local = new_speed.x * self.dir_view + new_speed.y * dir_view_right

        self.speed_bind.x = new_speed.x
        self.speed_bind.y = new_speed.y

        self.angle_speed = new_angle_speed

        self.position.x = new_position.x
        self.position.y = new_position.y

    def teach_action(self, target_point:vector, continuous:bool):

        t_action = []

        teach_set_dir = target_point - self.position
        delta_angle = teach_set_dir.deltaphi(self.dir_view)
            
        if continuous == True:
            teach_set_speed = 1
            if teach_set_dir.rho < 2:
                teach_set_speed /= 7

            delta_angle = f.f_clamp(delta_angle, -self.angle_speed_max, self.angle_speed_max)
            teach_set_angle_speed = delta_angle / self.angle_speed_max

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

