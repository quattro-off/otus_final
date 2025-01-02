import math
import vector 

def f_clamp(value: float, min: float, max: float):
        if value > max:
            value = max
        elif value < min:
            value = min
        return value
    
def f_death(value: float, min: float, max: float):
        if min < value and value < max:
            value = 0
        return value
    
def f_get_course_bind(global_phi:float, phi:float, cicle:bool = False)->float:
        if cicle == True:
            phi = phi if phi >= 0 else phi + 2 * math.pi
            view_phi = global_phi if global_phi >= 0 else global_phi + 2 * math.pi
            delta_angle = phi - view_phi
            if delta_angle < 0:
                delta_angle += 2 * math.pi

            if delta_angle < 0 or delta_angle > 2 * math.pi:
                print(delta_angle)
        else:
            delta_angle = phi - global_phi
            sign = -1 if delta_angle >= 0 else 1
            if abs(delta_angle) > math.pi:
                delta_angle = sign * (2 * math.pi - abs(delta_angle))
        return delta_angle
    
def f_equivalent(value1:float, value2:float, epsilon:float = 0.0001)->bool:
        return True if abs(value1 - value2) < epsilon else False
        
def f_equivalent_vec3(value1:vector, value2:vector, epsilon:float = 0.0001)->bool:
        if abs(value1.x - value2.x) < epsilon and abs(value1.y - value2.y) < epsilon and abs(value1.z - value2.z) < epsilon:
            return True
        else:
            return False
        
def f_min_then_max(v1:float, v2:float):
        if v1 < v2:
            return v1, v2
        else:
            return v2, v1
        
def f_sign(value:float)->float:
        if value < 0.:
            return -1
        else:
            return 1

def f_sign_or_zero(value:float)->float:
        if f_equivalent(value, 0.) == True:
            return 0.
        else:
            return f_sign(value)