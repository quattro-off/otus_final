import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import math
import random
import vector 
import pygame
from PIL import Image, ImageDraw
from typing import Any, Dict, Tuple, Union


class FindPath(gym.Env):


    def name(self):
        return "FindPathCNN" if self.observation_render == True else "FindPath"

    def __init__(self, 
                 area_size: int=5,
                 seed: int=42, 
                 render_mode: str=None, 
                 observation_render: bool = False,
                 target_point_rand:bool = False,
                 wall_count: int = 3,
                 wall_length: int = 1,
                 options = None  
                 ):

        if area_size > 5 and wall_count == 3:
            wall_count = int(0.5*area_size)

        self.area_size = area_size
        self.observation_render = observation_render    # среда ввиде матрицы данных или картинка(оттенки серого)
        self.target_point_rand = target_point_rand      # целевая точка движения в центре или случайная
        self.options = options


        self.save_positions = []

        self.position = vector.obj(x=0,y=0)
        self.target_point = vector.obj(x=area_size-1,y=area_size-1)

        # visualiser
        self.render_mode = render_mode
        self.if_render = False
        self.is_pygame = False
        self.screen = {}


        if observation_render == True:
            self.observation_space =  Box(low=0,
                                          high=255, 
                                          shape=(area_size, area_size, 1), 
                                          dtype=np.uint8
                                          )
            self.observation_space_local = Box( # среда для вычислений всегда параметрическая
                                                low = 0,
                                                high = 255,
                                                shape = (area_size**2,),
                                                dtype = np.int8,
                                            )
        else:
            self.observation_space = Box(
                                                low = 0,
                                                high = 255,
                                                shape = (area_size**2,),
                                                dtype = np.int8,
                                            )

            self.observation_space_local = Box(
                                                low = 0,
                                                high = 255,
                                                shape = (area_size**2,),
                                                dtype = np.int8,
                                            )
                                            


        self.action_space = Discrete(9, seed=42)
        self.action_space.seed(seed=seed)

        super().reset(seed=seed)
        random.seed(seed)
        self.observation_space.seed(seed)
        self.observation_space_local.seed(seed)

        self.tick = 0
        self.max_tick = 0
        self.max_step = 0
        self.zero_step = 0

        self.walls = []
        for _ in range(wall_count):
            self.walls.append(vector.obj(x=0,y=0))

        self.window_koef = int(400/self.area_size) if self.area_size < 200 else 1
        self.window_size = (self.window_koef*self.area_size, self.window_koef*self.area_size)

        if render_mode == 'human':
            self.is_render = False
            self.is_pygame = True
            pygame.init()
            self.screen = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Moving")
        elif render_mode == 'rgb_array':
            self.is_pygame = False
            self.is_render = True

    def set_options(self, options = None):
        self.options = options

    def reset(self, seed: int = None):

        if seed != None:
            random.seed(seed)
            self.observation_space.seed(seed)
            self.observation_space_local.seed(seed)

        self.tick = 0
        self.zero_step = 0

        observation = np.zeros(self.observation_space_local.shape, dtype=np.int16)

        if self.options == None:

            self.position.x = random.randint(0, self.area_size-1)
            self.position.y = random.randint(0, self.area_size-1)

            # целевая точка движения
            if self.target_point_rand == True:
                self.target_point.x = random.randint(0, self.area_size-1)
                self.target_point.y = random.randint(0, self.area_size-1)
            else:
                self.target_point.x = int(0.5*self.area_size)
                self.target_point.y = int(0.5*self.area_size)

            self.max_tick = self.area_size * 2

        else:
            
            # целевая точка движения
            if self.target_point_rand == True:
                self.target_point.x = random.randint(0, self.area_size-1)
                self.target_point.y = random.randint(0, self.area_size-1)
            else:
                self.target_point.x = int(0.5*self.area_size)
                self.target_point.y = int(0.5*self.area_size)

            rad =  self.options['start_radius']
            phi = math.radians(random.randint(0, 360))
            point = vector.obj(phi=phi,rho=rad)
            self.position.x = self.target_point.x + int(point.x)
            self.position.y = self.target_point.y + int(point.y)
            #self.position.x = random.randint(self.target_point.x - rad, self.target_point.x + rad)
            #self.position.y = random.randint(self.target_point.y - rad, self.target_point.y + rad)
            self.position.x = self._clamp(self.position.x, 0, self.area_size-1)
            self.position.y = self._clamp(self.position.y, 0, self.area_size-1)

            self.max_tick = rad * 4
        
        
        self.save_positions = [vector.obj(x=self.position.x,y=self.position.y)]

        vec_dist = self.target_point - self.position
        self.max_step = vec_dist.x if vec_dist.x > vec_dist.y else vec_dist.y

        for wall in self.walls:
                for _ in range(10):
                    wall.x = random.randint(0, self.area_size-1)
                    if wall.x != self.target_point.x and wall.x != self.target_point.x:
                        break
                for _ in range(10):
                    wall.y = random.randint(0, self.area_size-1)
                    if wall.y != self.target_point.y and wall.y != self.target_point.y:
                        break
        


        self.reward = 0.
        
        info = self._get_info()

        if self.observation_render == True:
            rend = self._get_render(True)
            return rend, info
        else:
            return observation, info

    def close(self):
        if self.is_pygame == True:
            pygame.quit()
        super().close()


    def _get_info(self):
        return {"tick progress: ": self.tick}
    
    def step(self, action):



        terminated = False
        truncated = False
        reward_step = -1
        reward_out = 0
        reward_old_path = 0
        reward_move = 0

        new_position = vector.obj(x=self.position.x,y=self.position.y)

        #   0 1 2 3 4 5 6 
        # 0 +-------------> X
        # 1 |
        # 2 |         up Y-
        # 3 | right X-    left X+
        # 4 |        down Y+
        # 5 |
        # 6 |
        #  \/
        #   Y

        if action == 0:
        # 0 - stop
            self.zero_step += 1
            reward_step = -5 * self.zero_step
        elif action == 1:
        # 1 - up
            new_position.y -= 1
        elif action == 2:
        # 2 - up right
            new_position.x += 1
            new_position.y -= 1
        elif action == 3:
        # 3 - right
            new_position.x += 1
        elif action == 4:
        # 4 - down right
            new_position.x += 1
            new_position.y += 1
        elif action == 5:
        # 5 - down
            new_position.y += 1
        elif action == 6:
        # 6 - down left
            new_position.x -= 1
            new_position.y += 1        
        elif action == 7:
        # 7 - left
            new_position.x -= 1
        elif action == 8:
        # 8 - up left
            new_position.x -= 1
            new_position.y -= 1

        if action != 0:
            self.zero_step = 0

        observation = np.zeros(self.observation_space_local.shape, dtype=np.int16)

        is_wall = False
        for wall in self.walls:
            place_wall = wall.x + wall.y * self.area_size
            if place_wall > len(observation):
                print('ERROR: out of range')
            observation[place_wall] = 200
            if wall == new_position:
                is_wall = True

        if is_wall or new_position.x >= self.area_size or new_position.y >= self.area_size or new_position.x < 0 or new_position.y < 0:
            reward_out = -4
        else:        

            for point in self.save_positions:
                if point == new_position:
                    reward_old_path = -4
                    break

            last_dist = self.target_point - self.position
            curr_dist = self.target_point - new_position
            if curr_dist.rho > last_dist.rho:
                reward_move = -1 
            else:
                reward_move = 2 

            self.position.x = new_position.x
            self.position.y = new_position.y 

        
        self.save_positions.append(vector.obj(x=self.position.x,y=self.position.y))

        place_target = self.target_point.x + self.target_point.y * self.area_size
        if place_target > len(observation):
            print('ERROR: out of range')
        observation[place_target] = 250

        place_position = self.position.x + self.position.y * self.area_size
        if place_position > len(observation):
            print('ERROR: out of range')
        observation[place_position] = 100


        reward_step += reward_out + reward_old_path + reward_move

        if self.position == self.target_point:
            reward_step = self.max_step
            terminated = True
            truncated = False
        elif self.tick > self.max_tick:
            reward_step = -self.tick
            terminated = False
            truncated = True

        
        info = self._get_info()
        self.tick += 1

        if self.is_pygame == True:
            self.human_render()

        if self.observation_render == True:
            return self._get_render(True), float(reward_step), terminated, truncated, info
        else:
            return observation, float(reward_step), terminated, truncated, info


    def teach_action(self):

        t_action = 0
            
        teach_set_dir = self.target_point - self.position

        if teach_set_dir.x > 0:
        # 3 - right
            t_action = 3
        elif teach_set_dir.x < 0:
        # 7 - left
            t_action = 7
        elif teach_set_dir.y > 0:
        # 5 - down
            t_action = 5
        elif teach_set_dir.y < 0:
        # 1 - up
            t_action = 1
        # 0 - stop
        # 2 - up right
        # 4 - down right
        # 6 - down left
        # 8 - up left

        return t_action
    

    def _get_render(self, to_observation: bool):
        black = (0, 0, 0)
        white = (255, 255, 255)
        red = (255, 0, 0)
        green = (0, 255, 0)
        blue = (0, 0, 255)

        
        


        if to_observation == True:
            im = Image.new('RGB', self.window_size, black)
            draw = ImageDraw.Draw(im)

            to_draw = vector.obj(x=self.target_point.x,y=self.target_point.y)
            to_draw *= self.window_koef
            draw.polygon(
                xy=(
                    (to_draw.x,                   to_draw.y),
                    (to_draw.x+self.window_koef,  to_draw.y),
                    (to_draw.x+self.window_koef,  to_draw.y+self.window_koef),
                    (to_draw.x,                   to_draw.y+self.window_koef),
                    ), fill=(0,0,250), outline=(0,0,250)
            )

            to_draw = vector.obj(x=self.position.x,y=self.position.y)
            to_draw *= self.window_koef
            draw.polygon(
                xy=(
                    (to_draw.x,                   to_draw.y),
                    (to_draw.x+self.window_koef,  to_draw.y),
                    (to_draw.x+self.window_koef,  to_draw.y+self.window_koef),
                    (to_draw.x,                   to_draw.y+self.window_koef),
                    ), fill=(0,0,100), outline=(0,0,100)
            )

            for wall in self.walls:
                to_draw = vector.obj(x=wall.x,y=wall.y)
                to_draw *= self.window_koef
                draw.polygon(
                    xy=(
                        (to_draw.x,                   to_draw.y),
                        (to_draw.x+self.window_koef,  to_draw.y),
                        (to_draw.x+self.window_koef,  to_draw.y+self.window_koef),
                        (to_draw.x,                   to_draw.y+self.window_koef)
                    ), fill=(0,0,200), outline=(0,0,200)
                )


            im = im.convert('L')
            im_np = np.asarray([im])
            im_np = im_np.transpose((1, 2, 0))
        else:

            x = self.window_size[0]
            y = self.window_size[1]
            k = self.window_koef
            if k == 1:
                if self.window_size[0] < 201:
                    k = 4
                elif self.window_size[0] < 401:
                    k = 2

                x *= k
                y *= k
                

            im = Image.new('RGB', (x,y), black)
            draw = ImageDraw.Draw(im)

            for point in self.save_positions:
                to_draw = vector.obj(x=point.x,y=point.y)
                to_draw *= k        
                draw.polygon(
                xy=(
                        (to_draw.x,    to_draw.y),
                        (to_draw.x+k,  to_draw.y),
                        (to_draw.x+k,  to_draw.y+k),
                        (to_draw.x,    to_draw.y+k),
                    ), fill=green, outline=green
                )

            to_draw = vector.obj(x=self.target_point.x,y=self.target_point.y)
            to_draw *= k
            draw.polygon(
                xy=(
                        (to_draw.x,    to_draw.y),
                        (to_draw.x+k,  to_draw.y),
                        (to_draw.x+k,  to_draw.y+k),
                        (to_draw.x,    to_draw.y+k),
                    ), fill=white, outline=white
            )

            to_draw = vector.obj(x=self.position.x,y=self.position.y)
            to_draw *= k
            draw.polygon(
                xy=(
                        (to_draw.x,    to_draw.y),
                        (to_draw.x+k,  to_draw.y),
                        (to_draw.x+k,  to_draw.y+k),
                        (to_draw.x,    to_draw.y+k),
                    ), fill=red, outline=red
            )

            for wall in self.walls:
                to_draw = vector.obj(x=wall.x,y=wall.y)
                to_draw *= k
                draw.polygon(
                    xy=(
                        (to_draw.x,    to_draw.y),
                        (to_draw.x+k,  to_draw.y),
                        (to_draw.x+k,  to_draw.y+k),
                        (to_draw.x,    to_draw.y+k),
                    ), fill=blue, outline=blue
                )

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

        for point in self.save_positions:

            to_draw = vector.obj(x=point.x,y=point.y)
            to_draw *= self.window_koef
            pygame.draw.polygon(self.screen, green,
            [
                (to_draw.x,                   to_draw.y),
                (to_draw.x+self.window_koef,  to_draw.y),
                (to_draw.x+self.window_koef,  to_draw.y+self.window_koef),
                (to_draw.x,                   to_draw.y+self.window_koef)
            ]
            )

        to_draw = vector.obj(x=self.target_point.x,y=self.target_point.y)
        to_draw *= self.window_koef
        pygame.draw.polygon(self.screen, white,
            [
                (to_draw.x,                   to_draw.y),
                (to_draw.x+self.window_koef,  to_draw.y),
                (to_draw.x+self.window_koef,  to_draw.y+self.window_koef),
                (to_draw.x,                   to_draw.y+self.window_koef)
            ]
        )

        to_draw = vector.obj(x=self.position.x,y=self.position.y)
        to_draw *= self.window_koef
        pygame.draw.polygon(self.screen, red,
            [
                (to_draw.x,                   to_draw.y),
                (to_draw.x+self.window_koef,  to_draw.y),
                (to_draw.x+self.window_koef,  to_draw.y+self.window_koef),
                (to_draw.x,                   to_draw.y+self.window_koef)
            ]
        )

        for wall in self.walls:
            to_draw = vector.obj(x=wall.x,y=wall.y)
            to_draw *= self.window_koef
            pygame.draw.polygon(self.screen, blue,
                [
                    (to_draw.x,                   to_draw.y),
                    (to_draw.x+self.window_koef,  to_draw.y),
                    (to_draw.x+self.window_koef,  to_draw.y+self.window_koef),
                    (to_draw.x,                   to_draw.y+self.window_koef)
                ]
            )


        pygame.display.update()

        # устанавливаем частоту обновления экрана
        pygame.time.Clock().tick(60)

    def _clamp(self, value: int, min: int, max: int):
        if value > max:
            value = max
        elif value < min:
            value = min
        return value