from typing import TYPE_CHECKING, Any, Generator, Optional
from pygame.mask import Mask
from pygame.math import Vector2
from pygame.rect import Rect
from pygame.sprite import Group, Sprite
from pygame.surface import Surface
from typing_extensions import Self

import pandas as pd
import math
import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi.util import random_angle, random_pos, round_pos
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize


@deserialize
@dataclass
class AggregationConfig(Config):
    random_weight = 3

    weigth_leave = 0.2 #gamma
    weight_join = 0.5


    #params for search
    join_param = 0.8
    loneliness_v1 = -0.4
    loneliness_v2 = 3

    mermory = 10 #memory of individual


    delta_time: float = 2
    mass: int = 20




class Cockroach(Agent):
    config: AggregationConfig
    def __init__(self, images: list[Surface], simulation: HeadlessSimulation, pos: Optional[Vector2] = None, move: Optional[Vector2] = None):
        Agent.__init__(self, images=images, simulation=simulation,pos=pos,move=move)
        self.loneliness = 0.5
        self.on_place = 0
        self.last_move = pg.Vector2((0,0))

        self.last_seen = [0]*self.config.mermory
        self.age = 1

    def change_position(self):
        n = list(self.in_proximity_accuracy()) #list of neighbors
        len_n = len(n)
        self.last_seen.append(len_n)
        self.last_seen.pop(0)

        self.loneliness = 1/(1+np.exp(self.config.loneliness_v1*np.average(self.last_seen)+self.config.loneliness_v2))*math.log(self.age)

        P_join = 0
        P_leave = self.config.weigth_leave / len_n ** math.log(len_n) if len_n > 0 else self.config.weigth_leave


        if self.on_site():
            self.on_place  +=2
            P_join = self.config.weight_join
        elif len(n) > 2:
            self.on_place+=1
            P_join = 1-(self.config.join_param**math.log(len_n))
        else: self.on_place=0


        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
        f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.mass
        self.move += f_total


        if np.random.uniform() < P_join:
            self.move *= self.config.weigth_leave ** math.log(self.on_place) #experiment with slowing

        if np.linalg.norm(self.move) < 0.4:
            self.move = pg.Vector2((0,0))

            self.change_image(self.on_site_id()+2) if self.on_site_id() is not None else self.change_image(1)
            self.move = np.random.uniform(low = -0.2, high = 0.2, size = 2)*(1-self.loneliness) if np.random.uniform() < P_leave else self.move
        else: self.change_image(0)



        #collision detection
        coll = list(self.obstacle_intersections(scale = 2))
        if len(coll) > 0:
            for c in coll:
                nm = self.move-(c-self.pos) #current move velocity - distance to the obstacle
                self.move = nm / np.linalg.norm(nm) #normalize vector

        self.pos += self.move * self.config.delta_time  #update pos
        self.age+=1


x, y = AggregationConfig().window.as_tuple()
print(x,y)
df = (
    Simulation(
        AggregationConfig(
            fps_limit = 0,
            duration = 10000,
            movement_speed=1,
            radius=50,
            seed=1,
        )
    )
    .spawn_obstacle("images/boundary.png", x//2,  y//2)
    .batch_spawn_agents(50, Cockroach, images=["images/white.png","images/red.png","images/blue.png","images/green.png"])
    .run()
)
#.spawn_site("images/shadow_norm.png", x//2 , y//2)

dfs = df.snapshots
dfs.write_csv(f"X.csv")

dx=pd.read_csv(f"X.csv")
score = dx["image_index"][dx["image_index"]!=1].count() / 10000
print(f"score: {score}")