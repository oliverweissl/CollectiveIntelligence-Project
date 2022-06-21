from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from pyparsing import Optional
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize
import math

@deserialize
@dataclass
class AggregationConfig(Config):
    random_weight = 3
    circle_area = math.pi*Config.radius**2
    delta_time: float = 2
    mass: int = 20
    timer_j = 100./delta_time
    D = 1

    weigth_leave = 0.4 #gamma
    weight_join = 0.5

    #params for search
    join_param = 0.9
    loneliness_v1 = -0.4
    loneliness_v2 = 3

    mermory = 10 

    def weights(self) -> tuple[float, float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight, self.random_weight)


class Cockroach(Agent):
    config: AggregationConfig
        
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_neighbours = 0
        self.ROLLING_SIZE = 1 
        self.timer_j_counting = self.config.timer_j
        self.timer_j_started = False
        self.timer_j_picked = None
        self.leaving = False
        self.p_leave = None
        self.p_join = None
        self.loneliness = 0.5
        self.on_place = 0
        self.last_move = pg.Vector2((0,0))
        self.last_seen = [0]*self.config.mermory
        self.age = 1

    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["p_leave"].append(self.p_leave)
        self._Agent__simulation._metrics._temporary_snapshots["p_join"].append(self.p_join)
        
    def change_position(self):
        n = list(self.in_proximity_accuracy()) #list of neighbors
        len_n = len(n)
        self.last_seen.append(len_n)
        self.last_seen.pop(0)
        self.mean_neighbours = (self.mean_neighbours * (self.ROLLING_SIZE-1) + len_n) / self.ROLLING_SIZE
        density = self.mean_neighbours/self.config.circle_area
        
        self.loneliness = 1/(1+np.exp(self.config.loneliness_v1*np.average(self.last_seen)+self.config.loneliness_v2))*math.log(self.age)

        P_join = 0
        P_leave = (1+np.log(1+density)*300) * self.config.weigth_leave / (len_n ** math.log(len_n)) if len_n > 0 else self.config.weigth_leave
        #print(P_leave, (1+np.log(1+density)*100))

        #if self.on_site():
        #    self.on_place  +=2
        #    P_join = self.config.weight_join
        if len(n) > 2:
            self.on_place+=1
            P_join = 1-((1+np.log(1+density)*40)*self.config.join_param**math.log(len_n))
        else: self.on_place=0


        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
        f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.mass
        self.move += f_total


        if np.random.uniform() < P_join:
            self.move *= self.config.weigth_leave ** math.log(self.on_place) #experiment with slowing

        if np.linalg.norm(self.move) < 0.4:
            self.move = pg.Vector2((0,0))
            self.change_image(1)
            #self.change_image(self.on_site_id()+2) if self.on_site_id() is not None else self.change_image(1)
            self.move = np.random.uniform(low = -0.2, high = 0.2, size = 2)*(1-self.loneliness) if np.random.uniform() < P_leave and np.random.uniform() < 1/self.config.D else self.move
        else: self.change_image(0)



        #collision detection
        coll = list(self.obstacle_intersections(scale = 2))
        if len(coll) > 0:
            for c in coll:
                nm = self.move-(c-self.pos) #current move velocity - distance to the obstacle
                self.move = nm / np.linalg.norm(nm) #normalize vector

        self.pos += self.move * self.config.delta_time  #update pos
        self.age+=1


class AggregationLive(Simulation):
    config: AggregationConfig

x, y = AggregationConfig().window.as_tuple()

df = (
    AggregationLive(
        AggregationConfig(
            movement_speed=1,
            radius=50,
            seed=1,
            fps_limit=0
        )
    )
    .spawn_obstacle("images/boundary.png", x//2,  y//2)
    #.spawn_site("images/shadow_norm.png", 150 , y//2)
    #.spawn_site("images/shadow_norm.png", x-150 , y//2)
    .batch_spawn_agents(200, Cockroach, images=["images/white.png","images/red.png"])
    .run()
)


dfs = df.snapshots
dfs.write_csv(f"Experiments/A.csv")