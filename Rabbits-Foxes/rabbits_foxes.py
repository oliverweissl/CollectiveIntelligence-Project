from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from pyparsing import Optional
from vi import Agent, Simulation, HeadlessSimulation, Window
from vi.config import Config, dataclass, deserialize
import math
from copy import copy

np.random.seed(1337)
@deserialize
@dataclass
class RFConfig(Config):
    random_weight = 3
    delta_time: float = 2
    fox_mass: int = 10
    rabbit_mass: int = 10
    rabbit_death_prob = 0.8

    alignment_weight: float = 0.80
    cohesion_weight: float = 0.2
    separation_weight: float = 0.2
    random_weight: float = 0.3

    def weights(self) -> tuple[float, float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight, self.random_weight)


class Rabbit(Agent):
    config: RFConfig
        
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elapsed = 0
        self.reproduce_factor = np.random.randint(160, 190)
        
    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["fox"].append(0)
        
    def change_position(self):
        self.there_is_no_escape()
        if self.alive():
            n = list(self.in_proximity_performance()) #list of neighbors
            len_n = len(n)

            if self.elapsed and self.elapsed % self.reproduce_factor == 0:
                self.reproduce()

            if len_n >= 1:
                for neigh in n:
                    if isinstance(neigh, Fox):
                        self.kill()

            if len_n > 0:
                pos = [s.pos for s in n] 
                vec = [s.move for s in n]

                #c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
                s = np.average([self.pos - x for x in pos], axis = 0) #seperation
                #a = np.average(vec, axis = 0) - self.move #alignment


                f_total = (#self.config.alignment_weight * a +
                        self.config.separation_weight * s +
                        #self.config.cohesion_weight * c +
                        self.config.random_weight * np.random.uniform(low = -0.5, high = 0.5, size = 2)) / self.config.rabbit_mass

            else: f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.rabbit_mass
            self.move += f_total
            self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
            self.pos += self.move * self.config.delta_time  #update pos
            self.elapsed += 1

class Fox(Agent):
    config: RFConfig
    EAT_THRESH = 400

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_reproduce = False
        self.reproduce_cooldown = np.random.randint(800, 1000)
        self.energy = 200
    
    #def _collect_replay_data(self):
    #    super()._collect_replay_data()
    #    self._Agent__simulation._metrics._temporary_snapshots["p_leave"].append(self.p_leave)
    #    self._Agent__simulation._metrics._temporary_snapshots["p_join"].append(self.p_join)
    
    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["fox"].append(1)

    def change_position(self):

        self.there_is_no_escape()


        if self.alive():
            n = self.in_proximity_performance()
            n_copy = list(copy(n)) #list of neighbors
            r = list(set(n.filter_kind(Rabbit)))
            len_n = len(n_copy)
            
            if self.energy <= 0:
                self.kill()

            if self.can_reproduce and self.reproduce_cooldown <= 0:
                self.reproduce()
                self.reproduce_cooldown = np.random.randint(800, 1000)
                self.has_reproduced = True

            if len_n >= 1 and self.energy < Fox.EAT_THRESH:
                for neigh in n_copy:
                    if isinstance(neigh, Rabbit):
                        self.can_reproduce = True
                        self.energy += 200

            if len(r) > 0:

                pos = [s.pos for s in r] 
                vec = [s.move for s in r]

                c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
                #s = np.average([self.pos - x for x in pos], axis = 0) #seperation
                a = np.average(vec, axis = 0) - self.move #alignment


                f_total = (self.config.alignment_weight * a +
                        #self.config.separation_weight * s +
                        self.config.cohesion_weight * c +
                        self.config.random_weight * np.random.uniform(low = -0.5, high = 0.5, size = 2)) / self.config.fox_mass
            else: 
                self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
            f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.fox_mass
            self.move += f_total
        
            self.pos += self.move * self.config.delta_time  #update pos
            self.energy -= 1
            if self.reproduce_cooldown >= 1:
                self.reproduce_cooldown -= 1

class RFLive(Simulation):
    config: RFConfig

    def tick(self, *args, **kwargs):
        super().tick(*args, **kwargs)
        if self.shared.counter % 100 == 0:
            print(self.shared.counter)

x, y = RFConfig().window.as_tuple()

df = (
    RFLive(
        RFConfig(
            movement_speed=1,
            radius=16,
            seed=1,
            duration=10000,
            fps_limit=60,
            window = Window(500, 500),
            image_rotation=True
        )
    ) 
    #.spawn_obstacle("images/boundary.png", x//2,  y//2)
    #.spawn_site("images/shadow_norm.png", 150 , y//2)
    #.spawn_site("images/shadow_norm.png", x-150 , y//2)
    .batch_spawn_agents(66, Rabbit, images=["images/white.png"])
    .batch_spawn_agents(10, Fox, images=["images/bird_red.png"])
    .run()
)


dfs = df.snapshots
dfs.write_csv(f"Experiments/A.csv")