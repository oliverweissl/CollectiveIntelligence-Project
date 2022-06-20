from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from pyparsing import Optional
from vi import Agent, Simulation, HeadlessSimulation, Window
from vi.config import Config, dataclass, deserialize
import math

@deserialize
@dataclass
class RFConfig(Config):
    random_weight = 3
    delta_time: float = 2
    fox_mass: int = 40
    rabbit_mass: int = 5
    rabbit_death_prob = 0.8

    def weights(self) -> tuple[float, float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight, self.random_weight)


class Rabbit(Agent):
    config: RFConfig

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.elapsed = 0
        self.reproduce_factor = np.random.randint(150, 220)

    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["fox"].append(0)


    def change_position(self):
        self.there_is_no_escape()

        if self.elapsed and self.elapsed % self.reproduce_factor == 0:
            self.elapsed = 0
            self.reproduce()

        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
        f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.rabbit_mass
        self.move += f_total


        self.pos += self.move * self.config.delta_time  #update pos
        self.elapsed += 1

class Fox(Agent):
    config: RFConfig
    EAT_THRESH = 700

    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_reproduce = False
        self.reproduce_cooldown = np.random.randint(800, 1000)
        self.energy = 200

    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["fox"].append(1)


    def change_position(self):
        self.there_is_no_escape()


        if self.energy <= 0:
            self.kill()

        if self.alive():
            if self.can_reproduce and self.reproduce_cooldown <= 0:
                self.reproduce()
                self.reproduce_cooldown = np.random.randint(800, 1000)

            r = list(set(self.in_proximity_accuracy().filter_kind(Rabbit)))
            if len(r) > 0:
                r[0][0].kill()
                self.can_reproduce = True
                self.energy += 300


            self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
            f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.fox_mass
            self.move += f_total

            self.pos += self.move * self.config.delta_time  #update pos
            self.energy -= 1

            self.reproduce_cooldown = max(self.reproduce_cooldown-1,0)


class RFLive(Simulation):
    config: RFConfig

x, y = RFConfig().window.as_tuple()

df = (
    RFLive(
        RFConfig(
            movement_speed=1,
            radius=16,
            seed=1,
            duration=10000,
            fps_limit=0,
            image_rotation=True,
            window = Window(500, 500),
            print_fps = True
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
dfs.write_csv(f"X.csv")