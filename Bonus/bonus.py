from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize

@deserialize
@dataclass
GLOBAL_SEED = 1

class AggregationConfig(Config):
    random_weight = 3
    delta_time: float = 2
    mass: int = 20



class Fox(Agent):
    config: AggregationConfig
    def change_position(self):

class Rabbit(Agent):
    config: AggregationConfig
    def change_position(self):



class Live(Simulation):
    config: Config

x, y = Config().window.as_tuple()
df = (
    Live(
        Config(
            movement_speed=1,
            radius=50,
            seed=GLOBAL_SEED,
        )
    )
        .batch_spawn_agents(50, Rabbit, images=["images/white.png","images/red.png","images/green.png"])
        .batch_spawn_agents(5, Fox, images=["images/bird.png","images/bird_red.png","images/bird_green.png"])
        .run()
)


dfs = df.snapshots
#dfs.write_csv(f"Experiments/A{FlockingConfig.alignment_weight:.2f}_C{FlockingConfig.cohesion_weight:.2f}_S{FlockingConfig.separation_weight:.2f}_W{FlockingConfig.random_weight:.2f}.csv")

