from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize

@deserialize
@dataclass
class AggregationConfig(Config):
    random_weight = 3

    delta_time: float = 2
    mass: int = 20

    def weights(self) -> tuple[float, float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight, self.random_weight)


class Cockroach(Agent):
    config: AggregationConfig
    def change_position(self):
        n = list(self.in_proximity_accuracy()) #list of neighbors
        leave = 0.3 **  len(n) if len(n) > 0 else 0.3

        """
        if len(n) > 1:
            if np.random.uniform() > 0.999:
                self.move = pg.Vector2((0,0))
            else: self.move *= (1 - len(n))
            if np.linalg.norm(self.move) < 0.1 and np.random.uniform() > (1-leave):
                self.move = np.random.uniform(low = -1, high = 1, size = 2)
        """

        #collision detection
        coll = list(self.obstacle_intersections(scale = 2))
        if len(coll) > 0:
            for c in coll:
                nm = self.move-(c-self.pos) #current move velocity - distance to the obstacle
                self.move = nm / np.linalg.norm(nm) #normalize vector

        if self.on_site():
            if np.random.uniform() > 0.9:
                self.move = pg.Vector2((0,0))
            if np.linalg.norm(self.move) < 0.1 and np.random.uniform() > (1-leave):
                self.move = np.random.uniform(low = -1, high = 1, size = 2)
        else:
            self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
            f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.mass
            self.move += f_total

        self.pos += self.move * self.config.delta_time  #update pos


class AggregationLive(Simulation):
    config: AggregationConfig

x, y = AggregationConfig().window.as_tuple()

df = (
    AggregationLive(
        AggregationConfig(
            movement_speed=1,
            radius=50,
            seed=1,
        )
    )
    .spawn_obstacle("images/boundary.png", x//2,  y//2)
    .spawn_site("images/shadow_norm.png", x//2 , y//2)
    .batch_spawn_agents(50, Cockroach, images=["images/white.png","images/red.png"])
    .run()
)


dfs = df.snapshots
#dfs.write_csv(f"Experiments/A{FlockingConfig.alignment_weight:.2f}_C{FlockingConfig.cohesion_weight:.2f}_S{FlockingConfig.separation_weight:.2f}_W{FlockingConfig.random_weight:.2f}.csv")