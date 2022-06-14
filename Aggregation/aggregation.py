from enum import Enum, auto
import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize
import pickle


for experiment_i in range(1):
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
            p_leave = experiment_i * 0.1
            p_join = experiment_j * 0.1
            leave = p_leave ** len(n) if len(n) > 0 else p_leave

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
                if np.random.uniform() > p_join:
                    self.move = pg.Vector2((0,0))
                    self.change_image(1)
                if np.linalg.norm(self.move) < 0.1 and np.random.uniform() > (1-leave):
                    self.move = np.random.uniform(low = -1, high = 1, size = 2)
                    self.change_image(0)
            else:
                self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
                f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.mass
                self.move += f_total

            self.pos += self.move * self.config.delta_time  #update pos



    class AggregationLive(Simulation):
        config: AggregationConfig

    x, y = AggregationConfig().window.as_tuple()


    dfs = (
        AggregationLive(
            AggregationConfig(
                movement_speed=1,
                radius=50,
                seed=1,
                duration=60*60*3 # 10 minutes
            )
        )
        .spawn_obstacle("images/boundary.png", x//2,  y//2)
        .spawn_obstacle("images/boundary.png", x//2,  y//2)
        .spawn_site("images/shadow_small.png", x//4 , y//2)
        .spawn_site("images/shadow_extralarge.png", (x//4)*3 , y//2)
        .batch_spawn_agents(50, Cockroach, images=["images/white.png","images/red.png"])
        .run()
    ).snapshots

    # pickle.dump(dfs,open(f"experiment{experiment_i * 0.1:.1f}.p",'wb'))

# pickle.dump(dfs,open("3min.p", "wb"))
