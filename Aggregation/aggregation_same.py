from enum import Enum, auto
import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from pyparsing import Optional
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize
import math
import pickle

for experiment_p_join in np.linspace(0,1,10,endpoint=False):
    for experiment_p_leave in np.linspace(0,1,10,endpoint=False):
        @deserialize
        @dataclass
        class AggregationConfig(Config):
            random_weight = 3
            circle_area = math.pi*Config.radius**2
            delta_time: float = 2
            mass: int = 20
            timer_j = 100./delta_time
            D = 1

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

            def _collect_replay_data(self):
                super()._collect_replay_data()
                self._Agent__simulation._metrics._temporary_snapshots["p_leave"].append(self.p_leave)
                self._Agent__simulation._metrics._temporary_snapshots["p_join"].append(self.p_join)
                
            def change_position(self):
                n = list(self.in_proximity_accuracy()) #list of neighbors
                num_n = len(n)
                self.mean_neighbours = (self.mean_neighbours * (self.ROLLING_SIZE-1) + num_n) / self.ROLLING_SIZE
                leave = experiment_p_leave ** len(n) if len(n) > 0 else experiment_p_leave
                join = experiment_p_join
                #print(leave, self.mean_neighbours/self.config.circle_area, num_n)
                #print(leave)
                self.p_leave = leave
                
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
                    join_bool = np.random.uniform() < join
                    if not self.timer_j_started and not self.leaving and join_bool:
                        self.timer_j_started = True
                        self.timer_j_picked = np.random.uniform(0, self.config.timer_j)
                    elif not self.leaving and join_bool: 
                        if int(self.timer_j_counting) <= self.timer_j_picked:
                            self.timer_j_started = False
                            self.timer_j_picked = None
                            self.timer_j_counting = self.config.timer_j
                            self.move = pg.Vector2((0,0))
                            self.change_image(self.on_site_id()+1) if self.on_site_id() is not None else self.change_image(1)
                        
                        self.timer_j_counting -= 1
                    
                    if np.random.uniform() < 1/self.config.D and np.random.uniform() <= leave and not self.leaving and not self.timer_j_started:
                        self.leaving = True
                        self.move = np.random.uniform(low = -1, high = 1, size = 2)
                        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
                        self.change_image(0)
                    
                else:
                    if self.leaving:
                        self.leaving = False
                    self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
                    f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2))/self.config.mass
                    self.move += f_total
                
                self.ROLLING_SIZE += 1
                self.pos += self.move * self.config.delta_time  #update pos


        class AggregationLive(HeadlessSimulation):
            config: AggregationConfig

        x, y = AggregationConfig().window.as_tuple()

        df = (
            AggregationLive(
                AggregationConfig(
                    movement_speed=1,
                    duration = 10000,
                    radius=50,
                    seed=1,
                    fps_limit=0
                )
            )
            .spawn_obstacle("images/boundary.png", x//2,  y//2)
            .spawn_site("images/shadow_norm.png", x//4 , y//2)
            .spawn_site("images/shadow_norm.png", 3*x//4 , y//2)
            .batch_spawn_agents(50, Cockroach, images=["images/white.png","images/red.png","images/blue.png","images/green.png"])
            .run()
        )


        dfs = df.snapshots
        pickle.dump(dfs,open(f"new_round/same/hpo/j{experiment_p_join:.1f}l{experiment_p_leave:.1f}.p",'wb'))

