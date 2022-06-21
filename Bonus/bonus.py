from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize, Window
GLOBAL_SEED = 1

@deserialize
@dataclass
class Conf(Config):
    alignment_weight: float = 0.50
    cohesion_weight: float = 0.2
    separation_weight: float = 0.25

    random_weight = 1.3
    delta_time: float = 2
    mass: int = 20



class Fox(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_age = 150*60
        self.breed_age = 15*60
        self.p_reproduction = 0.08

        self.age = np.random.randint(0,self.max_age/2)

        self.hunger = 10*60
        self.max_hunger = 10*60
        self.food_val = 9*60

    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["fox"].append(1)

    def random_move(self):
        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move

        if len(self.f) > 0:
            pos = [s[0].pos for s in self.f]
            vec = [s[0].move for s in self.f]

            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment

            f_total = (self.config.alignment_weight * a +
                       self.config.separation_weight * s +
                       self.config.cohesion_weight * c +
                       self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.fox_mass

        else: f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.fox_mass
        self.move += f_total

        self.pos += self.move * self.config.delta_time


    def change_position(self):
        self.change_image(1)
        self.there_is_no_escape()
        if self.age > self.max_age or self.hunger <= 0: self.kill()

        if self.is_alive():
            self.p_reproduce = 1/self.energy
            self.energy *= 0.94



            n = self.in_proximity_accuracy()
            r = list(n.filter_kind(Rabbit))
            self.f = list(n.filter_kind(Fox))
            if len(r) > 0:
                r[0][0].kill()
                self.energy = min(300, self.energy+40)
                if np.random.uniform() < self.p_reproduce:
                    self.reproduce()

            self.change_image(1)
            self.random_move()




class Rabbit(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.breed_age = 5*60
        self.max_age = 40*60

        self.age = np.random.randint(0,self.max_age/2)
        self.p_reproduction = 0.12/40

    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["fox"].append(0)

    def random_move(self):
        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move

        if len(self.r) > 0:
            pos = [s[0].pos for s in self.r]
            vec = [s[0].move for s in self.r]

            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment

            f_total = (self.config.alignment_weight * a +
                       self.config.separation_weight * s +
                       self.config.cohesion_weight * c +
                       self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.config.mass

        else: f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.config.mass
        self.move += f_total

        self.pos += self.move * self.config.delta_time


    def change_position(self):
        self.there_is_no_escape()
        if self.age > self.max_age: self.kill()
        if self.is_alive():
            self.r = list(self.in_proximity_accuracy())

            prob = self.p_reproduction/(len(self.r)) if len(self.r) > 0 else self.p_reproduction
            if np.random.uniform() < prob: self.reproduce()
            self.random_move()




class Live(Simulation):
    config: Conf



x, y = Conf().window.as_tuple()
df = (
    Live(
        Conf(
            window= Window(500,500),
            fps_limit=0,
            duration=50000,
            movement_speed=1,
            image_rotation=True,
            print_fps=True,
            radius=17,
            seed=GLOBAL_SEED

        )
    )
        .batch_spawn_agents(500, Rabbit, images=["images/white.png","images/red.png","images/green.png"])
        .batch_spawn_agents(20, Fox, images=["images/fox.png","images/bird_red.png","images/bird_green.png"])
        .run()
)

dfs = df.snapshots
dfs.write_parquet(f"X_{GLOBAL_SEED}.pqt")

