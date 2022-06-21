from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize, Window

GLOBAL_SEED = np.random.randint(0,1000000)

@deserialize
@dataclass
class Conf(Config):
    alignment_weight: float = 0.50
    cohesion_weight: float = 0.2
    separation_weight: float = 0.25

    random_weight = 1.3
    delta_time: float = 2
    mass: int = 20


class Grass(Agent):
    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["fox"].append(2) #fox:2 = grass


class Fox(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = np.random.uniform()*100
        self.age = 0
        self.max_age = 14
        self.p_reproduce = 0.15
        self.fox_mass = self.config.mass/2

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
        self.change_image(0)
        self.there_is_no_escape()
        if self.energy <= 1: self.kill()

        if self.is_alive():
            self.p_reproduce = 1/self.energy
            self.energy *= 0.94

            self.f = list(self.in_proximity_accuracy().filter_kind(Fox))
            r = list(self.in_proximity_accuracy().filter_kind(Rabbit))

            if len(r) > 0:
                self.change_image(1)
                r[0][0].kill()
                self.energy = min(300, self.energy+40)
                if np.random.uniform() < self.p_reproduce:
                    self.change_image(2)
                    self.reproduce()
            self.random_move()




class Rabbit(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 100
        self.age = 0
        self.max_age = 12
        self.p_reproduction = 0.008

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
        if self.energy == 0: self.kill()

        if self.is_alive():
            self.change_image(0)
            self.r = list(self.in_proximity_accuracy())

            prob = self.p_reproduction/(len(self.r)) if len(self.r) > 0 else self.p_reproduction
            if np.random.uniform() < prob:
                self.change_image(2)
                self.reproduce()
            self.random_move()




class Live(HeadlessSimulation):
    config: Conf
    def tick(self, *args, **kwargs):
        super().tick(*args, **kwargs)
        if self.shared.counter % 100 == 0:
            print(self.shared.counter)

        if len(list(filter(lambda x: isinstance(x,Fox), list(self._agents.__iter__())))) < 1:
            self.stop()



x, y = Conf().window.as_tuple()
df = (
    Live(
        Conf(
            window= Window(500,500),
            fps_limit=0,
            movement_speed=1,
            image_rotation=True,
            print_fps=False,
            radius=17,
            seed=GLOBAL_SEED

        )
    )
        .batch_spawn_agents(500, Rabbit, images=["images/white.png","images/red.png","images/green.png"])
        .batch_spawn_agents(20, Fox, images=["images/bird.png","images/bird_red.png","images/bird_green.png"])
        .run()
)

dfs = df.snapshots
dfs.write_parquet(f"X_{GLOBAL_SEED}.pqt")



