from enum import Enum, auto

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize, Window

for i in range(5):
    GLOBAL_SEED = np.random.randint(0,1000000)
    frames = 0

    @deserialize
    @dataclass
    class Conf(Config):
        alignment_weight: float = 0.50
        cohesion_weight: float = 0.2
        separation_weight: float = 0.25

        random_weight = 1.3
        delta_time: float = 2
        mass: int = 20

        hunter_visual_radius = 30
        hunter_eating_radius = 17
        prey_visual_radius = 30


    class Food(Agent):
        def _collect_replay_data(self):
            super()._collect_replay_data()
            self._Agent__simulation._metrics._temporary_snapshots["type"].append(2) # 2: food


    class Hunter(Agent):
        config: Conf
        def __init__(self,  *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.energy = np.random.uniform()*100
            self.age = 0
            self.max_age = 14
            self.p_reproduce = 0.15
            self.hunter_mass = self.config.mass/2

        def _collect_replay_data(self):
            super()._collect_replay_data()
            self._Agent__simulation._metrics._temporary_snapshots["type"].append(1) # 1: hunter

        def calc(self,pos,vec):
            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment
            return c,s,a

        def random_move(self):
            self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
            ad,sd,cd,rd = 0,0,0,1
            a,s,c = 0,0,0
            if len(self.hunters_in_visual_radius) > 0:
                pos = [s[0].pos for s in self.hunters_in_visual_radius]
                vec = [s[0].move for s in self.hunters_in_visual_radius]

                ad,sd,cd,rd = 1,1,1,1
                c,s,a, = self.calc(pos,vec)
            elif len(self.prey_in_visual_radius) > 0:
                pos = [s[0].pos for s in self.prey_in_visual_radius]
                vec = [s[0].move for s in self.prey_in_visual_radius]

                ad,sd,cd,rd = 0,0,1,0
                c,s,a, = self.calc(pos,vec)


            f_total = (ad * self.config.alignment_weight * a +
                       sd * self.config.separation_weight * s +
                       cd * self.config.cohesion_weight * c +
                       rd * self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.hunter_mass

            self.move += f_total
            self.pos += self.move * self.config.delta_time

        def change_position(self):
            # self.change_image(0)
            self.there_is_no_escape()
            if self.energy <= 1: self.kill()

            if self.is_alive():
                self.p_reproduce = 1/self.energy
                self.energy *= 0.94

                self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
                _prey_temp = list(self.in_proximity_accuracy().filter_kind(Prey))
                self.prey_in_visual_radius = list(filter(lambda x: x[-1] < self.config.hunter_visual_radius, _prey_temp))
                self.prey_in_eating_radius = list(filter(lambda x: x[-1] < self.config.hunter_eating_radius, _prey_temp))

                if len(self.prey_in_eating_radius) > 0:
                    # self.change_image(1)
                    self.prey_in_eating_radius[0][0].kill()
                    self.energy = min(300, self.energy+40)
                    if np.random.uniform() < self.p_reproduce:
                        # self.change_image(2)
                        self.reproduce()

                # self.change_image(1)
                self.random_move()


    class Prey(Agent):
        config: Conf
        def __init__(self,  *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.energy = 100
            self.age = 0
            self.max_age = 12
            self.p_reproduction = 0.008

        def _collect_replay_data(self):
            super()._collect_replay_data()
            self._Agent__simulation._metrics._temporary_snapshots["type"].append(0) # 0: prey

        def calc(self,pos,vec):
            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment
            return c,s,a

        def random_move(self):
            self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
            ad,sd,cd,rd = 0,0,0,1
            a,s,c = 0,0,0
            if len(self.hunters_in_visual_radius) > 0:
                pos = [s[0].pos for s in self.hunters_in_visual_radius]
                vec = [s[0].move for s in self.hunters_in_visual_radius]

                ad,sd,cd,rd = 0,1,0,0
                c,s,a, = self.calc(pos,vec)
            elif len(self.prey_in_visual_radius) > 0:
                pos = [s[0].pos for s in self.prey_in_visual_radius]
                vec = [s[0].move for s in self.prey_in_visual_radius]

                ad,sd,cd,rd = 1,1,1,1
                c,s,a, = self.calc(pos,vec)


            f_total = (ad * self.config.alignment_weight * a +
                       sd * self.config.separation_weight * s +
                       cd * self.config.cohesion_weight * c +
                       rd * self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.config.mass

            self.move += f_total
            self.pos += self.move * self.config.delta_time

        def change_position(self):
            self.there_is_no_escape()
            if self.energy == 0: self.kill()

            if self.is_alive():

                _temp_prey = list(self.in_proximity_accuracy().filter_kind(Prey))
                self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
                self.prey_in_visual_radius = list(filter(lambda x: x[-1] < self.config.prey_visual_radius, _temp_prey))

                prob = self.p_reproduction/(len(self.prey_in_visual_radius)) if len(self.prey_in_visual_radius) > 0 else self.p_reproduction
                if np.random.uniform() < prob:

                    self.reproduce()
                self.random_move()


    class Live(Simulation):
        config: Conf
        def tick(self, *args, **kwargs):
            super().tick(*args, **kwargs)
            hunter_count = len(list(filter(lambda x: isinstance(x,Hunter), list(self._agents.__iter__()))))
            if hunter_count == 0:
                frame = self.shared.counter
                self.stop()


    x, y = Conf().window.as_tuple()
    df = (
        Live(
            Conf(
                window= Window(500,500),
                fps_limit=0,
                duration=50000,
                movement_speed=1,
                image_rotation=True,
                print_fps=False,
                radius=30,
                seed=GLOBAL_SEED
            )
        )
            .batch_spawn_agents(500, Prey, images=["images/surfer.png"])
            .batch_spawn_agents(20, Hunter, images=["images/shark.png"])
            .run()
    )

    dfs = df.snapshots
    dfs.write_csv(f"X_{GLOBAL_SEED}.csv")

