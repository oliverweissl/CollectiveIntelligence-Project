from enum import Enum, auto

import numpy as np
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize


@deserialize
@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.75
    cohesion_weight: float = 0.4
    separation_weight: float = 0.4
    random_weight: float = 0.5

    delta_time: float = 3

    mass: int = 20

    def weights(self) -> tuple[float, float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight, self.random_weight)


class Bird(Agent):
    config: FlockingConfig

    def change_position(self):
        # Pac-man-style teleport to the other end of the screen when trying to escape
        self.there_is_no_escape()

        #YOUR CODE HERE -----------
        n = list(self.in_proximity_accuracy()) #list of neighbors
        if len(n) > 0: #if we have n
            pos = [s[0].pos for s in n] #positions of n
            vec = [s[0].move for s in n]

            if np.arccos(np.dot(np.average(vec,axis = 0), self.move))< 0.5:
                self.change_image(1)
            else: self.change_image(0)

            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment


            f_total = (self.config.alignment_weight * a +
                       self.config.separation_weight * s +
                       self.config.cohesion_weight * c +
                       self.config.random_weight * np.random.uniform(low = -0.5, high = 0.5, size = 2)) / self.config.mass

            self.move += f_total  # update move angle and velocity
        else: self.change_image(0)

        coll = list(self.obstacle_intersections(scale = 2))
        if len(coll) > 0:
            for c in coll:
                nm = self.move-(c-self.pos) #current move velocity - distance to the obstacle
                self.move = nm / np.linalg.norm(nm) #normalize vector

        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) < self.config.movement_speed else self.move
        self.move = pg.Vector2((self.move[0],self.move[1]))
        self.pos += self.move * self.config.delta_time #update pos
        #END CODE -----------------


class Selection(Enum):
    ALIGNMENT = auto()
    COHESION = auto()
    SEPARATION = auto()
    RANDOM = auto()


class FlockingLive(Simulation):
    selection: Selection = Selection.ALIGNMENT
    config: FlockingConfig

    def handle_event(self, by: float):
        if self.selection == Selection.ALIGNMENT:
            self.config.alignment_weight += by
        elif self.selection == Selection.COHESION:
            self.config.cohesion_weight += by
        elif self.selection == Selection.SEPARATION:
            self.config.separation_weight += by
        elif self.selection == Selection.RANDOM:
            self.config.random_weight += by

    def before_update(self):
        super().before_update()

        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.handle_event(by=0.05)
                elif event.key == pg.K_DOWN:
                    self.handle_event(by=-0.05)
                elif event.key == pg.K_1:
                    self.selection = Selection.ALIGNMENT
                elif event.key == pg.K_2:
                    self.selection = Selection.COHESION
                elif event.key == pg.K_3:
                    self.selection = Selection.SEPARATION
                elif event.key == pg.K_4:
                    self.selection = Selection.RANDOM

        a, c, s, w = self.config.weights()
        print(f"A: {a:.2f} - C: {c:.2f} - S: {s:.2f} - W: {w:.2f}")


x, y = FlockingConfig().window.as_tuple()
df = (
    FlockingLive(
        FlockingConfig(
            duration=10*60,
            image_rotation=True,
            movement_speed=1,
            radius=50,
            seed=1,
        )
    )
    .spawn_obstacle("images/bubble-full.png", x // 2, y // 2)
    .batch_spawn_agents(50, Bird, images=["images/bird.png","images/bird_red.png"])
    .run()
)


dfs= df.snapshots
dfs.write_csv(f"A{FlockingConfig.alignment_weight:.2f}_C{FlockingConfig.cohesion_weight:.2f}_S{FlockingConfig.separation_weight:.2f}_W{FlockingConfig.random_weight:.2f}.csv")