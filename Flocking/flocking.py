from enum import Enum, auto

import numpy as np
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize


@deserialize
@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 0.7
    cohesion_weight: float = 0.55
    separation_weight: float = 0.5
    random_weigth: float = 0.01

    delta_time: float = 3

    mass: int = 20

    def weights(self) -> tuple[float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight)


class Bird(Agent):
    config: FlockingConfig

    def change_position(self):

        # Pac-man-style teleport to the other end of the screen when trying to escape
        self.there_is_no_escape()

        #YOUR CODE HERE -----------


        n = list(self.in_proximity_accuracy()) #list of neighbors
        if len(n) > 0: #if we have n
            pos = [s[0].pos for s in n] #positions of n

            coheison = (np.average(pos,axis = 0) -self.pos) - self.move #fc - vel
            seperation = np.average([self.pos - x for x in pos], axis = 0)
            alignment = np.average([s[0].move for s in n], axis = 0) - self.move

            f_total = (self.config.alignment_weight * alignment +
                       self.config.separation_weight * seperation +
                       self.config.cohesion_weight * coheison +
                       self.config.random_weigth * np.random.random((2))) / self.config.mass

            self.move += f_total  # update move angle and velocity
            self.move = self.move / np.linalg.norm(self.move) if self.move[1] < self.config.movement_speed else self.move

        self.pos += self.move * self.config.delta_time #update pos




        #END CODE -----------------


class Selection(Enum):
    ALIGNMENT = auto()
    COHESION = auto()
    SEPARATION = auto()


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

    def before_update(self):
        super().before_update()

        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.handle_event(by=0.1)
                elif event.key == pg.K_DOWN:
                    self.handle_event(by=-0.1)
                elif event.key == pg.K_1:
                    self.selection = Selection.ALIGNMENT
                elif event.key == pg.K_2:
                    self.selection = Selection.COHESION
                elif event.key == pg.K_3:
                    self.selection = Selection.SEPARATION

        a, c, s = self.config.weights()
        print(f"A: {a:.1f} - C: {c:.1f} - S: {s:.1f}")


(
    FlockingLive(
        FlockingConfig(
            image_rotation=True,
            movement_speed=1,
            radius=50,
            seed=1,
        )
    )
    .batch_spawn_agents(50, Bird, images=["images/bird.png"])
    .run()
)
