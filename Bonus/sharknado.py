from enum import Enum, auto
from sre_parse import expand_template

import numpy as np
import pygame as pg
import pygame.camera as pgc
from pygame.math import Vector2
from vi import Agent, Simulation, HeadlessSimulation
from copy import copy
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

    hunter_visual_radius = 30
    hunter_eating_radius = 17
    prey_visual_radius = 30
    alpha = 0.1


class Food(Agent):
    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["type"].append(2) # 2: food


class Hunter(Agent):
    config: Conf
    REPRODUCE_THRESH = 200
    REPRODUCE_COOLDOWN = 200
    EXPENDITURE_BASE_COEF = 1.  # base expenditure
    EXP_MATING = 1.1            # mating coef exp
    EXP_CHASE = 2.              # chasing expenditure coef
    EXP_IDENTITY_MASS = 15      # Mass for which the penalty for mass is 1.
    EXP_IDENTITY_RADIUS = 30
    GENE_BOUNDS = [(7, 80), (1, 60)]
    
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 75
        self.energy_expenditure = Hunter.EXPENDITURE_BASE_COEF
        self.age = 0
        self.max_age = 14
        self.p_reproduce = 0.2
        self.can_reproduce = False
        self.reproduce_cooldown = Hunter.REPRODUCE_COOLDOWN
        self.genes = [np.random.uniform(Hunter.GENE_BOUNDS[0][0], Hunter.GENE_BOUNDS[0][1]),
                        np.random.uniform(Hunter.GENE_BOUNDS[1][0], Hunter.GENE_BOUNDS[1][1])]
        self.hunter_mass = self.genes[0]
        self.visual_radius = self.genes[1]
        self.exp_mass = Hunter.EXP_IDENTITY_MASS/self.hunter_mass 
        self.exp_sensing = Hunter.EXP_IDENTITY_RADIUS/self.visual_radius
        self.exp_from_genes = self.exp_sensing * self.exp_mass

    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["type"].append(1) # 1: hunter
        self._Agent__simulation._metrics._temporary_snapshots["mass"].append(self.genes[0])
        self._Agent__simulation._metrics._temporary_snapshots["vision"].append(self.genes[1])

    def reproduce(self, other):
        random_uniform_coef = np.random.uniform(-self.config.alpha, 1+self.config.alpha)
        child_genes = [None, None]
        child_genes[0] = min(Hunter.GENE_BOUNDS[0][1], max(Hunter.GENE_BOUNDS[0][0], random_uniform_coef*(other.genes[0]-self.genes[0])+self.genes[0]))
        random_uniform_coef = np.random.uniform(-self.config.alpha, 1+self.config.alpha)
        child_genes[1] = min(Hunter.GENE_BOUNDS[1][1], max(Hunter.GENE_BOUNDS[1][0], random_uniform_coef*(other.genes[1]-self.genes[1])+self.genes[1]))
        child = copy(self)
        child.genes = child_genes
        child.energy = 75
        return child

    def random_move(self):
        self.energy_expenditure = Hunter.EXPENDITURE_BASE_COEF
        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move
        
        if len(self.hunters_in_visual_radius) and \
         next((x for x in self.hunters_in_visual_radius if x[0].can_reproduce), None) and \
          self.can_reproduce: # alignment and cohesion with other hunters
                pos = [s[0].pos for s in self.hunters_in_visual_radius]
                vec = [s[0].move for s in self.hunters_in_visual_radius]

                c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
                #s = np.average([self.pos - x for x in pos], axis = 0) #seperation
                #a = np.average(vec, axis = 0) - self.move #alignment

                f_total = (#self.config.alignment_weight * a +
                        #self.config.separation_weight * s +
                        self.config.cohesion_weight * c)#+
                        #self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.hunter_mass

                self.energy_expenditure *= Hunter.EXP_MATING * self.exp_from_genes
        elif len(self.prey_in_visual_radius) > 0: # alignment and cohesion with prey
            pos = [s[0].pos for s in self.prey_in_visual_radius]
            vec = [s[0].move for s in self.prey_in_visual_radius]

            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            #s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            #a = np.average(vec, axis = 0) - self.move #alignment

            f_total = (#0 * a + # alignment
                       #0 * s + # separation
                       1 * c)/self.hunter_mass #+ # cohesion
                       #0 * np.random.uniform(low = -1, high = 1, size = 2)) / self.hunter_mass

            self.energy_expenditure *= Hunter.EXP_CHASE * self.exp_from_genes
        elif len(self.hunters_in_visual_radius) > 0: # alignment and cohesion with other hunters
            pos = [s[0].pos for s in self.hunters_in_visual_radius]
            vec = [s[0].move for s in self.hunters_in_visual_radius]

            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment

            f_total = (self.config.alignment_weight * a +
                       self.config.separation_weight * s +
                       self.config.cohesion_weight * c +
                       self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.hunter_mass
            self.energy_expenditure *= self.exp_from_genes
        else: 
            f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.hunter_mass
            self.energy_expenditure *= self.exp_from_genes

        self.move += f_total
        self.pos += self.move * self.config.delta_time

    def change_position(self):
        # self.change_image(0)
        self.there_is_no_escape()
        if self.energy <= 1: self.kill()

        if self.is_alive():
            #self.p_reproduce = 1/self.energy
            self.hunters_in_visual_radius = list(filter(lambda x: x[-1] < self.visual_radius,
                                                         list(self.in_proximity_accuracy().filter_kind(Hunter))))
            self._prey_temp = list(self.in_proximity_accuracy().filter_kind(Prey))
            self.prey_in_visual_radius = list(filter(lambda x: x[-1] < self.visual_radius, self._prey_temp))
            self.prey_in_eating_radius = list(filter(lambda x: x[-1] < self.config.hunter_eating_radius, self._prey_temp))
            self.hunters_in_repr_radius = list(filter(lambda x: x[-1] < self.config.hunter_eating_radius, self.hunters_in_visual_radius))    

            if len(self.hunters_in_repr_radius):
                reproduce_with = next((x[0] for x in self.hunters_in_repr_radius if x[0].can_reproduce), None)

                if self.energy >= Hunter.REPRODUCE_THRESH and self.can_reproduce and reproduce_with:
                    self.reproduce(reproduce_with)
                    self.reproduce_cooldown = Hunter.REPRODUCE_COOLDOWN
                    self.can_reproduce = False
                    self.change_image(0)
                    reproduce_with.can_reproduce = False
                    reproduce_with.reproduce_cooldown = Hunter.REPRODUCE_COOLDOWN


            if len(self.prey_in_eating_radius) > 0:
                # self.change_image(1)
                self.prey_in_eating_radius[0][0].kill()
                self.energy = min(300, self.energy+50)
            # self.change_image(1)
            self.random_move()
            self.energy -= self.energy_expenditure
            if not self.can_reproduce:
                self.reproduce_cooldown -= 1
            if self.reproduce_cooldown <= 0 and self.energy >= Hunter.REPRODUCE_THRESH:
                self.change_image(1)
                self.can_reproduce = True
            else:
                self.change_image(0)

class Prey(Agent):
    config: Conf
    EXP_IDENTITY_MASS = 20. # Mass for which the penalty for mass is 1.
    EXP_BASE = 1. # Base expendityre
    EXP_FLEE = 2. # expenditure coef for fleeing
    EXP_IDENTITY_RADIUS = 30
    NORMAL_COEF = 2 # Normal noise for mutation std
    GENE_BOUNDS = [(4, 60), (1, 60)]
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.energy = 300
        self.energy_expenditure = Prey.EXP_BASE
        self.age = 0
        self.max_age = 12
        self.p_reproduction = 0.009
        self.genes = [np.random.uniform(Prey.GENE_BOUNDS[0][0], Prey.GENE_BOUNDS[0][1]),
                        np.random.uniform(Prey.GENE_BOUNDS[1][0], Prey.GENE_BOUNDS[1][1])]
        self.prey_mass = self.genes[0]
        self.visual_radius = self.genes[1]
        self.exp_mass = Hunter.EXP_IDENTITY_MASS/self.prey_mass
        self.exp_sensing = Hunter.EXP_IDENTITY_RADIUS/self.visual_radius
        self.exp_from_genes = self.exp_sensing * self.exp_mass

    def _collect_replay_data(self):
        super()._collect_replay_data()
        self._Agent__simulation._metrics._temporary_snapshots["type"].append(0) # 0: prey
        self._Agent__simulation._metrics._temporary_snapshots["mass"].append(self.genes[0])
        self._Agent__simulation._metrics._temporary_snapshots["vision"].append(self.genes[1])


    def reproduce(self):
        random_normal_coef = np.random.normal(scale=Prey.NORMAL_COEF)
        child_genes = [None, None]
        child_genes[0] =  min(Prey.GENE_BOUNDS[0][1], max(Prey.GENE_BOUNDS[0][0], self.genes[0] + random_normal_coef))
        random_normal_coef = np.random.normal(scale=Prey.NORMAL_COEF)
        child_genes[1] = min(Prey.GENE_BOUNDS[1][1], max(Prey.GENE_BOUNDS[1][0], self.genes[1] + random_normal_coef))
        child = copy(self)
        child.genes = child_genes
        child.energy = 75
        return child

    def random_move(self):
        self.energy_expenditure = Prey.EXP_BASE
        self.move = self.move / np.linalg.norm(self.move) if np.linalg.norm(self.move) > 0 else self.move

        if len(self.hunters_in_visual_radius) > 0:
            pos = [s[0].pos for s in self.hunters_in_visual_radius]
            vec = [s[0].move for s in self.hunters_in_visual_radius]

            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment

            f_total = (#0 * a + # align
                       1 * s  # separate
                       #0 * c + # coheise
                       #0 * np.random.uniform(low = -1, high = 1, size = 2)))
                      ) / self.prey_mass
            self.energy_expenditure *= Prey.EXP_FLEE * Prey.EXP_IDENTITY_MASS / self.prey_mass

        elif len(self.prey_in_visual_radius) > 0:
            pos = [s[0].pos for s in self.prey_in_visual_radius]
            vec = [s[0].move for s in self.prey_in_visual_radius]

            c = (np.average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
            s = np.average([self.pos - x for x in pos], axis = 0) #seperation
            a = np.average(vec, axis = 0) - self.move #alignment

            f_total = (self.config.alignment_weight * a +
                       self.config.separation_weight * s +
                       self.config.cohesion_weight * c +
                       self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.prey_mass
        
            self.energy_expenditure *= Prey.EXP_IDENTITY_MASS / self.prey_mass
        else: 
            f_total = (self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.prey_mass
            self.energy_expenditure *= Prey.EXP_IDENTITY_MASS / self.prey_mass
        self.move += f_total

        self.pos += self.move * self.config.delta_time


    def change_position(self):
        self.there_is_no_escape()
        if self.energy == 0: self.kill()

        if self.is_alive():
            # self.change_image(0)
            self._temp_prey = list(self.in_proximity_accuracy().filter_kind(Prey))
            self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
            self.prey_in_visual_radius = list(filter(lambda x: x[-1] < self.config.prey_visual_radius, self._temp_prey))

            prob = self.p_reproduction/(len(self.prey_in_visual_radius)) if len(self.prey_in_visual_radius) > 0 else self.p_reproduction
            if np.random.uniform() < prob:
                # self.change_image(2)
                self.reproduce()
            self.random_move()
            self.energy -= self.energy_expenditure


class Live(Simulation):
    config: Conf
    def tick(self, *args, **kwargs):
        super().tick(*args, **kwargs)
        if self.shared.counter % 200 == 0:
            print(self.shared.counter)

        agents = list(self._agents.__iter__())
        prey_count = len(list(filter(lambda x: isinstance(x,Prey), agents)))
        hunter_count = len(list(filter(lambda x: isinstance(x,Hunter), agents)))
        if prey_count == 0 or hunter_count == 0:
            print(f'Simulation stopped because no {"prey" if prey_count == 0 else "hunter"} was left.')
            self.stop()


x, y = Conf().window.as_tuple()
df = (
    Live(
        Conf(
            window= Window(500,500),
            fps_limit=0,
            duration=20000,
            movement_speed=1,
            image_rotation=True,
            print_fps=False,
            radius=30,
            seed=GLOBAL_SEED
        )
    )
        .batch_spawn_agents(500, Prey, images=["images/surfer.png"])
        .batch_spawn_agents(20, Hunter, images=["images/shark.png", "images/shark_green.png"])
        .run()
)

dfs = df.snapshots
dfs.write_csv(f"X_{GLOBAL_SEED}.csv")