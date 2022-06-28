import gc
import pickle
#import numpy as np

from numpy import average, random, linalg
from copy import copy
from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize, Window



def gen_gene(): #gen random gene
    return random.uniform(size = 2)

@deserialize
@dataclass
class Conf(Config):
    #base values for all agents
    #do not change
    alignment_weight: float = 0.50
    cohesion_weight: float = 0.2
    separation_weight: float = 0.25
    random_weight: float = 1.3

    delta_time: float = 2
    mass: int = 20
    radius: int = 30


    visual_bounds = [10,70]
    mass_bounds = [10,80]

    alpha: float = 0.1


class Hunter(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene = gen_gene() #get gene
        self.change_image(int(self.gene[0] * 10))  # change image to size
        self.id = int(f"{self.id}{''.join([str(int(x*100)).zfill(3) for x in self.gene])}") #record gene in df


        self.mass = self.config.mass_bounds[0] + 60 * self.gene[0] #expression of mass gene f(x) = x/13 +0.3
        self.vision = self.config.visual_bounds[0] + 60 * self.gene[1] #expression of vision gene - former: visual_radius

        self.max_energy = self.mass * 6 #max energy
        self.energy = self.max_energy
        self.repr_energy = int(self.max_energy*0.70)-20
        self.consumption = 0.95 * (0.2*self.gene[0]+0.9)

        self.reach = self.vision / 1.8 #reach calulation - former: eating_radius
        self.speed = self.gene[1]*2 + 1*self.gene[0] #calcualtion of speed - WIP


        self.repr_cool = 0
        self.partner = None

    def _collect_replay_data(self):
        snapshots = self._Agent__simulation._metrics._temporary_snapshots
        snapshots["frame"].append(self.shared.counter)
        snapshots["id"].append(self.id)
        snapshots["image_index"].append(self._image_index)

        self._Agent__simulation._metrics._temporary_snapshots["type"].append(1)  # 1: hunter

    def calc(self,pos,vec):
        c = (average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
        s = average([self.pos - x for x in pos], axis = 0) #seperation
        a = average(vec, axis = 0) - self.move #alignment
        return c,s,a

    def reproduce(self, other):
        random_uniform_coef = random.uniform(-self.config.alpha, 1 + self.config.alpha)
        child_genes = [None, None]

        child_genes[0] = min(self.config.mass_bounds[1],
                             max(self.config.mass_bounds[0],
                                 random_uniform_coef * (other.gene[0] - self.gene[0]) + self.gene[0]))

        child_genes[1] = min(self.config.visual_bounds[1],
                             max(self.config.visual_bounds[0],
                                 random_uniform_coef * (other.gene[1] - self.gene[1]) + self.gene[1]))

        child = copy(self)
        child.gene = child_genes


    def random_move(self):
        self.move = self.move / linalg.norm(self.move) if linalg.norm(self.move) > 0 else self.move

        ad,sd,cd,rd = 0,0,0,1 #activtor for a,s,c,r
        a,s,c = 0,0,0 #alignment, seperation, cohesion coefficients
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
                   rd * self.config.random_weight * random.uniform(low = -1, high = 1, size = 2)) / self.mass

        self.move += f_total
        self.pos += self.move * self.speed

    def change_position(self):
        self.there_is_no_escape()
        if self.energy <= 1: self.kill()

        if self.is_alive():
            self.energy *= 0.96
            self.repr_cool = max(0, self.repr_cool-1)
            if self.repr_cool == 1:
                self.reproduce(self.partner)


            self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
            _prey_temp = list(self.in_proximity_accuracy().filter_kind(Prey))
            self.prey_in_visual_radius = list(filter(lambda x: x[-1] < self.vision, _prey_temp))
            prey_in_eating_radius = list(filter(lambda x: x[-1] < self.reach, _prey_temp))

            if len(prey_in_eating_radius) > 0:
                prey_in_eating_radius[0][0].kill()
                self.energy = min(self.max_energy, self.energy+40)


            if len(self.hunters_in_visual_radius) > 0 \
                    and self.repr_cool == 0 \
                    and self.energy >= self.repr_energy \
                    and self.hunters_in_visual_radius[0][0].energy >= self.hunters_in_visual_radius[0][0].repr_energy:

                self.partner = self.hunters_in_visual_radius[0][0] if self.partner == None else self.partner
                self.repr_cool = random.randint(80,90)
                #self.change_image(int(self.gene[0] * 10) + 10)


            self.random_move()


class Prey(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_reproduction = 0.008
        self.visual_radius = self.config.radius

    def _collect_replay_data(self):
        snapshots = self._Agent__simulation._metrics._temporary_snapshots
        snapshots["frame"].append(self.shared.counter)
        snapshots["id"].append(self.id)
        snapshots["image_index"].append(self._image_index)

        self._Agent__simulation._metrics._temporary_snapshots["type"].append(0)  # 0: prey

    def calc(self,pos,vec):
        c = (average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
        s = average([self.pos - x for x in pos], axis = 0) #seperation
        a = average(vec, axis = 0) - self.move #alignment
        return c,s,a

    def random_move(self):
        self.move = self.move / linalg.norm(self.move) if linalg.norm(self.move) > 0 else self.move

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
                   rd * self.config.random_weight * random.uniform(low = -1, high = 1, size = 2)) / self.config.mass

        self.move += f_total
        self.pos += self.move * self.config.delta_time

    def change_position(self):
        self.there_is_no_escape()

        if self.is_alive():
            self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
            self.prey_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Prey))

            prob = self.p_reproduction/(len(self.prey_in_visual_radius)) if len(self.prey_in_visual_radius) > 0 else self.p_reproduction

            if random.uniform() < prob: self.reproduce()

            self.random_move()


class Live(Simulation):
    config: Conf
    def tick(self, *args, **kwargs):
        super().tick(*args, **kwargs)
        hunter = list(filter(lambda x: isinstance(x, Hunter), list(self._agents.__iter__())))
        hunter_count = len(hunter)
        if hunter_count == 0 or (hunter_count == 1 and hunter[0].repr_cool == 0):
            self.stop()


x, y = Conf().window.as_tuple()
birds = [f"images/bird_{x}.png" for x in range(10)] #list of all bird sprites

#if adding pregnancy image change:
#preg_birds = [f"images/bird_{x}p.png" for x in range(10)] #list of all bird sprites]
#birds.append(preg_birds)
for i in range(5):
    GLOBAL_SEED = random.randint(0,1000000)
    df = (
        Live(
            Conf(
                window= Window(500,500),
                fps_limit=0,
                movement_speed=1,
                #image_rotation=True,
                print_fps=False,
                radius=30,
                seed=GLOBAL_SEED
            )
        )
            .batch_spawn_agents(100, Prey, images=["images/white.png"])
            .batch_spawn_agents(20, Hunter, images=birds)
            .run()
    )
    dfs = df.snapshots
    dfs.write_csv(f"X_{GLOBAL_SEED}.csv")
    gc.collect()


