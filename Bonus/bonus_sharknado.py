import gc

from numpy import average, random, linalg, log
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
    random_weight: float = 2

    delta_time: float = 2
    mass: int = 20
    radius: int = 30


    visual_bounds = [30,70]
    mass_bounds = [10,80]

    alpha: float = 0.1


class Hunter(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene = gen_gene() #get gene
        self.change_image(int(self.gene[0] * 10))  # change image to size

        self.max_age = random.randint(900,1200)
        self.age = 0

        self.mass = self.config.mass_bounds[0] + 70 * self.gene[0] #expression of mass gene f(x) = x/13 +0.3
        self.vision = self.config.visual_bounds[0] + 50 * self.gene[1] #expression of vision gene - former: visual_radius

        self.max_energy = self.mass ** log(self.mass/2)+200 #max energy
        self.energy = self.max_energy
        self.consumption = 0.97 * (0.01*(1-self.gene[0])+0.99)

        self.reach = self.vision / 1.8 #reach calulation - former: eating_radius
        self.speed = self.gene[1]*2 + 1*(1-self.gene[0]) #calcualtion of speed - WIP

        self.repr_age = 200#random.randint(100,50)
        self.p_reproduce = 0.3
        self.repr_cool = 0
        self.partner = None

    def _collect_replay_data(self):
        snapshots = self._Agent__simulation._metrics._temporary_snapshots
        snapshots["frame"].append(self.shared.counter)
        snapshots["type"].append(1)

        snapshots["mass"].append(int(self.gene[0] * 1000))  # place here the gene variables of the agent
        snapshots["vision"].append(int(self.gene[1] * 1000))

    def calc(self,pos,vec):
        c = (average(pos,axis = 0) - self.pos) - self.move #fc - vel --> coheison
        s = average([self.pos - x for x in pos], axis = 0) #seperation
        a = average(vec, axis = 0) - self.move #alignment
        return c,s,a

    def reproduce(self, other):
        for x in range(max(int(self.energy/30) , 1)):#range(random.choice(6,1,p=[0,0.3,0.4,0.15,0.1,0.05])[0]):
            random_uniform_coef_0 = random.normal(0, self.config.alpha)
            random_uniform_coef_1 = random.normal(0, self.config.alpha)
            random_noise_0 = random.normal(0, self.config.alpha/5)
            random_noise_1 = random.normal(0, self.config.alpha/5)

            child_genes = [None, None]

            child_genes[0] = min(1, max(0, random_uniform_coef_0 * (other.gene[0] - self.gene[0]) + self.gene[
                0] + random_noise_0))

            child_genes[1] = min(1, max(0, random_uniform_coef_1 * (other.gene[1] - self.gene[1]) + self.gene[
                1] + random_noise_1))

            child = copy(self)
            child.gene = child_genes


    def random_move(self):
        self.move = self.move / linalg.norm(self.move) if linalg.norm(self.move) > 0 else self.move

        ad,sd,cd,rd = 0,0,0,1 #activtor for a,s,c,r
        a,s,c = 0,0,0 #alignment, seperation, cohesion coefficients
        if len(self.hunters_in_visual_radius) > 0:
            pos = [s[0].pos for s in self.hunters_in_visual_radius]
            vec = [s[0].move for s in self.hunters_in_visual_radius]

            ad, sd, cd, rd = 1, 1, 1, 1
            c, s, a, = self.calc(pos, vec)
            if self.age < self.repr_age:
                ad, sd, cd, rd = 0, 1, 0, 0
            elif next((x for x in self.hunters_in_visual_radius if x[0].repr_cool == 0), None) and self.repr_cool == 0:
                ad, sd, cd, rd = 0, 0, 1, 0

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
        if self.energy <= 1 or self.age > self.max_age: self.kill()

        if self.is_alive():
            self.age += 1
            self.energy *= self.consumption
            self.repr_cool = max(0, self.repr_cool-1)
            if self.repr_cool == 1:
                self.change_image(int(self.gene[0] * 10) - 1)
                self.reproduce(self.partner)


            self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
            _prey_temp = list(self.in_proximity_accuracy().filter_kind(Prey))
            self.prey_in_visual_radius = list(filter(lambda x: x[-1] < self.vision, _prey_temp))
            prey_in_eating_radius = list(filter(lambda x: x[-1] < self.reach, _prey_temp))

            if len(prey_in_eating_radius) > 0:
                prey_in_eating_radius[0][0].kill()
                self.energy = min(self.max_energy, self.energy+40)


            if len(self.hunters_in_visual_radius) > 0 and self.repr_cool == 0 \
                    and random.uniform() < self.p_reproduce \
                    and self.hunters_in_visual_radius[0][0].repr_cool == 0 \
                    and self.age > self.repr_age \
                    and self.hunters_in_visual_radius[0][0].age > self.hunters_in_visual_radius[0][0].repr_age:
                #self.change_image(int(self.gene[0] * 10) + 9)
                self.partner = self.hunters_in_visual_radius[0][0] if self.partner == None else self.partner
                self.hunters_in_visual_radius[0][0].partner = self if self.hunters_in_visual_radius[0][0].partner == None else self.hunters_in_visual_radius[0][0].partner
                #self.hunters_in_visual_radius[0][0].repr_cool = random.randint(200,400)
                self.repr_cool = random.randint(150,300)
                self.change_image(int(self.gene[0] * 10) + 8)
            self.random_move()


class Prey(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_reproduction = 0.03
        self.visual_radius = self.config.radius

    def _collect_replay_data(self):
        snapshots = self._Agent__simulation._metrics._temporary_snapshots
        snapshots["frame"].append(self.shared.counter)
        snapshots["type"].append(0)

        snapshots["mass"].append(0)  # place here the gene variables of the agent
        snapshots["vision"].append(0)

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

            ad,sd,cd,rd = 0,0.8,0,0
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
birds = [f"images/bird_{x}.png" for x in range(20)]
#green_birds = [f"images/bird_green_{x}.png" for x in range(10)] #list of all bird sprites

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
            .batch_spawn_agents(200, Prey, images=["images/white.png"])
            .batch_spawn_agents(20, Hunter, images=birds)
            .run()
    )
    dfs = df.snapshots
    dfs.write_csv(f"X_sexual_{GLOBAL_SEED}.csv")
    gc.collect()


