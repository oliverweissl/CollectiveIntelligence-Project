import gc
import pickle
import numpy as np

from vi import Agent, Simulation, HeadlessSimulation
from vi.config import Config, dataclass, deserialize, Window



def gen_gene(): #gen random gene
    return np.random.uniform(size = 2)

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


class Hunter(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gene = gen_gene() #get gene
        self.id = int(f"{self.id}{''.join([str(x) for x in self.gene])}") #record gene in df

        self.mass = self.config.mass*(self.gene[0]/13 +0.3) #expression of mass gene f(x) = x/13 +0.3
        self.vision = self.config.radius*(self.gene[1]/13 +0.3) #expression of vision gene - former: visual_radius



        #self.reach = self.vision / (self.size/30+0.3) #reach calulation - former: eating_radius
        #self.energy = self.mass * 6 #max energy
        #self.change_image(self.gene[2]) #change image to size
        #self.speed = self.config.delta_time * self.strength / np.sqrt((self.mass + self.size)/10) #calcualtion of speed - WIP


        self.p_reproduce = 0.15

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
                   rd * self.config.random_weight * np.random.uniform(low = -1, high = 1, size = 2)) / self.mass

        self.move += f_total
        self.pos += self.move * self.speed

    def change_position(self):
        self.there_is_no_escape()
        if self.energy <= 1: self.kill()

        if self.is_alive():
            self.p_reproduce = 1/self.energy
            self.energy *= 0.94

            self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
            _prey_temp = list(self.in_proximity_accuracy().filter_kind(Prey))
            self.prey_in_visual_radius = list(filter(lambda x: x[-1] < self.vision, _prey_temp))
            prey_in_eating_radius = list(filter(lambda x: x[-1] < self.reach, _prey_temp))

            if len(prey_in_eating_radius) > 0:
                prey_in_eating_radius[0][0].kill()
                self.energy = min(300, self.energy+40)
                if np.random.uniform() < self.p_reproduce:
                    self.reproduce()

            self.random_move()


class Prey(Agent):
    config: Conf
    def __init__(self,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_reproduction = 0.008
        self.visual_radius = self.config.radius

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

        if self.is_alive():
            self.hunters_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Hunter))
            self.prey_in_visual_radius = list(self.in_proximity_accuracy().filter_kind(Prey))

            prob = self.p_reproduction/(len(self.prey_in_visual_radius)) if len(self.prey_in_visual_radius) > 0 else self.p_reproduction

            if np.random.uniform() < prob: self.reproduce()

            self.random_move()


class Live(Simulation):
    config: Conf
    def tick(self, *args, **kwargs):
        global counter_t
        super().tick(*args, **kwargs)
        counter_t += 1
        hunter_count = len(list(filter(lambda x: isinstance(x, Hunter), list(self._agents.__iter__()))))
        if hunter_count == 0:
            self.stop()

frame_counter = []
x, y = Conf().window.as_tuple()
birds = [f"images/bird_{x}.png" for x in range(10)] #list of all bird sprites
for i in range(5):
    GLOBAL_SEED = np.random.randint(0,1000000)

    counter_t = 0
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
            .batch_spawn_agents(500, Prey, images=["images/white.png"])
            .batch_spawn_agents(20, Hunter, images=birds)
            .run()
    )

    frame_counter.append([GLOBAL_SEED,counter_t])
    dfs = df.snapshots
    dfs.write_csv(f"X_{GLOBAL_SEED}.csv")
    with open(f"framecount", "wb") as fp:
        pickle.dump(frame_counter, fp)
    gc.collect()


