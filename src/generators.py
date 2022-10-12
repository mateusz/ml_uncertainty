import scipy.stats as stats
import random
import pandas as pd

class Resource:
    def __init__(self, id, loc, scale, m, seed):
        self.id = id
        self.loc = loc
        self.scale = scale
        self.m = m
        self.seed = seed

    def generate(self):
        self.seed += 1
        return stats.norm.rvs(size=self.m, loc=self.loc, scale=self.scale, random_state=self.seed)

class NormalGenerator:
    def __init__(
        self,
        seed=0,
        r=1000,
        r_loc=(-5,2),
        r_scale=(0.1,1),
        m_a=1,
        m_loc=1,
        m_scale=3
    ):
        random.seed(seed)
        ms = stats.gamma.rvs(size=r, a=m_a, loc=m_loc, scale=m_scale, random_state=seed)
        self.resources = []
        for r in range(0, r, 1):
            self.resources.append(
                Resource(
                    id='/%s' % r,
                    loc=random.uniform(*r_loc),
                    scale=random.uniform(*r_scale),
                    m=round(ms[r]),
                    seed=seed
                    )
                )

    def generate(
        self,
        n=100000,
    ):
        collected = []
        count = 0
        while count<n:
            for resource in self.resources:
                if count>n:
                    break
                
                gen = resource.generate()
                count += len(gen)
                for g in gen:
                    collected.append([resource.id, g])

        return pd.DataFrame(collected, columns=['id', 'val'])