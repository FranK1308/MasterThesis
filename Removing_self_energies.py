#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ase.io import read, write
from ase.db.core import connect
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

PERIODIC_TABLE = ['Dummy'] + """
    H                                                                                                                           He
    Li  Be                                                                                                  B   C   N   O   F   Ne
    Na  Mg                                                                                                  Al  Si  P   S   Cl  Ar
    K   Ca  Sc                                                          Ti  V   Cr  Mn  Fe  Co  Ni  Cu  Zn  Ga  Ge  As  Se  Br  Kr
    Rb  Sr  Y                                                           Zr  Nb  Mo  Tc  Ru  Rh  Pd  Ag  Cd  In  Sn  Sb  Te  I   Xe
    Cs  Ba  La  Ce  Pr  Nd  Pm  Sm  Eu  Gd  Tb  Dy  Ho  Er  Tm  Yb  Lu  Hf  Ta  W   Re  Os  Ir  Pt  Au  Hg  Tl  Pb  Bi  Po  At  Rn
    Fr  Ra  Ac  Th  Pa  U   Np  Pu  Am  Cm  Bk  Cf  Es  Fm  Md  No  Lr  Rf  Db  Sg  Bh  Hs  Mt  Ds  Rg  Cn  Nh  Fl  Mc  Lv  Ts  Og
    """.strip().split()
# In[ ]:


total = read('/home/frank/multif/egnn/data/ani1ccx.extxyz', index = ':',format='extxyz')


# In[ ]:


print(len(total))


# In[ ]:


list_of_dicts = []
for i in range(len(total)):
    temp_dict = {}
    temp_dict['species'] = total[i].get_chemical_symbols()
    temp_dict['energies'] = total[i].get_potential_energy()
    list_of_dicts.append(temp_dict)


# In[ ]:


from collections import Counter
import numpy as np

def subtract_self_energies(reenterable_iterable, self_energies=None, species_order=None, fit_intercept=True):
    if self_energies is None:
        self_energies = {}
        
    if species_order is None:
        species_order = PERIODIC_TABLE

    counts = {}
    Y = []
    for n, d in enumerate(reenterable_iterable):
        species = d['species']
        count = Counter()
        for s in species:
            count[s] += 1
        for s, c in count.items():
            if s not in counts:
                counts[s] = [0] * n
            counts[s].append(c)
        for s in counts:
            if len(counts[s]) != n + 1:
                counts[s].append(0)
        Y.append(d['energies'].item())

    species = sorted(list(counts.keys()), key=lambda x: species_order.index(x))
    X = [counts[s] for s in species]

    if fit_intercept:
        X.append([1] * (n + 1))

    X_np = np.array(X).transpose()
    Y = np.array(Y)

    sae, _, _, _ = np.linalg.lstsq(X_np, Y, rcond=None)

    intercept = 0
    if fit_intercept:
        intercept = sae[-1]
        sae = sae[:-1]

    for s, e in zip(species, sae):
        self_energies[s] = e
    print(self_energies,intercept)

    # Subtract self energies from each example
    subtracted_energies = []
    for d in reenterable_iterable:
        species = d['species']
        count = Counter(species)
        subtracted_energy = d['energies'] - sum([count[s] * self_energies[s] for s in count]) - intercept
        subtracted_energies.append(subtracted_energy)

    return subtracted_energies


# In[ ]:


en_nakon = subtract_self_energies(list_of_dicts,fit_intercept=False)


# In[ ]:


for i in range(len(total)):
    total[i].calc.results['energy'] = en_nakon[i]


# In[ ]:


for i in total:
    write('/home/frank/multif/egnn/data/subtracted_ani1ccx.extxyz', i, format='extxyz',append=True)

