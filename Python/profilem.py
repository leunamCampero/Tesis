# profilem.py

n=7

import pstats
import cProfile
import representations
from sympy.combinatorics import Permutation

G1 = representations.grafica_de_emparejamiento(6)
G = representations.clique_graph(G1)
sc = representations.SimplicialComplex(G)

cProfile.runctx("sc.specific_function(n)",
                globals(), locals(), "Profilem.prof")

s = pstats.Stats("Profilem.prof")
s.strip_dirs().sort_stats("time").print_stats()
