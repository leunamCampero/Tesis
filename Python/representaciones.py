from sympy import Matrix
from sympy.combinatorics.named_groups import SymmetricGroup


def characteristic_function(G, g, i, j):
    elems = list(G.elements)
    if g*elems[i] == elems[j]:
        return 1
    else:
        return 0


def regular_representation(G):
    elems = list(G.elements)
    n = len(elems)
    mydict = {}
    for g in elems:
        mydict[g] = Matrix(n, n,
                           lambda i, j:
                           characteristic_function(G, g, i, j))
    return mydict


H = SymmetricGroup(2)

repr = regular_representation(H)
print(repr)
