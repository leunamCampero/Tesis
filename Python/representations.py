import sympy as sp


class MatrixRepresentation:
    def __init__(self, d, G, n):
        self.map = d
        self.group = G
        self.degree = n


def _characteristic_function(G, g, i, j):
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
        mydict[g] = sp.Matrix(n, n,
                              lambda i, j:
                              _characteristic_function(G, g, i, j))
    return MatrixRepresentation(mydict, G, n)
