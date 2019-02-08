import sympy

from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.physics.quantum.dagger import Dagger


def MTS(A):
    """Calcula una matriz triangular superior"""
    A1 = A
    n = A.shape[0]
    V = sympy.eye(n)
    for i in range(0, n):
        C = sympy.eye(n)
        C[i, i] = 1/sympy.sqrt(A1[i, i])
        for j in range(i+1, n):
            C[i, j] = -(1/A1[i, i])*A1[i, j]
        V = V*C
        V.simplify()
        A1 = Dagger(C)*A1*C
        A1.simplify()
    return V


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
        mydict[g] = sympy.Matrix(n, n,
                                 lambda i, j:
                                 characteristic_function(G, g, i, j))
    return mydict


H = SymmetricGroup(3)

repr = regular_representation(H)
print(repr)
