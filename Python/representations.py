import sympy as sp


class MatrixRepresentation:
    """
    A class of matricial representation of a group.
    """
    def __init__(self, d, G, n):
        self.map = d
        self.group = G
        self.degree = n

    def character(self):
        return dict([(g, self.map[g].trace()) for g in self.group.elements])

    def is_unitary(self):
        for g in self.group.elements:
            if sp.expand(self.map[g].H*self.map[g]) != sp.eye(self.degree):
                return False
        else:
            return True

    def is_irreducible(self):
        prod = sum([self.character()[g]*sp.conjugate(self.character()[g]) for
                    g in self.group.elements])
        return prod == self.degree

    def equivalent_by(self, P):
        """Equivalent representation, by conjugation with the matrix P."""
        d = dict([(g, P.inv()*self.map[g]*P) for g in self.group.elements])
        return MatrixRepresentation(d, self.group, self.degree)


def _char_f(G, g, i, j):
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
        mydict[g] = sp.ImmutableMatrix(sp.Matrix(n, n,
                                                 lambda i, j:
                                                 _char_f(G, g, i, j)))
    return MatrixRepresentation(mydict, G, n)
