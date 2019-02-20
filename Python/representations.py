import sympy as sp


class MatrixRepresentation:
    """
    A class of matricial representation of a group.
    """
    def __init__(self, d, G, n):
        """
        Parameters
        ----------
        d : dict
            Mapping of group elements to matrices
        G : sympy.combinatorics.perm_groups.PermutationGroup
            Group being represented
        n : int
            Degree of the representation
        """

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

    def unitary_equivalent(self):
        """An equivalent unitary representation.

        This implements Theorem 2.3 from Bannai-Ito (1994)

        """
        n, D = self.degree, self.map
        A = sp.zeros(n, n)
        for g in D:
            J = D[g].H*D[g]
            J.expand().applyfunc(sp.radsimp)
            A = J+A
        C = _UTforHermitian(A)
        return self.equivalent_by(C)


def _char_f(G, g, i, j):
    elems = list(G.elements)
    if g*elems[i] == elems[j]:
        return 1
    else:
        return 0


def _UTforHermitian(A):
    """Implements Lemma 2.2. from Bannai-Ito (1984)

    Given a Hermitian positive definite matrix, returns a nonsingular
    upper triangular matrix C such that C.H*A*C is the identity
    matrix.

    """
    A1, n = A, A.shape[0]
    V = sp.eye(n)
    for i in range(0, n):
        C = sp.eye(n)
        C[i, i] = sp.S(1)/sp.sqrt(A1[i, i])
        for j in range(i+1, n):
            C[i, j] = -(sp.S(1)/A1[i, i])*A1[i, j]
        V = V*C
        V.expand().applyfunc(sp.radsimp)
        A1 = C.H*A1*C
        A1.expand().applyfunc(sp.radsimp)
    return V


def regular_representation(G):
    elems = list(G.elements)
    n, mydict = len(elems), {}
    for g in elems:
        mydict[g] = sp.ImmutableMatrix(sp.Matrix(n, n,
                                                 lambda i, j:
                                                 _char_f(G, g, i, j)))
    return MatrixRepresentation(mydict, G, n)
