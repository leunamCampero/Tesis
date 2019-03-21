import sympy as sp
sp.init_printing()


def simplifier_function(x):
    return sp.expand(sp.radsimp(x))


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

    def _test(self):
        D = self.map
        for g in self.group.elements:
            for h in self.group.elements:
                A = D[g]*D[h]
                A = A.applyfunc(simplifier_function)
                if A != D[g*h]:
                    return (g, h)
        else:
            return True

    def character(self):
        return dict([(g, self.map[g].trace()) for g in self.group.elements])

    def is_unitary(self):
        for g in self.group.elements:
            A = self.map[g].H*self.map[g]
            A = A.applyfunc(simplifier_function)
            if A != sp.eye(self.degree):
                return False
        else:
            return True

    def is_irreducible(self):
        prod = sum([self.character()[g]*sp.conjugate(self.character()[g]) for
                    g in self.group.elements])
        return prod == self.degree

    def equivalent_by(self, P):
        """Equivalent representation, by conjugation with the matrix P."""
        d = {}
        Pinv = P.inv()
        Pinv = Pinv.applyfunc(simplifier_function)
        for g in self.group.elements:
            A = Pinv*self.map[g]*P
            A = A.applyfunc(simplifier_function)
            d[g] = A
        return MatrixRepresentation(d, self.group, self.degree)

    def unitary_equivalent(self):
        """An equivalent unitary representation.

        This implements Theorem 2.3 from Bannai-Ito (1994)

        """
        n, D = self.degree, self.map
        A = sp.zeros(n, n)
        for g in D:
            J = D[g].H*D[g]
            J = J.applyfunc(simplifier_function)
            A = J+A
        C = _UTforHermitian(A)
        return self.equivalent_by(C)

    def _matrix_to_reduce(self, r, s):
        """A function that can be used to reduce a reducible matrix
        representation, when it returns a non escalar matrix.

        """
        n = self.degree
        M = sp.zeros(n)
        for g in self.map:
            M = M + self.map[g].H*_hermitian_rs(n, r, s)*self.map[g]
        M = (sp.S(1)/n)*M
        M = M.applyfunc(simplifier_function)
        return M


def _char_f(G, g, i, j):
    elems = list(G.elements)
    if elems[i]*g == elems[j]:
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
        V = V.applyfunc(simplifier_function)
        A1 = C.H*A1*C
        A1 = A1.applyfunc(simplifier_function)
    return V


def _matrix_rs(n, r, s):
    Ers = sp.zeros(n, n)
    Ers[r, s] = 1
    return Ers


def _hermitian_rs(n, r, s):
    if r == s:
        return _matrix_rs(n, r, r)
    elif r > s:
        return _matrix_rs(n, r, s) + _matrix_rs(n, s, r)
    else:
        return sp.I*(_matrix_rs(n, r, s) - _matrix_rs(n, s, r))


def regular_representation(G):
    elems = list(G.elements)
    n, mydict = len(elems), {}
    for g in elems:
        mydict[g] = sp.Matrix(n, n, lambda i, j: _char_f(G, g, i, j))
    return MatrixRepresentation(mydict, G, n)


def _check_candidate(A, c):
    n = A.shape[0]
    for i in range(c, n):
        for j in range(c):
            if A[i, j] != 0 or A[j, i] != 0:
                return False
    else:
        return True


def _detect_block(A):
    n = A.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n:
            if A[i, j] == 0 and A[j, i] == 0:
                j = j+1
            else:
                pass


def block(M):
    """A function that return where end the blocks of a matrix."""
    v = []
    c1 = 0
    i = 0
    n = M.shape[0]
    while (c1 < n):
        c = 0
        for j in range(c1, n):
            if (M[i, j] != 0 or M[j, i] != 0):
                if (sp.Abs(i-j) > c):
                    c = sp.Abs(i-j)
        if (c == 0):
            v.append(c1)
            c1 = c1+1
            i = c1
        else:
            bloques = False
            while not bloques:
                bloques = True
                for j in range(c1, c1+c+1):
                    for k in range(c1+c+1, n):
                        if (M[j, k] != 0 or M[k, j] != 0):
                            if (sp.Abs(i-k) > c):
                                c = sp.Abs(i-k)
            v.append(c1+c)
            c1 = c1+c+1
            i = c1
    return v


def blockI(M, n, i):
    """A function that given a matrix, put it since the entry (i,i) of a
    identity matrix of degree n"""
    a = M.shape[0]
    N = sp.eye(n)
    for j in range(0, a):
        for k in range(0, a):
            N[j+i, k+i] = M[j, k]
    return N


# def reduce(G, d):
#     """This function give a matrix that reduce the matrix representation"""
#     M = is_irreducible(G, d)
#     b = d.degree
#     if M is True:
#         return(sp.eye(b))
#     else:
#         (P, J) = M.jordan_form()
#         P = sp.expand(P)
#         w = []
#         for g in d.map:
#             w.append(block(P.inv()*d.map[g]*P))
#         lon = len(w[0])
#         au = w[0]
#         for g in w:
#             if (len(g) < lon):
#                 lon = len(g)
#                 au = g
#         e = 0
#         U = P
#         for a in au:
#             d1 = {}
#             for g in list(G.elements):
#                 d1[g] = sp.ImmutableMatrix((P.inv()*d.map[g]*P)[e:a+1, e:a+1])
#             U = U*blockI(reduce(G, MatrixRepresentation(d1, G, (a+1-e))), b, e)
#             e = a+1
#         return U
