import sympy as sp
sp.init_printing()
from itertools import permutations 
from itertools import combinations_with_replacement
from itertools import combinations
import numpy as np
from scipy.sparse import dok_matrix
from operator import add
import sys
import matplotlib.pyplot as plt
import networkx as nx
from sympy.matrices import Matrix, zeros
from sympy.solvers.solveset import linsolve

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
class YoungTableaux:
    def __init__(self, lamb, rho):
        self.lamb = lamb
        self.rho = rho
    def tableaux(self, v):
        for i in v:
            for j in range(0,len(i)-1):
                if (i[j]>i[j+1]):
                    return False
        for i in range(1,len(v)):
            for j in range(0,len(v[i])):
                if (v[i][j]<v[i-1][j]):
                    return False
        for i in range(0,len(v)):
            for j in range(0,len(v[i])):
                c=0
                if (j != 0):
                    if (v[i][j] == v[i][j-1]):
                        c=1
                if (j != (len(v[i])-1)):
                    if (v[i][j] == v[i][j+1]):
                        c=1
                if (i != 0):
                    if (v[i][j] == v[i-1][j]):
                        c=1
                if (i != (len(v)-1)):
    #            if (len(v[i+1]) <= len(v[i])):
                    if (j < len(v[i+1])):
                        if (v[i][j] == v[i+1][j]):
                            c=1
                if ((c == 0) and (self.rho[v[i][j]-1]>1)):
                    return False
        else:
            return True

    def MNR(self):
        p=[]
        i=1
        for h in self.rho:
            for j in range(0,h):
                p.append(i)
            i=i+1
        perm = permutations(p)
        D=[]
        for i in list(perm):
            v=[]   
            for g in i:
                v.append(g)
            c=0
            w=[]
            for p in self.lamb:
                u=[]
                for i in range(c,c+p):
                    u.append(v[i])
                w.append(u)
                c=c+p
            if (self.tableaux(w) == True):
                D.append(w)
        D1=[D[0]]
        for k1 in D:
            if k1 not in D1:
                D1.append(k1)
        return(D1)
    
    def Heights(self):
        H = self.MNR()
        He = []
        for h in H:
            he=[]
            for i in range(0,len(self.rho)):
                c = 0
                for g in h:
                    if ((i+1) in g):
                        c = c+1
                he.append(c-1)
            He.append(sum(he))
        return He

    def CMNR(self):
        He = self.Heights()
        s=0
        for j in He:
            s = s + (-1)**(j)
        return s
class Group_p_chains:
    def __init__(self, ke, unos):
        self.ke = ke
        self.unos = unos
        self.dic = {}
        c = 0
        for x in self.ke:
            self.dic[x] = self.unos[c]
            c = c+1
    def __add__(self, other):
        for y in other.ke:
            if y not in self.dic:
                self.dic[y] = other.dic[y]
            else:
                self.dic[y] = self.dic[y] + other.dic[y]
        return Group_p_chains(list(self.dic.keys()),list(self.dic.values()))    
    def __sub__(self, other):
        for y in other.ke:
            if y not in self.dic:
                self.dic[y] = -other.dic[y]
            else:
                self.dic[y] = self.dic[y] - other.dic[y]
        return Group_p_chains(list(self.dic.keys()),list(self.dic.values()))    
    def __eq__(self, other):
        return self.dic == other.dic 
    def __ne__(self, other):
        return not self.__eq__(other)

class SimplicialComplex:
    def __init__(self, G):
        self.G = G
        self.vertices = []
        for x in self.G.nodes():
            self.vertices.append(x)
    def faces(self):
        self.simplices = []
        self.simplices = self.vertices
        faceset = set()
        n = len(self.simplices)
        for k in range(n, 0, -1):
            for face in list(combinations(self.simplices,k)):
                if face not in faceset:
                    faceset.add(face)
#Aquí tengo problemas con la longitud de enteros
#        for simplex in self.simplices:
#            n = len(simplex)
#            for k in range(n-1, 0, -1):
#                for face in combinations(simplex,k):
#                    if sorted(face) not in faceset:
#                        faceset.append(sorted(face))
        return faceset

    def n_faces(self, n):
        return list(filter(lambda face: (len(face) == n+1) , self.faces())) 
    def zero_simplex(self):
        zero_s = Group_p_chains([], [1])
        if (list(self.G.nodes()) != []):
            for x in range(0, len(list(self.G.nodes()))):
                zero_s = zero_s + Group_p_chains([list(self.G.nodes())[x]], [1])
        return zero_s
    def one_simplex(self):
        one_s = Group_p_chains([], [1])
        if (list(self.G.edges()) != []):
            for x in range(0, len(list(self.G.edges()))):
                one_s = one_s + Group_p_chains([list(self.G.edges())[x]], [1])
        return one_s
    def two_simplex(self):
        two_s = Group_p_chains([], [1])
        if (len(self.G.edges()) >= 3):
            comb = combinations(list(self.G.nodes()), 3) 
            for g in list(comb):
                test=True
                for w in g:
                    for k in g:
                        if (k != w):
                            if (w not in [n for n in self.G.neighbors(k)]):
                                test=False
                if (test==True):
                    two_s = two_s + Group_p_chains([g], [1])
            return two_s
        else:
            return two_s
    def three_simplex(self):
        three_s = Group_p_chains([], [1])
        if (len(self.G.edges()) >= 4):
            comb = combinations(list(self.G.nodes()), 4) 
            for g in list(comb):
                test=True
                for w in g:
                    for k in g:
                        if (k != w):
                            if (w not in [n for n in self.G.neighbors(k)]):
                                test=False
                if (test==True):
                    three_s = three_s + Group_p_chains([g], [1])
            return three_s
        else:
            return three_s
    def simplex(self):
        sim = {}
        sim[0] = self.zero_simplex()
        sim[1] = self.one_simplex()
        sim[2] = self.two_simplex()
        sim[3] = self.three_simplex()
        return sim
    def p_homology_group_dimention(self, k):
        vk = self.simplex()[k]
        vkf = self.n_faces(k-1)
        M = zeros(len(vkf),len(vk.dic))
        j=0
        for u in list(vk.dic.keys()):
            d={u: vk.dic[u]}
            for a in list((boundary_op(d).dic).keys()):
                i=0
                for w in list(vkf):
                    if (a == w):
                        M[i,j]=(boundary_op(d).dic)[w]
                    i=i+1
            j=j+1
        dimKe = len(M.rref()[1])
        vk1 = self.simplex()[k+1]
        vkf1 = self.n_faces(k)
        N = zeros(len(vkf1),len(vk1.dic))
        j=0
        for u in list(vk1.dic.keys()):
            d={u: vk1.dic[u]}
            for a in list((boundary_op(d).dic).keys()):
                i=0
                for w in list(vkf1):
                    if (a == w):
                        N[i,j]=(boundary_op(d).dic)[w]
                    i=i+1
            j=j+1
        dimIm = len((N.T).rref()[1])
        dimH = dimKe - dimIm
        return dimKe, dimIm, dimH
def boundary_op(d):
    s = Group_p_chains([],1)
    for u in d.keys():
        c = 0
        for i in u:  
            w = set()
            for j in u:
                if (i != j):
                    w.add(j)
            s1 = Group_p_chains([tuple(w)],[1])
            if (d[u]*(-1)**c == -1):
                s = s - s1
            else:
                s = s + s1
            c = c+1
    return s

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


def unitary_representation(G, d):
    """
    A function that given a matrix representation become it
    in a unitary matrix representation.
    """
    n = d.degree
    A = sp.zeros(n, n)
    for g in d.map:
        J = (d.map[g].H)*d.map[g]
        J = sp.expand(J)
        A = J+A
    A1 = A
    V = sp.eye(n)
    for i in range(0, n):
        C = sp.eye(n)
        C[i, i] = sp.S(1)/sp.sqrt(A1[i, i])
        for j in range(i+1, n):
            C[i, j] = -(1/A1[i, i])*A1[i, j]
        V = V*C
        V = sp.expand(V)
        A1 = (C.H)*A1*C
        A1 = sp.expand(A1)
    M = {}
    for g in list(G.elements):
        M[g] = sp.ImmutableMatrix((V.inv())*d.map[g]*V)
    return MatrixRepresentation(M, G, n)


def is_irreducible(G, d):
    """A function that determine if a matrix representation is irreducible,
    if is reducible, return a matrix non escalar that reduce the
    matrix representation in otherwise return True"""
    n = d.degree
    N = sp.eye(n)
    for r in range(0, n):
        for s in range(0, n):
            H = sp.zeros(n)
            if (n-1-r == n-1-s):
                H[n-1-r, n-1-r] = 1
            else:
                if (n-1-r > n-1-s):
                    H[n-1-r, n-1-s] = 1
                    H[n-1-s, n-1-r] = 1
                else:
                    H[n-1-r, n-1-s] = 1*sp.I
                    H[n-1-s, n-1-r] = -1*sp.I
            M = sp.zeros(n, n)
            R = unitary_representation(G, d)
            for g in R.map:
                M = M+(R.map[g].H*H*R.map[g])
            M = (sp.S(1)/n)*M
            M = sp.expand(M)
            if (M != M[0, 0]*N):
                return M
    else:
        return True


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
            bloques=False
            while (bloques==False):
                bloques=True
                for j in range(c1,c1+c+1):
                    for k in range(c1+c+1,n):
                        if (M[j,k]!=0 or M[k,j]!=0):
                            bloques=False
                            c=Abs(i-k)
            v.append(c1+c)
            c1=c1+c+1
            i=c1
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


def reduce(G, d):
    """This function give a matrix that reduce the matrix representation"""
    M = is_irreducible(G, d)
    b = d.degree
    if M is True:
        return(sp.eye(b))
    else:
        (P, J) = M.jordan_form()
        P = sp.expand(P)
        w = []
        for g in d.map:
            w.append(block(P.inv()*d.map[g]*P))
        lon = len(w[0])
        au = w[0]
        for g in w:
            if (len(g) < lon):
                lon = len(g)
                au = g
        e = 0
        U = P
        for a in au:
            d1 = {}
            for g in list(G.elements):
                d1[g] = sp.ImmutableMatrix((P.inv()*d.map[g]*P)[e:a+1, e:a+1])
            if (is_representation(G,MatrixRepresentation(d1, G, (a+1-e)))==True):
                U = U*blockI(reduce(G, MatrixRepresentation(d1, G, (a+1-e))), b, e)
            else:
                display("Map d1 does not meet the requirements to be a representation.")
            e = a+1
        return U

    
def delta(G):
    d={}
    L=list(G.elements)
    n=len(L)
    for g in L:
        N=Matrix([])
        M=eye(n)
        for i in range(0,n):
            for j in range(0,n):
                if (L[i]*g==L[j]):
                        N=Matrix([N,M.row(j)])
        d[g]=N
    return MatrixRepresentation(d, G, n)


def is_representation(G,d):
    for g in list(G.elements):
        for h in list(G.elements):
            if (d.map[h*g]!=expand(d.map[g]*d.map[h])):
                return False
            if (g*h == G.identity()):
                if (d.map[h] != d.map[g].inv()):
                    return False
    if (d.map[G.identity()]!=eye(d.map[G[0]].shape[0])):
        return False
    else:
        return True
