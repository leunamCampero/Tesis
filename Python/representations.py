from sympy import *
init_printing(use_unicode=True)
#import sympy 
from sympy import sympify
from sympy import solve
from sympy.abc import x
import networkx as nx
import numpy as np
from itertools import combinations
from scipy.sparse import dok_matrix
from operator import add
import sys
import matplotlib.pyplot as plt
#from sympy import Matrix
from sympy.matrices import Matrix, zeros
from sympy.solvers.solveset import linsolve
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.combinatorics import Permutation
import itertools as itert 
from itertools import permutations 
from sympy import Identity, eye
from itertools import combinations_with_replacement
from sympy.combinatorics.partitions import IntegerPartition
import math
import copy
import unittest

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

class Vec2D:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vec2D(self.x + other.x, self.y + other.y)
class P_chains:
    """
    A class used to do and operate p-chains.

    ...
    Attributes:
    ----------
    keys : list
        a list with the elements of the p-chains
    values : list
        a list with the coefficient for every element in the p-chains


    """
    
    def __init__(self, keys, values):
        '''Makes a p-chain.
        
        Args:
            keys (list): A list with the elements of the p-chains.
            values (list): A list with the coefficient for every element in the p-chain.
        
        Raises:
            IndexError: If the number of p-simplexs is not equal to the number of coefficients. 
            TypeError: If the p-simplex given is not immutable data types like a tuple.
        
        Examples:
            To make a p-chain, use the ``P_chains`` class.  A p-chain is 
            constructed by providing a list of p-simplex and a list with 
            their coefficients, i.e.
            
            >>> P = P_chains([(0,1,2,3)],[1])
            >>> Q = P_chains([(0,1,2),(0,1,3)],[-1,2])
            >>> display(P.dic)
            >>> display(Q.dic)
            {(0, 1, 2, 3): 1}
            {(0, 1, 2): -1, (0, 1, 3): 2}
            
            One important thing to note about P_chains is that the p-simplex must be 
            a immutable data types like a tuple, and to see the P-chains, you need 
            use ``.dic``.
        
        '''
        self.keys = keys
        self.values = values
        self.dic = {}
        c = 0
        for x in self.keys:
            self.dic[x] = self.values[c]
            c = c+1
            
    def __add__(self, other):
        """Sum two p-chains.
        
        Args:
            other ( __main__.P_chains): Other p-chain.
            
        Returns
            __main__.P_chains: A new p-chains that is the sum of the two p-chains given.
        
        Examples:
            To sum two p-chains, use ``+``.
            
            >>> P = P_chains([(0,1,2)],[2])
            >>> Q = P_chains([(0,1,3)],[5])
            >>> T = P_chains([(0,1,2)],[-2])
            >>> R = P + Q
            >>> L = P + T
            >>> display(R.dic)
            >>> display(L.dic)
            {(0, 1, 2): 2, (0, 1, 3): 5}
            {(0, 1, 2): 0}
            
            Note that the coefficient of L is zero because the P and T have the
            same simplex, then P + T is the sum of their respective coefficients.
        
        """
        D = {}
        for x in list(self.dic.keys()):
            D[x] = self.dic[x]
        for y in list(other.dic.keys()):
            if y not in list(D.keys()):
                D[y] = other.dic[y]
            else:
                D[y] = D[y] + other.dic[y]
        w1 = []
        w2 = []
        for h in list(D.keys()):
            w1.append(h)
            w2.append(D[h])
        return P_chains(w1,w2)    
    
    def __sub__(self, other):
        """Subtract two p-chains.
        
        Args:
            other ( __main__.P_chains): Other p-chain.
        
        Returns
            __main__.P_chains: A new p-chains that is the substract of the two p-chains given.
            
        Examples:
            To subtract two p-chains, use ``-``.
            
            >>> P = P_chains([(3,4,5)],[3])
            >>> Q = P_chains([(1,8,9)],[1])
            >>> T = P_chains([(0,1,2,3)],[-7])
            >>> R = P - Q
            >>> L = P - T
            >>> display(R.dic)
            >>> display(L.dic)
            {(3, 4, 5): 3, (1, 8, 9): -1}
            {(3, 4, 5): 3, (0, 1, 2, 3): 7}
           
        """
        D = {}
        for x in list(self.dic.keys()):
            D[x] = self.dic[x]
        for y in list(other.dic.keys()):
            if y not in list(D.keys()):
                D[y] = -other.dic[y]
            else:
                D[y] = D[y] - other.dic[y]
        w1 = []
        w2 = []
        for h in list(D.keys()):
            w1.append(h)
            w2.append(D[h])
        return P_chains(w1,w2)       
    
    def __eq__(self, other):
        '''Return if the two P_chains are equal
        
        Args:
            other ( __main__.P_chains): Other p-chain.
        
        Returns
            bool: The return value. True for success, False otherwise.  
            
        Examples:
            To know if two P_chains are equal use ``==``.
            
            >>> P = P_chains([(0,1,2),(3,4,5)],[1,1])
            >>> Q = P_chains([(3,4,5),(0,1,2)],[1,1])
            >>> R = P_chains([(0,1,2)],[1])
            >>> L = P_chains([(1,0,2)],[1])
            >>> display(P == Q)
            >>> display(R == L)
            True
            False
            
            ..Note:: R and L are not equal even though they only are 
            distint in orientation, moreover, in this class the 
            orientation is not defined yet.
            
        '''
        return self.dic == other.dic
    
    def __ne__(self, other):
        '''Return if the two P_chains are not equal
        
        Args:
            other ( __main__.P_chains): Other p-chain.
        
        Returns
            bool: The return value. True for success, False otherwise.  
            
        Examples:
            To know if two P_chains are equal use ``!=``.
            
            >>> P = P_chains([(0,1,2,4)],[1])
            >>> Q = P_chains([(0,1,4,2)],[-2])
            >>> R = P_chains([(5,6,7)],[1])
            >>> L = P_chains([(5,6,7)],[1])
            >>> display(P != Q)
            >>> display(R != L)
            True
            False
            
        '''
        return not self.__eq__(other)
    
    def mul_esc(self, esc):
        '''Return if the two P_chains are not equal
        
        Args:
            other ( __main__.P_chains): Other p-chain.
        
        Returns
            bool: The return value. True for success, False otherwise.  
            
        Examples:
            To know if two P_chains are equal use ``!=``.
            
            >>> P = P_chains([(7,8,9),(10,11,12)],[3,2])
            >>> Q = P_chains([(0,1,4,2,6)],[-5])
            >>> display(P.mul_esc(3).dic)
            >>> display(Q.mul_esc(-1).dic)
            {(7, 8, 9): 9, (10, 11, 12): 6}
            {(0, 1, 4, 2, 6): 5}
        
        '''
        aux = P_chains([],[])
        for x in self.dic:
            aux.dic[x] = esc*self.dic[x]
        return aux
    
    
class SimplicialComplex:
    def __init__(self, G):
        self.G = G
        self.vertices = []
        for x in self.G.nodes():
            self.vertices.append(x)
    def faces(self):
        faceset = []
        for face in list(nx.enumerate_all_cliques(self.G)):
            faceset.append(tuple(face))
        return faceset
    def n_faces(self, n):
        return list(filter(lambda face: (len(face) == n+1) , self.faces()))
    def dimension(self):
        a = 0
        for x in self.faces():
            if (len(x) > a):
                a = len(x) 
        return a-1
    def p_simplex(self, k):
        p_simplex = []
        for x in self.n_faces(k):
            p_simplex.append(x)
        return p_simplex
    def elementary_chain(self, simplicie):
        ec = P_chains([], [])
        for x in set_oriented_p_simplices(simplicie):
            if (orientation_function(tuple_sorted(simplicie), x, len(simplicie)-1) == True):
                ec = ec + P_chains([x], [1])
            else:
                ec = ec - P_chains([x], [1])
        return ec
    def oriented_p_chains(self, k):
        if ((k<0) or (k>self.dimension())):
            return 0
        else:
            c_p = P_chains([], [1])
            for x in self.p_simplex(k):
                c_p = c_p + self.elementary_chain(tuple_sorted(x))
            return c_p
    def oriented_p_chains_op(self, k):
        if ((k<0) or (k>self.dimension())):
            return 0
        else:
            c_p = P_chains([], [1])
            for x in self.p_simplex(k):
                c_p = c_p + P_chains([tuple_sorted(x)], [1])
            return c_p
        
    def representate_in_simplex(self, vec, P):
        s = P_chains([],[])
        if (vec.dic != {}):
            v = list(vec.dic.keys())
            p = len(list(vec.dic.keys())[0]) - 1
            ve = self.oriented_p_chains_op(p)
            for a in v:
                if (isinstance(a, int) == True): 
                    return vec
                else:
                    w = tuple_permutation(a,P)
                    for b in list(ve.dic.keys()):
                        if (eq_elements(b,w) == True):
                            if (orientation_function(b,w,p) == True):
                                s = s + P_chains([b],[vec.dic[a]])
                            else:
                                s = s - P_chains([b],[vec.dic[a]])
            return s
        else:
            return s
    def matrix_simmetric_representate(self, p):
        if (p >0 and (p <= self.dimension()) ):
            v = self.oriented_p_chains_op(p)
            p = p - 1
            ve = self.oriented_p_chains_op(p)
            M = zeros(len(ve.dic),len(v.dic))
            j = 0
            for u1 in list(v.dic.keys()):
                d =  P_chains([u1],[v.dic[u1]])
                for u2 in list(boundary_op_n(d).dic.keys()):
    #                display(boundary_op(d, self.G).dic)
                    i = 0
                    for w in list(ve.dic.keys()):
                        if (eq_elements(w,u2) == True):
#                            if (orientation_function(u2,tuple(w),p) == True):
                                M[i,j] = int((boundary_op_n(d).dic)[u2])
#                            else:
#                                M[i,j] = int((boundary_op_n(d).dic)[u2])*(-1)
                        i = i + 1
                j = j + 1
            return M
        else:
            if (p == 0):
                return eye(len(list(self.oriented_p_chains_op(0).dic.keys())))
            else:
                return False
                
    def kernel_boundary_op(self, p):
#        display(p,self.dimension())
        if ((p > 0) and (p <= self.dimension())):
            u = nullspace(self.matrix_simmetric_representate(p))
#            display(u)
            if (u != False):
                v = self.oriented_p_chains_op(p)
                w = []
                for i in range(len(u)):
                    s = P_chains([],[])
                    for j in range(len(u[i])):
                        if (u[i][j] != 0):
                            s = s + P_chains([list(v.dic.keys())[j]],[u[i][j]])
                    w.append(s)
                return w
            else:
                return False
        else:
            if (p == 0):
                return [self.oriented_p_chains_op(p)]
            else:
                return False
    def image_boundary_op(self, p):
#        display(p, self.dimension())
        if ((p > 0) and (p <= self.dimension())):
            u = columnspace(self.matrix_simmetric_representate(p))
            if (u != False):
                v = self.oriented_p_chains_op(p-1)
                w = []
                for i in range(len(u)):
                    s = P_chains([],[])
                    for j in range(len(u[i])):
                        if (u[i][j] != 0):
                            s = s + P_chains([list(v.dic.keys())[j]],[u[i][j]])
                    w.append(s)
                return w
            else:
                return False
        else:
            return False
    def character_kernel(self, p, P):
        A=self.matrix_simmetric_representate(p)
        if (p>0 and (p <= self.dimension())):
            M = []
            null = nullspace(A)
            for i in range(len(null[0])):
                w = []
                for j in range(len(null)):
                    w.append(null[j][i])
                M.append(w)
            Q=Reduce(Matrix(M))
            M=Q[0]*Matrix(M)
        else:
            if (p == 0):
                M = A
                null = []
                for i in range(A.shape[0]):
                    aux = []
                    for j in range(A.shape[1]):
                        aux.append(M[i,j])
                    null.append(aux)
                Q=Reduce(Matrix(M))
                M=Q[0]*Matrix(M)
            else:
                return 0
        if (all(elem == null[0][0] for elem in null[0])):
            return 0
        else:
            w1=[]
            he = self.oriented_p_chains_op(p)
            for a in range(len(null)):
                N = []
                v = P_chains([],[])
                c = 0
                for j in list(he.dic.keys()):
                    v = v + P_chains([j],[null[a][c]])
                    c=c+1
                v1 = permutation_in_simplex_test(v, P)
                u=[]
                for i in list(he.dic.keys()):
                    for j in list(v1.dic.keys()):
                        if (eq_elements(i, j) == True):
                            u.append(np.array([v1.dic[j]]))
                u=Q[0]*Matrix(u)
                N = np.append(M, u, axis=1)
                N = Matrix(N)
                w2 = []
                for i in tuple(linsolve(N)):
                    for j in i:
                        w2.append(j)
                w1.append(w2)
            N = Matrix(w1)
#            display(N.T)
            return np.trace(N.T)
    def character_image(self, p, P):
        if (p>0 and (p <= self.dimension())):
            A=self.matrix_simmetric_representate(p)
            w1=[]
            M = []
            col = columnspace(A)
            for i in range(len(col[0])):
                w = []
                for j in range(len(col)):
                    w.append(col[j][i])
                M.append(w)
            Q=Reduce(Matrix(M))
            M=Q[0]*Matrix(M)
            he = self.oriented_p_chains_op(p-1)
            for a in range(len(col)):
                N = []
                v = P_chains([],[])
                c = 0
                for j in list(he.dic.keys()):
                    v = v + P_chains([j],[col[a][c]])
                    c=c+1
                v1 = permutation_in_simplex_test(v, P)
                u=[]
                for i in list(he.dic.keys()):
                    for j in list(v1.dic.keys()):
                        if (eq_elements(i, j) == True):
                            u.append(np.array([v1.dic[j]]))
                u=Q[0]*Matrix(u)
                N = np.append(M, u, axis=1)
                N = Matrix(N)
                w2 = []
                for i in tuple(linsolve(N)):
                    for j in i:
                        w2.append(j)
                w1.append(w2)
            N = Matrix(w1)
#            display(N.T)
            return np.trace(N.T)
        else:
            return 0
    def character_p_homology(self, p, P):
        return self.character_kernel(p, P) - self.character_image(p + 1, P)
    def specific_function(self, n):
        w = list_partition(n)
        M = form_matrix_yt(w)
        card = math.factorial(n)
        vec_dic = {}
        for k in range(self.dimension()+1):
            D = {}
            u = []
            v = []
            for h in w:
                u.append(self.character_p_homology(k, make_permutation(h)))
                v.append(size_conjugacy_class(h))
            for i in range(M.shape[0]):
                Ip = 0
                for j in range(M.shape[1]):
                    Ip = Ip + M[i,j]*u[j]*v[j]
                Ip = Ip/card
                D[tuple(w[i])]=Ip
            vec_dic[k] = D
        return vec_dic

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
                c1 = 0
                if (j != 0):
                    if (v[i][j] == v[i][j-1]):
                        c=1
                        c1 = c1 + 1
                if (j != (len(v[i])-1)):
                    if (v[i][j] == v[i][j+1]):
                        c=1
                        c1 = c1 + 1
                if (i != 0):
                    if (v[i][j] == v[i-1][j]):
                        c=1
                if (i != (len(v)-1)):
    #            if (len(v[i+1]) <= len(v[i])):
                    if (j < len(v[i+1])):
                        if (v[i][j] == v[i+1][j]):
                            c=1
                            c1 = c1 + 1
                    if (j < (len(v[i+1]) - 1)):
                        if (v[i][j] == v[i+1][j+1]):
                            c1 = c1 + 1
                if ((c == 0) and (self.rho[v[i][j]-1]>1)):
                    return False
                if (c1 == 3):
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
        D1 = []
        if (D != []):
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
def boundary_op(v, G):
    sc = SimplicialComplex(G)
    p = len(list(v.dic.keys())[0]) - 1
    s = P_chains([],[])
    if (p != 0):
        ve = sc.oriented_p_chains_op(p)
        for u in v.dic.keys():
            for k in ve.dic.keys():
                if (eq_elements(k,u) == True):
                    if (orientation_function(k,u,p) == True):
                        c = 0
                        for i in k:  
                            w = list(k).copy()
                            w.remove(i)
                            s1 = P_chains([tuple(w)],[abs(v.dic[u])])
                            if (np.sign((v.dic[u])*(-1)**c) < 0):
                                s = s - s1
                            else:
                                s = s + s1
                            c = c+1
                    else:
                        c = 0
                        for i in k:  
                            w = list(k).copy()
                            w.remove(i)
                            s1 = P_chains([tuple(w)],[abs(v.dic[u])])
                            if (np.sign((v.dic[u])*(-1)**(c+1)) < 0):
                                s = s - s1
                            else:
                                s = s + s1
                            c = c+1
        return s
    else:
        return s
def boundary_op_n(v):
#    sc = SimplicialComplex(G)
    p = len(list(v.dic.keys())[0]) - 1
    s = P_chains([],[])
    if (p != 0):
#        ve = sc.oriented_p_chains_op(p)
        for u in v.dic.keys():
            c = 0
            for i in u:  
                w = list(u).copy()
                w.remove(i)
                if (orientation_function(tuple(tuple_sorted(tuple(w))),tuple(w),p) == True):
                    s1 = P_chains([tuple(tuple_sorted(tuple(w)))],[abs(v.dic[u])])
                    if (np.sign((v.dic[u])*(-1)**c) < 0):
                        s = s - s1
                    else:
                        s = s + s1
                    c = c+1
                else:
                    s1 = P_chains([tuple(tuple_sorted(tuple(w)))],[abs(v.dic[u])])
                    if (np.sign((v.dic[u])*(-1)**(c+1)) < 0):
                        s = s - s1
                    else:
                        s = s + s1
                    c = c+1
        return s
    else:
        return s
def form_matrix_yt(w):
    M = zeros(len(w),len(w))
    for i in range(len(w)):
        for j in range(len(w)):
            M[i,j] = YoungTableaux(w[i],w[j]).CMNR()
    return M
def list_partition(n):
    p = IntegerPartition([n])
    w = []
    while list(p.args[1]) not in w:
        w.append(list(p.args[1]))
        p = p.next_lex()
    return w
def eq_elements(a, b):
    if (isinstance(a, int) == True):
        return a == b
    if (isinstance(a[0], int) == True):
        return (set() == set(a).difference(set(b)))
    else:      
        for i in range(len(a)):
            test = False 
            for j in range(len(b)):
                if (eq_elements(a[i],b[j]) == True):
                    test = True
            if (test == False):
                return False
        else:
            return True
def orientation_function(a,b,p):
    if (p == 0):
        return True
    else: 
        v = np.zeros((len(a),), dtype = int)
        for i in range(len(a)):
            for j in range(len(b)):
                if (eq_elements(a[i],b[j]) == True):
                    v[j] = i
        P = Permutation(v)
        return P.is_even
def grafica_de_emparejamiento(n):
    k_n = nx.complete_graph(n)
    G = nx.Graph()
    for i in k_n.edges():
        G.add_node(i)
    w = []
    for i in k_n.edges():
        for j in k_n.edges():
            if ((j[0] not in i) and (j[1] not in i) and ((i,j) not in w) and ((j,i) not in w)): 
                w.append((i,j))
                G.add_edge(i,j)
    return G
def set_oriented_p_simplices(simplicie):
        return list(permutations(simplicie))
def elementary_chain_f(simplicie):
    ec = P_chains([], [1])
    for x in set_oriented_p_simplices(simplicie):
        if (orientation_function(simplicie, x, len(x)-1) == True):
            ec = ec + P_chains([x], [1])
        else:
            ec = ec - P_chains([x], [1])
    return ec
def clique_graph(g, cmax=math.inf):
    ite = nx.find_cliques(g)
    cliques = []
    K = nx.Graph()
    while True:
        try:
            cli = next(ite)
            cliques.append(frozenset(cli))
            if len(cliques) > cmax:
                return None
        except StopIteration:
            break
    K.add_nodes_from(cliques)
    clique_pairs = itert.combinations(cliques, 2)
    K.add_edges_from((c1, c2) for (c1, c2) in clique_pairs if c1 & c2)
    G1 = nx.Graph()
    for i in K.nodes():
        G1.add_node(tuple(sorted(i)))
    for i in K.edges():
        if (tuple(sorted(i[0]))<tuple(sorted(i[1]))):
            G1.add_edge(tuple(sorted(i[0])),tuple(sorted(i[1])))
        else:
            e = tuple_sorted((tuple(sorted(i[1])),tuple(sorted(i[0]))))
            G1.add_edge(*e)
    return G1
def tuple_sorted(a):
    if (isinstance(a, int) == True):
        return a
    if (isinstance(a[0], int) == True):
        return sorted(a)
    else:
        w = []
        for b in a:
            w.append(tuple(tuple_sorted(b)))
        return tuple(sorted(tuple(w)))
def tuple_permutation(v,P):
    u = []
    w = list(v).copy()
    test = True
    for i in range(len(v)):
        if (isinstance(v[i], int) == True):
            if (v[i] in P):
                w[i] = P(v[i])
        else:
            u.append(tuple_permutation(tuple(v[i]),P))
            test = False
    if (test == True):
        return tuple(w)
    else:
        return tuple(u)
def convert_to_int(v,k):
    w=[]
    for i in v:
        u=[]
        for j in i:
            if (sympify(j*k).is_integer == False):
                return convert_to_int(v,int(solve(x*j + np.sign(j)*(-1))[0]))
            else:
                u.append(j*k)
        w.append(u) 
    return w
def nullspace(A):
    u = A.nullspace()
    w= []
    for g in u:
        v=[]
        for i in g:
#            i = sympify(i)
#            display(type(i))
            v.append(i)
        w.append(v)
    w = convert_to_int(w,1)
    if (w == []):
        return [np.zeros((A.shape[1],), dtype = int)]
    else:
        return w
def columnspace(A):
    u = A.columnspace()
    w= []
    for g in u:
        v=[]
        for i in g:
            v.append(i)
        w.append(v)
    w = convert_to_int(w,1)
    if (w == []):
        return [np.zeros((A.shape[0],), dtype = int)]
    else:
        return w
def permutation_in_simplex_test(vec, P):
    s = P_chains([],[])
    if (vec.dic != {}):
        v = list(vec.dic.keys())
        p = len(list(vec.dic.keys())[0]) - 1
        ve = vec
        for a in v:
            if (isinstance(a, int) == True): 
                return vec
            else:
                w = tuple_permutation(a,P)
                if (orientation_function(tuple_sorted(w),w,p) == True):
                    s = s + P_chains([tuple_sorted(w)],[vec.dic[a]])
                else:
                    s = s - P_chains([tuple_sorted(w)],[vec.dic[a]])
        return s
    else:
        return s
def permutation_in_simplex(vec, P):
    s = P_chains([],[])
    if (vec.dic != {}):
        v = list(vec.dic.keys())
        p = len(list(vec.dic.keys())[0]) - 1
        ve = vec
        for a in v:
            if (isinstance(a, int) == True): 
                return vec
            else:
                w = tuple_permutation(a,P)
                for b in list(ve.dic.keys()):
                    if (eq_elements(b,w) == True):
                        if (orientation_function(b,w,p) == True):
                            s = s + P_chains([b],[vec.dic[a]])
                        else:
                            s = s - P_chains([b],[vec.dic[a]])
        return s
    else:
        return s
def permutation_in_simplex_es(vec, P):
    s = P_chains([],[])
    if (vec.dic != {}):
        v = vec.dic
        p = len(list(vec.dic.keys())[0]) - 1
        ve = vec
        for a in v:
            if (isinstance(a, int) == True): 
                return vec
            else:
                if (v[a] != 0):
                    w = tuple_permutation(a,P)
                else:
                    w = a
                    for b in ve.dic:
                        s = s + P_chains([b],[0])
                        if (ve.dic[b] != 0):
                            if (eq_elements(b,w) == True):
                                if (orientation_function(b,w,p) == True):
                                    s = s + P_chains([b],[vec.dic[a]])
                                else:
                                    s = s - P_chains([b],[vec.dic[a]])
        return s
    else:
        return s
def size_conjugacy_class(partition):
    aux1=1
    c=0
    aux=partition[0]
    flag = 1
    c2 = 0
    for j in range(len(partition)):
        if (aux == partition[j]):
            c = c + 1
            flag = 1
        else:
            aux1 = aux1*(partition[j-1]**c)*(math.factorial(c))
            aux = partition[j]
            c = 1
            flag = 0
            c2 = j
    if (flag == 1):
        aux1 = aux1*(partition[j-1]**c)*(math.factorial(c))
    else: 
        aux1 = aux1*(partition[j]**c)*(math.factorial(c))
    card = (math.factorial(n))/aux1
    return int(card)
def make_permutation(partition):
    P = Permutation()
    c = 0
    for j in range(len(partition)):
        a = []
        for h in range(partition[j]):
            a.append(c)
            c = c + 1 
        if (c == 1):
            P1 = Permutation()
            c = 0
        else:
            P1 = Permutation([a])
        P = P*P1
    return P
def Reduce(M):
    lead = 0
    rowCount = M.shape[0]
    columnCount = M.shape[1]
    A=eye(rowCount)
    #v=[]
    #v.append(A)
    for r in range(rowCount):
        #display(r)
        B1=eye(rowCount)
        #B2=eye(rowCount)
        #B3=eye(rowCount)
        if (columnCount <= lead):
            return A,M
        i = r
        while (M[i, lead] == 0):
            i = i + 1
            if (rowCount == i):
                i = r
                lead = lead + 1
                if (columnCount == lead):
                    return A,M
        B1.row_swap(i, r)
        M.row_swap(i, r)
        a=M[r,lead]
        for k in range(columnCount):
            M[r,k]=S(M[r,k])/a
            if (k < rowCount):
                B1[r,k]=S(B1[r,k])/a
        for i in range(0,rowCount):
            if (i != r):
                a=M[i,lead]
                for k in range(0,columnCount):
                    M[i,k]=M[i,k]-M[r,k]*a
                    if (k < rowCount):
                        B1[i,k]=B1[i,k]-B1[r,k]*a
        lead = lead + 1
        A=B1*A
    return A,M

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
import sympy
from sympy import sympify
from sympy import solve
from sympy.abc import x
import networkx as nx
import numpy as np
from itertools import combinations
from scipy.sparse import dok_matrix
from operator import add
import sys
import matplotlib.pyplot as plt
#from sympy import Matrix
from sympy.matrices import Matrix, zeros
from sympy.solvers.solveset import linsolve
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy.combinatorics import Permutation
import itertools as itert
from itertools import permutations
from sympy import Identity, eye
from itertools import combinations_with_replacement
from sympy.combinatorics.partitions import IntegerPartition
import math
import copy
import unittest
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
    def mul_esc(self, esc):
        aux = Group_p_chains([],[])
        for x in self.dic:
            aux.dic[x] = esc*self.dic[x]
        return aux
class SimplicialComplex:
    def __init__(self, G):
        self.G = G
        self.vertices = []
        for x in self.G.nodes():
            self.vertices.append(x)
    def faces(self):
        faceset = []
        for face in list(nx.enumerate_all_cliques(self.G)):
            faceset.append(tuple(face))
        return faceset
    def n_faces(self, n):
        return list(filter(lambda face: (len(face) == n+1) , self.faces()))
    def dimension(self):
        a = 0
        for x in self.faces():
            if (len(x) > a):
                a = len(x)
        return a-1
    def p_simplex(self, k):
        p_simplex = []
        for x in self.n_faces(k):
            p_simplex.append(x)
        return p_simplex
    def elementary_chain(self, simplicie):
        ec = Group_p_chains([], [1])
        for x in set_oriented_p_simplices(simplicie):
            if (orientation_function(simplicie, x, len(simplicie)-1) == True):
                ec = ec + Group_p_chains([x], [1])
            else:
                ec = ec - Group_p_chains([x], [1])
        return ec
    def group_of_oriented_p_chains(self, k):
        if ((k<0) or (k>self.dimension())):
            return 0
        else:
            c_p = Group_p_chains([], [1])
            for x in self.p_simplex(k):
                c_p = c_p + self.elementary_chain(x)
            return c_p
    def group_of_oriented_p_chains_op(self, k):
        if ((k<0) or (k>self.dimension())):
            return 0
        else:
            c_p = Group_p_chains([], [1])
            for x in self.p_simplex(k):
                c_p = c_p + Group_p_chains([x], [1])
            return c_p
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
    def representate_in_simplex(self, vec, P):
        s = Group_p_chains([],[])
        if (vec.dic != {}):
            v = list(vec.dic.keys())
            p = len(list(vec.dic.keys())[0]) - 1
            ve = self.group_of_oriented_p_chains_op(p)
            for a in v:
                if (isinstance(a, int) == True):
                    return vec
                else:
                    w = tuple_permutation(a,P)
                    for b in list(ve.dic.keys()):
                        if (eq_elements(b,w) == True):
                            if (orientation_function(b,w,p) == True):
                                s = s + Group_p_chains([b],[vec.dic[a]])
                            else:
                                s = s - Group_p_chains([b],[vec.dic[a]])
            return s
        else:
            return s
    def matrix_simmetric_representate(self, p):
        if (p >0 and (p <= self.dimension()) ):
            v = self.group_of_oriented_p_chains_op(p)
            p = p - 1
            ve = self.group_of_oriented_p_chains_op(p)
            M = zeros(len(ve.dic),len(v.dic))
            j = 0
            for u1 in list(v.dic.keys()):
                d =  Group_p_chains([u1],[v.dic[u1]])
                for u2 in list(boundary_op(d, self.G).dic.keys()):
    #                display(boundary_op(d, self.G).dic)
                    i = 0
                    for w in list(ve.dic.keys()):
                        if (eq_elements(w,u2) == True):
                            M[i,j] = int((boundary_op(d,self.G).dic)[u2])
                        i = i + 1
                j = j + 1
            return M
        else:
            if (p == 0):
                return eye(len(list(self.group_of_oriented_p_chains_op(0).dic.keys())))
            else:
                return False
               
    def kernel_boundary_op(self, p):
#        display(p,self.dimension())
        if ((p > 0) and (p <= self.dimension())):
            u = nullspace(self.matrix_simmetric_representate(p))
#            display(u)
            if (u != False):
                v = self.group_of_oriented_p_chains_op(p)
                w = []
                for i in range(len(u)):
                    s = Group_p_chains([],[])
                    for j in range(len(u[i])):
                        if (u[i][j] != 0):
                            s = s + Group_p_chains([list(v.dic.keys())[j]],[u[i][j]])
                    w.append(s)
                return w
            else:
                return False
        else:
            if (p == 0):
                return [self.group_of_oriented_p_chains_op(p)]
            else:
                return False
    def image_boundary_op(self, p):
#        display(p, self.dimension())
        if ((p > 0) and (p <= self.dimension())):
            u = columnspace(self.matrix_simmetric_representate(p))
            if (u != False):
                v = self.group_of_oriented_p_chains_op(p-1)
                w = []
                for i in range(len(u)):
                    s = Group_p_chains([],[])
                    for j in range(len(u[i])):
                        if (u[i][j] != 0):
                            s = s + Group_p_chains([list(v.dic.keys())[j]],[u[i][j]])
                    w.append(s)
                return w
            else:
                return False
        else:
            return False
    def character_kernel(self, p, P):
        A=self.matrix_simmetric_representate(p)
        if (p>0 and (p <= self.dimension())):
            M = []
            null = nullspace(A)
            for i in range(len(null[0])):
                w = []
                for j in range(len(null)):
                    w.append(null[j][i])
                M.append(w)
        else:
            if (p == 0):
                M = A
                null = []
                for i in range(A.shape[0]):
                    aux = []
                    for j in range(A.shape[1]):
                        aux.append(M[i,j])
                    null.append(aux)
            else:
                return 0
        if (all(elem == null[0][0] for elem in null[0])):
            return 0
        else:
            w1=[]
            he = self.group_of_oriented_p_chains_op(p)
            for a in range(len(null)):
                N = []
                v = Group_p_chains([],[])
                c = 0
                for j in list(he.dic.keys()):
                    v = v + Group_p_chains([j],[null[a][c]])
                    c=c+1
                v1 = permutation_in_simplex(v, P)
                u=[]
                for i in list(he.dic.keys()):
                    for j in list(v1.dic.keys()):
                        if (eq_elements(i, j) == True):
                            u.append(np.array([v1.dic[j]]))
                N = np.append(M, u, axis=1)
                N = Matrix(N)
                w2 = []
                for i in tuple(linsolve(N)):
                    for j in i:
                        w2.append(j)
                w1.append(w2)
            N = Matrix(w1)
#            display(N.T)
            return np.trace(N.T)
    def character_image(self, p, P):
        if (p>0 and (p <= self.dimension())):
            A=self.matrix_simmetric_representate(p)
            w1=[]
            M = []
            col = columnspace(A)
            for i in range(len(col[0])):
                w = []
                for j in range(len(col)):
                    w.append(col[j][i])
                M.append(w)
            he = self.group_of_oriented_p_chains_op(p-1)
            for a in range(len(col)):
                N = []
                v = Group_p_chains([],[])
                c = 0
                for j in list(he.dic.keys()):
                    v = v + Group_p_chains([j],[col[a][c]])
                    c=c+1
                v1 = permutation_in_simplex(v, P)
                u=[]
                for i in list(he.dic.keys()):
                    for j in list(v1.dic.keys()):
                        if (eq_elements(i, j) == True):
                            u.append(np.array([v1.dic[j]]))
                N = np.append(M, u, axis=1)
                N = Matrix(N)
                w2 = []
                for i in tuple(linsolve(N)):
                    for j in i:
                        w2.append(j)
                w1.append(w2)
            N = Matrix(w1)
#            display(N.T)
            return np.trace(N.T)
        else:
            return 0
    def character_p_homology(self, p, P):
        return self.character_kernel(p, P) - self.character_image(p + 1, P)
    def specific_function(self, n):
        w = list_partition(n)
        M = form_matrix_yt(w)
        card = math.factorial(n)
        vec_dic = {}
        for k in range(self.dimension()+1):
            D = {}
            u = []
            v = []
            for h in w:
                u.append(self.character_p_homology(k, make_permutation(h)))
                v.append(size_conjugacy_class(h))
            for i in range(M.shape[0]):
                Ip = 0
                for j in range(M.shape[1]):
                    Ip = Ip + M[i,j]*u[j]*v[j]
                Ip = Ip/card
                D[tuple(w[i])]=Ip
            vec_dic[k] = D
        return vec_dic
    def character_matrix_permutation(self, P, p):
#        display(1)
        v = self.group_of_oriented_p_chains_op(p)
#        display(2)
        v1 = permutation_in_simplex(v,P)
        M = zeros(len(v.dic.keys()),len(v.dic.keys()))
        i = 0
        for u1 in v.dic.keys():
#            display(3)
            j = 0
            for u2 in v1.dic.keys():
                if (eq_elements(u1,u2) == True):
                    M[i,j] = (v1.dic)[u2]
                j = j + 1
            i = i + 1
        return np.trace(M)
    def specific_function_1(self, n):
        w = list_partition(n)
        M = form_matrix_yt(w)
        card = math.factorial(n)
        vec_dic = {}
        for k in range(self.dimension()+1):
            D = {}
            u = []
            v = []
            for h in w:
                u.append(self.character_matrix_permutation(make_permutation(h), k))
                v.append(size_conjugacy_class(h,n))
            for i in range(M.shape[0]):
                Ip = 0
                for j in range(M.shape[1]):
                    Ip = Ip + M[i,j]*u[j]*v[j]
                Ip = Ip/card
                D[tuple(w[i])]=Ip
            vec_dic[k] = D
        return vec_dic
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
                c1 = 0
                if (j != 0):
                    if (v[i][j] == v[i][j-1]):
                        c=1
                        c1 = c1 + 1
                if (j != (len(v[i])-1)):
                    if (v[i][j] == v[i][j+1]):
                        c=1
                        c1 = c1 + 1
                if (i != 0):
                    if (v[i][j] == v[i-1][j]):
                        c=1
                if (i != (len(v)-1)):
    #            if (len(v[i+1]) <= len(v[i])):
                    if (j < len(v[i+1])):
                        if (v[i][j] == v[i+1][j]):
                            c=1
                            c1 = c1 + 1
                    if (j < (len(v[i+1]) - 1)):
                        if (v[i][j] == v[i+1][j+1]):
                            c1 = c1 + 1
                if ((c == 0) and (self.rho[v[i][j]-1]>1)):
                    return False
                if (c1 == 3):
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
        D1 = []
        if (D != []):
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
def boundary_op(v, G):
    sc = SimplicialComplex(G)
    p = len(list(v.dic.keys())[0]) - 1
    s = Group_p_chains([],[])
    if (p != 0):
        ve = sc.group_of_oriented_p_chains_op(p)
        for u in v.dic.keys():
            for k in ve.dic.keys():
                if (eq_elements(k,u) == True):
                    if (orientation_function(k,u,p) == True):
                        c = 0
                        for i in k:  
                            w = list(k).copy()
                            w.remove(i)
                            s1 = Group_p_chains([tuple(w)],[abs(v.dic[u])])
                            if (np.sign((v.dic[u])*(-1)**c) < 0):
                                s = s - s1
                            else:
                                s = s + s1
                            c = c+1
                    else:
                        c = 0
                        for i in k:  
                            w = list(k).copy()
                            w.remove(i)
                            s1 = Group_p_chains([tuple(w)],[abs(v.dic[u])])
                            if (np.sign((v.dic[u])*(-1)**(c+1)) < 0):
                                s = s - s1
                            else:
                                s = s + s1
                            c = c+1
        return s
    else:
        return s
def form_matrix_yt(w):
    M = zeros(len(w),len(w))
    for i in range(len(w)):
        for j in range(len(w)):
            M[i,j] = YoungTableaux(w[i],w[j]).CMNR()
    return M
def list_partition(n):
    p = IntegerPartition([n])
    w = []
    while list(p.args[1]) not in w:
        w.append(list(p.args[1]))
        p = p.next_lex()
    return w
def eq_elements(a, b):
    if (isinstance(a, int) == True):
        return a == b
    if (isinstance(a[0], int) == True):
        return (set() == set(a).difference(set(b)))
    else:      
        for i in range(len(a)):
            test = False
            for j in range(len(b)):
                if (eq_elements(a[i],b[j]) == True):
                    test = True
            if (test == False):
                return False
        else:
            return True
def orientation_function(a,b,p):
    if (p == 0):
        return True
    else:
        v = np.zeros((len(a),), dtype = int)
        for i in range(len(a)):
            for j in range(len(b)):
                if (eq_elements(a[i],b[j]) == True):
                    v[j] = i
        P = Permutation(v)
        return P.is_even
def grafica_de_emparejamiento(n):
    k_n = nx.complete_graph(n)
    G = nx.Graph()
    for i in k_n.edges():
        G.add_node(i)
    w = []
    for i in k_n.edges():
        for j in k_n.edges():
            if ((j[0] not in i) and (j[1] not in i) and ((i,j) not in w) and ((j,i) not in w)):
                w.append((i,j))
                G.add_edge(i,j)
    return G
def set_oriented_p_simplices(simplicie):
        return list(permutations(simplicie))
def elementary_chain_f(simplicie):
    ec = Group_p_chains([], [1])
    for x in set_oriented_p_simplices(simplicie):
        if (orientation_function(simplicie, x, len(x)-1) == True):
            ec = ec + Group_p_chains([x], [1])
        else:
            ec = ec - Group_p_chains([x], [1])
    return ec
def clique_graph(g, cmax=math.inf):
    ite = nx.find_cliques(g)
    cliques = []
    K = nx.Graph()
    while True:
        try:
            cli = next(ite)
            cliques.append(frozenset(cli))
            if len(cliques) > cmax:
                return None
        except StopIteration:
            break
    K.add_nodes_from(cliques)
    clique_pairs = itert.combinations(cliques, 2)
    K.add_edges_from((c1, c2) for (c1, c2) in clique_pairs if c1 & c2)
    G1 = nx.Graph()
    for i in K.nodes():
        G1.add_node(tuple(sorted(i)))
    for i in K.edges():
        G1.add_edge(tuple(sorted(i[0])),tuple(sorted(i[1])))
    return G1
def tuple_permutation(v,P):
    u = []
    w = list(v).copy()
    test = True
    for i in range(len(v)):
        if (isinstance(v[i], int) == True):
            if (v[i] in P):
                w[i] = P(v[i])
        else:
            u.append(tuple_permutation(tuple(v[i]),P))
            test = False
    if (test == True):
        return tuple(w)
    else:
        return tuple(u)
def convert_to_int(v,k):
    w=[]
    for i in v:
        u=[]
        for j in i:
            if (sympify(j*k).is_integer == False):
                return convert_to_int(v,int(solve(x*j + np.sign(j)*(-1))[0]))
            else:
                u.append(j*k)
        w.append(u)
    return w
def nullspace(A):
    u = A.nullspace()
    w= []
    for g in u:
        v=[]
        for i in g:
#            i = sympify(i)
#            display(type(i))
            v.append(i)
        w.append(v)
    w = convert_to_int(w,1)
    if (w == []):
        return [np.zeros((A.shape[1],), dtype = int)]
    else:
        return w
def columnspace(A):
    u = A.columnspace()
    w= []
    for g in u:
        v=[]
        for i in g:
            v.append(i)
        w.append(v)
    w = convert_to_int(w,1)
    if (w == []):
        return [np.zeros((A.shape[0],), dtype = int)]
    else:
        return w
def permutation_in_simplex(vec, P):
    s = Group_p_chains([],[])
    if (vec.dic != {}):
        v = list(vec.dic.keys())
        p = len(list(vec.dic.keys())[0]) - 1
        ve = vec
        for a in v:
            if (isinstance(a, int) == True):
                return vec
            else:
                w = tuple_permutation(a,P)
                for b in list(ve.dic.keys()):
                    if (eq_elements(b,w) == True):
                        if (orientation_function(b,w,p) == True):
                            s = s + Group_p_chains([b],[vec.dic[a]])
                        else:
                            s = s - Group_p_chains([b],[vec.dic[a]])
        return s
    else:
        return s
def tuple_sorted(a):
    if (isinstance(a, int) == True):
        return a
    if (isinstance(a[0], int) == True):
        return sorted(a)
    else:
        w = []
        for b in a:
            w.append(tuple(tuple_sorted(b)))
        return tuple(sorted(tuple(w)))
def permutation_in_simplex_1(vec, P):
    s = Group_p_chains([],[])
    if (vec.dic != {}):
        v = list(vec.dic.keys())
        p = len(list(vec.dic.keys())[0]) - 1
        ve = vec
        for a in v:
            if (isinstance(a, int) == True):
                return vec
            else:
                w = tuple_permutation(a,P)
                b = tuple_sorted(w)
                if (orientation_function(b,w,p) == True):
                    s = s + Group_p_chains([tuple_sorted(w)],[vec.dic[a]])
                else:
                    s = s - Group_p_chains([tuple_sorted(w)],[vec.dic[a]])
        return s
    else:
        return s
def permutation_in_simplex_es(vec, P):
    s = Group_p_chains([],[])
    if (vec.dic != {}):
        v = vec.dic
        p = len(list(vec.dic.keys())[0]) - 1
        ve = vec
        for a in v:
            if (isinstance(a, int) == True):
                return vec
            else:
                if (v[a] != 0):
                    w = tuple_permutation(a,P)
                else:
                    w = a
                    for b in ve.dic:
                        s = s + Group_p_chains([b],[0])
                        if (ve.dic[b] != 0):
                            if (eq_elements(b,w) == True):
                                if (orientation_function(b,w,p) == True):
                                    s = s + Group_p_chains([b],[vec.dic[a]])
                                else:
                                    s = s - Group_p_chains([b],[vec.dic[a]])
        return s
    else:
        return s
def size_conjugacy_class(partition,n):
    aux1=1
    c=0
    aux=partition[0]
    flag = 1
    c2 = 0
    for j in range(len(partition)):
        if (aux == partition[j]):
            c = c + 1
            flag = 1
        else:
            aux1 = aux1*(partition[j-1]**c)*(math.factorial(c))
            aux = partition[j]
            c = 1
            flag = 0
            c2 = j
    if (flag == 1):
        aux1 = aux1*(partition[j-1]**c)*(math.factorial(c))
    else:
        aux1 = aux1*(partition[j]**c)*(math.factorial(c))
    card = (math.factorial(n))/aux1
    return int(card)
def make_permutation(partition):
    P = Permutation()
    c = 0
    for j in range(len(partition)):
        a = []
        for h in range(partition[j]):
            a.append(c)
            c = c + 1
        if (c == 1):
            P1 = Permutation()
            c = 0
        else:
            P1 = Permutation([a])
        P = P*P1
    return P
