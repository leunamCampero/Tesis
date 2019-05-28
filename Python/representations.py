from sympy import *
import sympy as sp
from sympy import sympify
from sympy import solve
from sympy.abc import x
import networkx as nx
import numpy as np
from itertools import combinations
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
    """A class of matricial representation of a group.

    ...
    
    Attributes:
    
        d (dict): A dict that must contains the map of a group into the group
            of nonsingular linear transformations of some finite dimensional
            vector space.
        G (sympy.combinatorics.perm_groups.PermutationGroup): The group. 
            ..Note:: For our purposes we will work with this class of groups.
        n (int): The degree of the group.
        
    """
    def __init__(self, d, G, n):
        '''Define a matrix representation.
        
        Args:
            d (dict): A dict that must contains the map of a group into the group
                of nonsingular linear transformations of some finite dimensional
                vector space.
            G (sympy.combinatorics.perm_groups.PermutationGroup): The group. 
        
            n (int): The degree of the group.
            
        '''
        self.map = d
        self.group = G
        self.degree = n

    def character(self):
        """Returns the character of a representation for every element in the group.
        
        Returns
            dict: A dictionary with the character of the matrix representation
            for every element in the group.
        
        Examples:
            To calculate the character of a matrix representation
            use ``MatrixRepresentation.character()``, in this case
            we will help us of the ``regular representation(G)``.
            
            >>> G=SymmetricGroup(2)
            >>> rr=regular_representation(G)
            >>> print(rr.character())
            {Permutation(1): 2, Permutation(0, 1): 0}
        
        """
        return dict([(g, self.map[g].trace()) for g in self.group.elements])
    
    def is_unitary(self):
        """Returns if the matrix representation is unitary.
        
        Returns
            bool: True if the matrix representation is unitary, False otherwise.
        
        Examples:
            To see if the representation is unitary use 
            ``MatrixRepresentation.is_unitary()``, in this case
            we will help us of the ``regular representation(G)``.
            
            >>> G=SymmetricGroup(3)
            >>> rr=regular_representation(G)
            >>> print(rr.is_unitary())
            True
        
        """
        for g in self.group.elements:
            if sp.expand(self.map[g].H*self.map[g]) != sp.eye(self.degree):
                return False
        else:
            return True
        
class P_chains:
    """A class used to do and operate p-chains.

    ...
    
    Attributes:    
        keys (list): A list with the elements of the p-chains
        values (list): A list with the coefficient for every element in the p-chains

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
            To make a p-chain, use the ``P_chains(list1,list2)`` class.  A p-chain is 
            constructed by providing a list1 of p-simplex and a list2 with 
            their coefficients, i.e.
            
            >>> P = P_chains([(0,1,2,3)],[1])
            >>> Q = P_chains([(0,1,2),(0,1,3)],[-1,2])
            >>> print(P.dic)
            {(0, 1, 2, 3): 1}
            >>> print(Q.dic)
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
        """Sums two p-chains.
        
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
            >>> print(R.dic)
            {(0, 1, 2): 2, (0, 1, 3): 5}
            >>> print(L.dic)
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
        """Subtracts two p-chains.
        
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
            >>> print(R.dic)
            {(3, 4, 5): 3, (1, 8, 9): -1}
            >>> print(L.dic)
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
        '''Returns if the two P_chains are equal
        
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
            >>> print(P == Q)
            True
            >>> print(R == L)
            False
                
            ..Note:: R and L are not equal even though they only are 
            distint in orientation, moreover, in this class the 
            orientation is not defined yet.
            
        '''
        return self.dic == other.dic
    
    def __ne__(self, other):
        '''Returns if the two P_chains are not equal
        
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
            >>> print(P != Q)
            True
            >>> print(R != L)
            False
            
        '''
        return not self.__eq__(other)
    
    def mul_esc(self, esc):
        '''Returns if the two P_chains are not equal
        
        Args:
            other ( __main__.P_chains): Other p-chain.
        
        Returns
            bool: The return value. True for success, False otherwise.  
            
        Examples:
            To know if two P_chains are equal use ``!=``.
            
            >>> P = P_chains([(7,8,9),(10,11,12)],[3,2])
            >>> Q = P_chains([(0,1,4,2,6)],[-5])
            >>> print(P.mul_esc(3).dic)
            {(7, 8, 9): 9, (10, 11, 12): 6}
            >>> print(Q.mul_esc(-1).dic)
            {(0, 1, 4, 2, 6): 5}
        
        '''
        aux = P_chains([],[])
        for x in self.dic:
            aux.dic[x] = esc*self.dic[x]
        return aux    
    
class SimplicialComplex:
    """A class to make simplicial complex asociated with a graph.

    ...

    Attributes:
        G (networkx.classes.graph.Graph): A graph used to build a simplicial complex.

    """

    def __init__(self, G):
        '''Saves the graph and their nodes.
        
        Args:
            G (networkx.classes.graph.Graph): A graph used to build a simplicial complex.
            
        Raises:
            AttributeError: If G is not a graph.
            
        Examples:
            To make a simplicial complex asociated with a graph, 
            use the ``SimplicialComplex`` class. We need a graph G.
            
            >>> G = nx.complete_graph(5)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.vertices)
              [0, 1, 2, 3, 4]
 
        
        '''
        self.G = G
        self.vertices = []
        for x in self.G.nodes():
            self.vertices.append(x)
            
    def faces(self):
        """Makes the faces of a simplicial complex.
        
        A simplicial complex must contains every face of a simplex
        and the intersection of any two simplexes of G is a face of
        each of them. 

        Returns:
            list: A list of the faces of a simplex.

        Examples:
            To create the faces of a simplical complex use, 
            ``SimplicialComplex.faces()``.
            
            >>> G = nx.complete_graph(3)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.faces())
            [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
        
            .. Note:: The faces are sorted by their dimension.
            
            """
            
        faceset = []
        for face in list(nx.enumerate_all_cliques(self.G)):
            faceset.append(tuple(face))
        return faceset
    
    def p_simplex(self, p):
        """Creates a list of the faces of a simplex with dimension p.
        
        Args:
            p (int): The dimension of the faces.

        Returns:
            list: A list of the faces of a simplex with dimension p.
            
        Examples:
            The p-simplices are done with 
            "SimplicialComplex.p_simplex(p)".
            
            >>> G = nx.complete_graph(3)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.faces())
            [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
            >>> print(sc.p_simplex(0))
            [(0,), (1,), (2,)]
            >>> print(sc.p_simplex(2))
            [(0, 1, 2)]
            >>> print(sc.p_simplex(1))
            [(0, 1), (0, 2), (1, 2)]
            >>> print(sc.p_simplex(5))
            []
            
            .. Note:: If there are not faces of dimension p, 
            the method return a empty list like in  
            ``sc.p_simplex(5)``.
            
        """
        
        return list(filter(lambda face: (len(face) == p+1) , self.faces()))
    
    def dimension(self):
        """Gives the dimension of a simplicial complex.

        Returns:
            a - 1 (int): The dimension of the simplicial complex.
            
        Raises:
            Return ``-1`` if the graph is empty.
        
        Examples:
            To use the method dimension write 
            ``SimplicialComplex.dimension()``.
            
            >>> G = nx.petersen_graph()
            >>> sc = SimplicialComplex(G)
            >>> print(sc.dimension())
            1
        
        """
        a = 0
        for x in self.faces():
            if (len(x) > a):
                a = len(x) 
        return a-1
    
#    def p_simplex(self, k):
#        p_simplex = []
#        for x in self.p_simplex(k):
#            p_simplex.append(x)
#        return p_simplex
#    def elementary_chain(self, simplex):
#        """Give the p-chains with their repective orientation.
#
#        Args:
#            simplex (tuple): A tuple which orientation is taken positive.
#                
#        Returns:
#            __main__.P_chains: A new p-chains that is.
#            
#
#        """
#        ec = P_chains([], [])
#        for x in set_oriented_p_simplices(simplex):
#            if (orientation_function(tuple_sorted(simplex), x, len(simplex)-1) == True):
#                ec = ec + P_chains([x], [1])
#            else:
#                ec = ec - P_chains([x], [1])
#        return ec
#    def oriented_p_chains(self, k):
#        if ((k<0) or (k>self.dimension())):
#            return 0
#        else:
#            c_p = P_chains([], [1])
#            for x in self.p_simplex(k):
#                c_p = c_p + self.elementary_chain(tuple_sorted(x))
#            return c_p
    def basis_group_oriented_p_chains(self, p):
        """Gives a basis for the group of oriented p-chains.

        Args:
            p (int): Indicated the dimension of the p-simplex.
            
        Returns:
            __main__.P_chains: A new p-chains that contains the basis of the group.
        
            .. Note:: To every element is given the coefficiente ``1`` 
            by default, this is because the p-simplex are sorted with the
            lexicographical order, i.e, this orientation is taken positive.
        
        Raises:
            AttributeError: If p is lower than zero or bigger than the dimension of the
            simplicial complex dimension.
        
        Examples:
            To create a basis for the group of oriented p-chains, use 
            ``SimplicialComplex.basis_group_oriented_p_chains(p)``.
            
            >>> G = matching_graph(3)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.basis_group_oriented_p_chains(0).dic)
            {((0, 1),): 1, ((0, 2),): 1, ((1, 2),): 1}
                
             .. Note:: We use the function ``matching_graph`` which
             will be explain after.
            
        """
        if ((p<0) or (p>self.dimension())):
            return 0
        else:
            c_p = P_chains([], [])
            for x in self.p_simplex(p):
                c_p = c_p + P_chains([tuple(tuple_sorted(x))], [1])
            return c_p
#    def p_homology_group_dimention(self, k):
#        vk = self.simplex()[k]
#        vkf = self.n_faces(k-1)
#        M = zeros(len(vkf),len(vk.dic))
#        j=0
#        for u in list(vk.dic.keys()):
#            d={u: vk.dic[u]}
#            for a in list((boundary_op(d).dic).keys()):
#                i=0
#                for w in list(vkf):
#                    if (a == w):
#                        M[i,j]=(boundary_op(d).dic)[w]
#                    i=i+1
#            j=j+1
#        dimKe = len(M.rref()[1])
#        vk1 = self.simplex()[k+1]
#        vkf1 = self.n_faces(k)
#        N = zeros(len(vkf1),len(vk1.dic))
#        j=0
#        for u in list(vk1.dic.keys()):
#            d={u: vk1.dic[u]}
#            for a in list((boundary_op(d).dic).keys()):
#                i=0
#                for w in list(vkf1):
#                    if (a == w):
#                        N[i,j]=(boundary_op(d).dic)[w]
#                    i=i+1
#            j=j+1
#        dimIm = len((N.T).rref()[1])
#        dimH = dimKe - dimIm
#        return dimKe, dimIm, dimH
    def representate_in_simplex(self, vec, P):
        s = P_chains([],[])
        if (vec.dic != {}):
            v = list(vec.dic.keys())
            p = len(list(vec.dic.keys())[0]) - 1
            ve = self.basis_group_oriented_p_chains(p)
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
    """
        Before see the documentation of the next methods, you must
        see the documentation of the functions.
    """
    def matrix_simmetric_representate(self, p):
        """Give the matrix associated to the boundary operator.

        Args:
            p (int): Determine with basis of the p-simplex will be used.
        Returns:
            A matrix if p is bigger than -1, and lower than the dimension
            of the simplicial complex, return False otherwise.
                
        Examples:
            To compute the matrix associated to the boundary operator, use
            ``SimplicialComplex.matrix_simmetric_representate(p)``.
                
            >>> n=3
            >>> G = matching_graph(n)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.matrix_simmetric_representate(0))
            Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            
            .. Note:: This matrix will be so useful to our purposes.
        
        """
        if (p >0 and (p <= self.dimension()) ):
            v = self.basis_group_oriented_p_chains(p)
            p = p - 1
            ve = self.basis_group_oriented_p_chains(p)
            M = zeros(len(ve.dic),len(v.dic))
            j = 0
            for u1 in list(v.dic.keys()):
                d =  P_chains([u1],[v.dic[u1]])
                for u2 in list(boundary_op_n(d).dic.keys()):
    #                display(boundary_op(d, self.G).dic)
                    i = 0
                    for w in list(ve.dic.keys()):
                        if (w == u2):
#                            if (orientation_function(u2,tuple(w),p) == True):
                                M[i,j] = int((boundary_op_n(d).dic)[u2])
#                            else:
#                                M[i,j] = int((boundary_op_n(d).dic)[u2])*(-1)
                        i = i + 1
                j = j + 1
            return M
        else:
            if (p == 0):
                return eye(len(list(self.basis_group_oriented_p_chains(0).dic.keys())))
            else:
                return False
                
    def kernel_boundary_op(self, p):
#        display(p,self.dimension())
        if ((p > 0) and (p <= self.dimension())):
            u = nullspace(self.matrix_simmetric_representate(p))
#            display(u)
            if (u != False):
                v = self.basis_group_oriented_p_chains(p)
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
                return [self.basis_group_oriented_p_chains(p)]
            else:
                return False
    def image_boundary_op(self, p):
#        display(p, self.dimension())
        if ((p > 0) and (p <= self.dimension())):
            u = columnspace(self.matrix_simmetric_representate(p))
            if (u != False):
                v = self.basis_group_oriented_p_chains(p-1)
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
        """Gives character of a basis of the kernel under a Permutation.

        Args:
            p (int): Indicated the dimension of the p-simplex.
            P (sympy.combinatorics.permutations.Permutation): The permutation.
            
        Returns:
            int: The character of a basis of the kernel under a Permutation,
            in our case, we are interest in representates of the conjugacy 
            class of the symmetric group.
        
        Examples:
            To get the character of a permutation acting on a basis
            of the kernel, use ``SimplicialComplex.character_kernel(p,Permutation)``.
            
            >>> n=4
            >>> G1 = matching_graph(n)
            >>> G = clique_graph(G1)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.character_kernel(1,Permutation(0,1)))
            0
            >>> print(sc.character_kernel(1,Permutation(0,1,2,3)))
            0
            
        """
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
            he = self.basis_group_oriented_p_chains(p)
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
                        if (i == j):
#                        if (eq_elements(i, j) == True):
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
        """Gives character of a basis of the image under a Permutation.

        Args:
            p (int): Indicated the dimension of the p-simplex.
            P (sympy.combinatorics.permutations.Permutation): The permutation.
            
        Returns:
            int: The character of a basis of the image under a Permutation,
            in our case, we are interest in representates of the conjugacy 
            class of the symmetric group.
        
        Examples:
            To get the character of a permutation acting on a basis
            of the image, use ``SimplicialComplex.character_image(p,Permutation)``.
            
            >>> n=5
            >>> G = matching_graph(n)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.character_image(1,Permutation(0,1)))
            3
            >>> print(sc.character_image(1,Permutation()))
            9
            
        """
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
            he = self.basis_group_oriented_p_chains(p-1)
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
                        if (i == j):
#                        if (eq_elements(i, j) == True):
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
        """Gives character of the pth homology.
        
        Args:
            p (int): Indicated the dimension of the p-simplex.
            P (sympy.combinatorics.permutations.Permutation): The permutation.
            
        Returns:
            int: The character of the pth homology.
        
        Examples:
            To get the character of the pth homology use 
            ``SimplicialComplex.character_p_homology(p, Permutation)``.
            
            >>> n=6
            >>> G1 = matching_graph(n)
            >>> G = clique_graph(G1)
            >>> sc = SimplicialComplex(G)
            >>> print(sc.character_p_homology(1,Permutation(0,1)))
            0
            >>> print(sc.character_p_homology(1,Permutation()))
            16
            
            ..Note:: The funcion only is the subtract of the
                character of the kernel (dimension p) and the 
                character of the image (dimension p-1).
        
        """
        return self.character_kernel(p, P) - self.character_image(p + 1, P)
   
    def specific_function(self, n):
        """Returns a dictionary showing the descomposition into irreducibles.
        
        Args:
            n (int): Indicated what symmetric group act on the p-simplex.
           
        Returns:
            dict: A dictionary that contains show the descomposition
            into irreducibles, i.e the reduce homologies of the 
            simplicial complex determinated by a graph.
        
        Examples:
            To get the reduce homologies of the simplicial complex by a 
            graph, use ``SimplicialComplex.specific_function(n)``.
            
            >>> n=4
            >>> G1 = matching_graph(n)
            >>> G = clique_graph(G1)
            >>> sc1 = SimplicialComplex(G1)
            >>> print(sc1.specific_function(n))
            {0: {(4,): 1, (1, 1, 1, 1): 0, (2, 1, 1): 0, (2, 2): 1, (3, 1): 0}, 1: {(4,): 0, (1, 1, 1, 1): 0, (2, 1, 1): 0, (2, 2): 0, (3, 1): 0}}
            >>> sc2 = SimplicialComplex(G)
            >>> print(sc2.specific_function(n))
            {0: {(4,): 1, (1, 1, 1, 1): 0, (2, 1, 1): 0, (2, 2): 1, (3, 1): 0}}
        
        """
        w = partitions_list(n)
        M = form_matrix_yt(w)
        card = math.factorial(n)
        vec_dic = {}
        for k in range(self.dimension()+1):
            D = {}
            u = []
            v = []
            for h in w:
                u.append(self.character_p_homology(k, make_permutation(h)))
                v.append(size_conjugacy_class(h,n))
            for i in range(M.shape[0]):
                Ip = 0
                for j in range(M.shape[1]):
                    Ip = Ip + M[i,j]*u[j]*v[j]
                Ip = Ip/card
                D[tuple(w[i])]=Ip
            vec_dic[k] = D
        return vec_dic
    def character_matrix_permutation(self, p, P):
#        display(1)
        v = self.basis_group_oriented_p_chains(p)
#        display(2)
        v1 = permutation_in_simplex_test(v,P)
#        display(3)
        M = zeros(len(v.dic.keys()),len(v.dic.keys()))
        i = 0
        for u1 in v.dic.keys():
#            display(4)
            j = 0
            for u2 in v1.dic.keys():
                if (u1 == u2):
                    M[i,j] = (v1.dic)[u2]
                j = j + 1
            i = i + 1
        return np.trace(M)
    def specific_function_1(self, n):
        w = partitions_list(n)
        M = form_matrix_yt(w)
        card = math.factorial(n)
        vec_dic = {}
        for k in range(3):
            D = {}
            u = []
            v = []
            for h in w:
                u.append(self.character_matrix_permutation(k,make_permutation(h)))
                v.append(size_conjugacy_class(h,n))
            for i in range(M.shape[0]):
                Ip = 0
                for j in range(M.shape[1]):
                    Ip = Ip + M[i,j]*u[j]*v[j]
                Ip = Ip/card
                D[tuple(w[i])]=Ip
            vec_dic[k] = D
        return vec_dic
    def specific_function_2(self, p, n):
        w = partitions_list(n)
        print(1)
        M = form_matrix_yt(w)
        card = math.factorial(n)
        print(2)
        v = self.basis_group_oriented_p_chains(p)
        print(3)
        leng = len(v.dic.keys())
        print(4)
        M1 = zeros(leng,leng)
        print(5)
        vec_dic = {}
        D = {}
        au = []
        av = []
        for h in w:
            print(6)
            M2 = M1.copy()
            print(7)
            v1 = permutation_in_simplex_test(v,make_permutation(h))
            print(8)
            i = 0
            for u1 in v.dic.keys():
    #            display(4)
                j = 0
                for u2 in v1.dic.keys():
                    if (u1 == u2):
                        M2[i,j] = (v1.dic)[u2]
                    j = j + 1
                i = i + 1
            traza = np.trace(M2)
            au.append(traza)
            av.append(size_conjugacy_class(h,n))
            print(traza, h)
        for i in range(M.shape[0]):
            Ip = 0
            for j in range(M.shape[1]):
                Ip = Ip + M[i,j]*au[j]*av[j]
            Ip = Ip/card
            D[tuple(w[i])]=Ip
        vec_dic[p] = D
        return vec_dic
    def specific_function_5(self, p, n, valores):
        w=[[7],[1, 1, 1, 1, 1, 1, 1],[2, 1, 1, 1, 1, 1],[2, 2, 1, 1, 1],[2, 2, 2, 1],[3, 1, 1, 1, 1],[3, 2, 1, 1],[3, 2, 2],[3, 3, 1],[4, 1, 1, 1],[4, 2, 1],[4, 3],[5, 1, 1],[5, 2],[6, 1]]
#        w = partitions_list(n)
        M = form_matrix_yt(w)
        card = math.factorial(n)
        vec_dic = {}
        D = {}
        au = valores
        av = []
        for h in w:
            av.append(size_conjugacy_class(h,n))
        for i in range(M.shape[0]):
            Ip = 0
            for j in range(M.shape[1]):
                Ip = Ip + M[i,j]*au[j]*av[j]
            Ip = Ip/card
            D[tuple(w[i])]=Ip
        vec_dic[p] = D
        return vec_dic

class YoungTableaux:
    """A class to compute irreducible character values of a symmetric group.

    ...
    
    Attributes:
        p_lambda (list): A list that represent the first partition.
        p_rho (list): A list that represent the second partition.

    """
    
    def __init__(self, p_lambda, p_rho):
        '''
        Args:
            p_lambda (list): A list that represent the first partition.
            p_rho (list): A list that represent the second partition.
        
        '''
        self.p_lambda = p_lambda
        self.p_rho = p_rho
        
    def choose_tableaux(self, v):
        """A method to identify if a given list is a border-strip tableaux or not.
        
        Args:
            v (list): A list of list that is the candidate to be a border-strip tableaux.
        
        Returns:
            True if the list is a border-strip tableaux, False otherwise.
            
        Raises:
            IndexError: If the list given is not consistent with the partitions given.
        
        Examples:
            To see if a list given is a border-strip tableaux, you must 
            use ``YoungTableaux.choose_tableaux(list)``.
            
            >>> YT = YoungTableaux([2,1],[1,1,1])
            >>> print(YT.choose_tableaux([[1,1,1],[1,2]]))
            True
            >>> print(YT.choose_tableaux([[1,1,1],[2,1]]))
            False
            >>> print(YT.choose_tableaux([[1,1,1],[1,1]]))
            False
            
            .. Note:: The examples given must be interpreted 
           like:
            Tableaux 1:    
                        | 1 | 1 | 1 |
                        | 1 | 2 |
            Tableaux 2: 
                        | 1 | 1 | 1 |
                        | 2 | 1 |
            Tableaux 1: 
                        | 1 | 1 | 1 |
                        | 1 | 1 |
                        
            In the two first examples we use two partitions of 3, 
            and the third example is done even though the second partition 
            is a partition of 2, for our purposes that mistake is not done, 
            because we give the correct partitions.
         
        """
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
                if ((c == 0) and (self.p_rho[v[i][j]-1]>1)):
                    return False
                if (c1 == 3):
                    return False
        else:
            return True
    def MNR(self):
        """A method to generate all the border-strip tableauxs according to the partitions given.
        
        Returns:
            D1 (list): A list that contains the border-strip tableauxs.
            
        Examples:
            To see the border-strip tableauxs, you must 
            use ``YoungTableaux.MNR()``.
            
            YT1 = YoungTableaux([2,2,2,1],[3,3,1])
            YT2 = YoungTableaux([4,1],[3,2])
            YT3 = YoungTableaux([2,2,1,1],[6])
            print(YT1.MNR())
            [[[1, 1], [1, 2], [2, 2], [3]], [[1, 2], [1, 2], [1, 2], [3]]]
            print(YT2.MNR())
            [[[1, 1, 2, 2], [1]]]
            print(YT3.MNR())
            []
            
            .. Note:: In the two first example the method found two list 
            that are border-strip tableaux, the which could be interpreted 
            like:   
            | 1 | 1 |
            | 1 | 2 |
            | 2 | 2 
            | 3 |
            and
            | 1 | 2 |
            | 1 | 2 |
            | 1 | 2 |
            | 3 |.
            And for the second example the method only found a list, 
            and his parallel border-strip tableaux looks like:
            | 1 | 1 | 2 | 2 |
            | 1 | 
             And for the third example there are not any border-strip
            tableaux.
        
        """
        p=[]
        i=1
        for h in self.p_rho:
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
            for p in self.p_lambda:
                u=[]
                for i in range(c,c+p):
                    u.append(v[i])
                w.append(u)
                c=c+p
            if (self.choose_tableaux(w) == True):
                D.append(w)
        D1 = []
        if (D != []):
            D1=[D[0]]
            for k1 in D:
                if k1 not in D1:
                    D1.append(k1)
        return D1
    
    def heights(self):
        """A method to calculate the heights i.e the sum of the heights of the border strips.
        
        Returns:
            He (list): A list that contains heights.
            
        Examples:
            To see the heights of the border-strip tableauxs, you must 
            use ``heights.MNR()``.
            
            >>> YT1 = YoungTableaux([5,2,1],[3,3,1,1])
            >>> print(YT1.heights())
            [1, 1, 1, 2, 2, 3]
            .. Note:: The border-strip tableauxs generated by these partitions 
            are:
            T1:
                | 1 | 1 | 1 | 3 | 4 |
                | 2 | 2 |
                | 2 |
            T2:
                | 1 | 1 | 2 | 2 | 2 |
                | 1 | 3 |
                | 4 |
            T3:
                | 1 | 1 | 2 | 2 | 2 |
                | 1 | 4 |
                | 3 |
            T4:
                | 1 | 2 | 2 | 2 | 3 |
                | 1 | 4 |
                | 1 |
            T5:
                | 1 | 2 | 2 | 2 | 4 |
                | 1 | 3 |
                | 1 |
            T6:
                | 1 | 2 | 2 | 3 | 4 |
                | 1 | 2 |
                | 1 |.
                
            And the last list are their respective heights, which can be verified
            like follows:
            ht(T1) = 0 + 1 + 0 + 0 = 1
            ht(T2) = 1 + 0 + 0 + 0 = 1
            ht(T3) = 1 + 0 + 0 + 0 = 1
            ht(T4) = 2 + 0 + 0 + 0 = 2
            ht(T5) = 2 + 0 + 0 + 0 = 2
            ht(T6) = 2 + 1 + 0 + 0 = 3
            
            And the results above coincide with the list [1, 1, 1, 2, 2, 3].
            
        """
        H = self.MNR()
        He = []
        for h in H:
            he=[]
            for i in range(0,len(self.p_rho)):
                c = 0
                for g in h:
                    if ((i+1) in g):
                        c = c+1
                he.append(c-1)
            He.append(sum(he))
        return He
    
    def CMNR(self):
        """A method to calculate irreducible character values ​​through the Murnaghan-Nakayama rule.
        
        Returns:
            s (int):  A irreducible character value of a symmetric group according to the given partition.
            
        Examples:
            Here are the last method of the class ``YoungTableaux``. To 
            get the irreducible character value of a symmetric group according
            to the given partition use ``YoungTableaux.CMNR`` (We will use
            the same example that in the method ``heights``. 
            
            >>> YT1 = YoungTableaux([5,2,1],[3,3,1,1])
            >>> print(YT1.heights())
            [1, 1, 1, 2, 2, 3]
            >>> print(YT1.CMNR())
            -2
            
            .. Note:: In the method ``heights`` we saw that for the partitions of 6 given
            there are six such border-strip tableaux, and their heights are:
            [1, 1, 1, 2, 2, 3]
            
            According with the Murnaghan–Nakayama rule the character value is therefore:
            Chi_{3,3,1,1}^{5,2,1} = (-1)^(1) + (-1)^(1) + (-1)^(1) + (-1)^(2) + (-1)^(2) + (-1)^(3) =
                                  = - 1 - 1 - 1 + 1 + 1 - 1 = -2
            
            Like the method ``CMNR`` got.
            
        """
        He = self.heights()
        s=0
        for j in He:
            s = s + (-1)**(j)
        return s
    
def boundary_op(v, G):
    sc = SimplicialComplex(G)
    p = len(list(v.dic.keys())[0]) - 1
    s = P_chains([],[])
    if (p != 0):
        ve = sc.basis_group_oriented_p_chains(p)
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
    """Returns the action of the boundary operator on p-chains.

    Args:
        v ( __main__.P_chains): The p-chain of 

    Returns:
        v ( __main__.P_chains): The (p-1)-chain lower the boundary operator.
         
    Examples:
        To use of the boundary operator, write ``boundary_op_n``.
        
        >>> w = P_chains([(0,),(1,),(2,),(3,)],[1,1,1,1])
        >>> v = P_chains([(0,1,2),(0,1,3),(0,2,3),(1,2,3)],[1,1,1,1])
        >>> u = boundary_op_n(v)
        >>> print(boundary_op_n(w).dic)
        {}
        >>> print(u.dic)
        {(1, 2): 2, (0, 2): 0, (0, 1): 2, (1, 3): 0, (0, 3): -2, (2, 3): 2}
        >>> print(boundary_op_n(u).dic)
        {(2,): 0, (1,): 0, (0,): 0, (3,): 0}
        
        .. Note:: Above w, v are the 0-simplex, 2-simplex of the tetrahedron respectively,
            if v is a p-simplex we denoted the boundary_op_n(v) = \partial_{p}(v), the theory 
            said that \partial_{p-1}(\partial_{p}(v)) = 0, that was checked in 
            the ``boundary_op_n(u)``. In the case when you use 0-simplex, the result is 
            a empty dictionary like in ``boundary_op_n(w)``. 

   """
    p = len(list(v.dic.keys())[0]) - 1
    s = P_chains([],[])
    if (p != 0):
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

def partitions_list(n):
    """Returns a list of the partitions of n.

    Args:
        n (int): A integer that determine the partitions.
    
    Returns:
        w (list): A list of list that are the different partitions of n.
    
    Raises:
        ValueError: If ``n`` is not bigger than zero.
        
    Examples:
        To form all the partitions for the integer ``n``, use 
        ``list_partitions``.
        
        >>> v = partitions_list(3)
        >>> u = partitions_list(4)
        >>> print(v)
        [[3], [1, 1, 1], [2, 1]]
        >>> print(u)
        [[4], [1, 1, 1, 1], [2, 1, 1], [2, 2], [3, 1]]
        
    
    """
    p = IntegerPartition([n])
    w = []
    while list(p.args[1]) not in w:
        w.append(list(p.args[1]))
        p = p.next_lex()
    return w

def form_matrix_yt(w):
    """Returns the a matrix that represent the character table of the symmetric group.

    Args:
        w (list): A list with the partitions for certain symmetric group.
    
    Returns:
        M (<class 'sympy.matrices.dense.MutableDenseMatrix'>): A matrix with the characters 
        of the character table of the symmetric group.
    
    Examples:
        To form the matrix that represent the character table of the
        symmetric group, use ``form_matrix_yt``.
        
        >>> v = partitions_list(3)
        >>> print(form_matrix_yt(v))
        Matrix([[1, 1, 1], [1, 1, -1], [-1, 2, 0]])
        
        .. Note:: The function need a list of the partitions for
        ``n``, then is used the function ``partitions_list.``

    """
    M = zeros(len(w),len(w))
    for i in range(len(w)):
        for j in range(len(w)):
            M[i,j] = YoungTableaux(w[i],w[j]).CMNR()
    return M

def eq_elements(a, b):
    """A function that identify when tuples are equal except by orientation.

    Args:
        a (tuple): The first tuple.
        b (tuple): The second tuple.
    
    Returns:
        bool: True if the tuples are equal except by orientation, 
        False otherwise.
    
    Raises: 
        TypeError: If the tuples don't have the same structure, for 
        example: 
        a = ((0,1),(2,3),(5,6)) 
        b = ((1,0))
        
    Examples:
        To see if two tuples are equal use ``eq_elements``.
        
        >>> a1 = ((0,1),(2,3),(5,6)) 
        >>> b1 = ((0,3),(2,1),(5,6)) 
        >>> a2 = ((0,1),(2,3),(5,6)) 
        >>> b2 = ((6,5),(1,0),(3,2)) 
        >>> print(eq_elements(a1,b1))
        False
        >>> print(eq_elements(a2,b2))
        True

    """
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
    """Determines the orientation of ``b`` taken the orientation of ``a`` positive.

    Args:
        a (tuple): The first tuple.
        b (tuple): The second tuple.
        p (tuple): The dimension of the simplex.
    
    Returns:
        ValueError: If the tuples are not equal under the
        function ``eq_elements``.
            
    Examples:
        To see if two tuples are equal use ``eq_elements``.
        
        >>> a1 = (((0,1),(2,3),(4,6)),((0,1),(2,4),(3,5))) 
        >>> b1 = (((4,2),(1,0),(5,3)),((2,3),(1,0),(6,4))) 
        >>> a2 = ((0,1),(2,3),(5,6)) 
        >>> b2 = ((6,5),(1,0),(3,2)) 
        >>> print(orientation_function(a1,b1,1))
        False
        >>> print(orientation_function(a2,b2,2))
        True

        .. Note:: For ``a1`` and ``b1`` the function receive
        the integer ``1``, and in the case of ``a2`` and ``b2`` 
        receive ``2`` to indentify the dimension of the
        p-simplex, in the practice the class determine this
        number.
        
    """
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
    
def tuple_sorted(a):
    """Sorted tuples of tuples.
    Args:
        a (tuple): A tuple the which will be sorted.
        
    Returns:
        (tuple): The tuple sorted. 
            
    Examples:
        The function ``sorted`` don't sort tuples of tuples, but
        this function can do it, below is showing examples of 
        both functions:
        
        >>> a1 = ((6,5),(1,0),(3,2)) 
        >>> a2 = (((4,2),(1,0),(5,3)),((2,3),(1,0),(6,4))) 
        >>> print(sorted(a1))
        [(1, 0), (3, 2), (6, 5)]
        >>> print(tuple_sorted(a1))
        ((0, 1), (2, 3), (5, 6))
        >>> print(sorted(a2))
        [((2, 3), (1, 0), (6, 4)), ((4, 2), (1, 0), (5, 3))]
        >>> print(tuple_sorted(a2)) 
        (((0, 1), (2, 3), (4, 6)), ((0, 1), (2, 4), (3, 5)))
        
    """
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
    """Determines the orientation of ``b`` taken the orientation of ``a`` positive.

    Args:
        a (tuple): The tuple which will under the Permunation ``p``.
        p (<class 'sympy.combinatorics.permutations.Permutation'>): The Permutation.
    Returns:
        (tuple): The tuple with their elements permutated under
        the permutation ``p``.
            
    Examples:
        To do act the Permutation on the tuple use 
        ``tuple_permutation(tuple)``.
        
        >>> a1 = (0,1,2,3,4)
        >>> a2 = ((2,4),(1,5),(3,0))
        >>> a3 = (((0,1),(2,4),(3,5)),((0,5),(1,3),(2,4)))
        >>> print(tuple_permutation(a1,Permutation(0,1,2)))
        (1, 2, 0, 3, 4)
        >>> print(tuple_permutation(a2,Permutation(1,3)))
        ((2, 4), (3, 5), (1, 0))
        >>> print(tuple_permutation(a3,Permutation(0,1)(2,3)))
        (((1, 0), (3, 4), (2, 5)), ((1, 5), (0, 2), (3, 4)))

        .. Note:: The function return other tuple that represent
        how the Permutation is acting in a natural way in the origin 
        tuple.
    
    """
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

def matching_graph(n):
    """Makes a matching graph since a complete graph.
    
    Args:
        n (int): A integer that to do the complete graph.

    Returns:
        networkx.classes.graph.Graph: The matching graph. 
    
    Raises: 
        NetworkXError: If n is a negative number.

    """
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
    """Makes a clique graph since a matching graph.
    
    Args:
        n (int): A integer that to do the matching graph.

    Returns:
        networkx.classes.graph.Graph: The clique graph. 
    
    Raises: 
        NetworkXError: If n is a negative number.

    """
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

#def convert_to_int(v,k):
#    """From a list return other list where all the entries are integer.
#
#    Args:
#        v (list): The tuple which will under the Permunation ``p``.
#        k (floar): The reciprocal number to convert all the entries to integer.
#    
#    Returns:
#        w (list): A tuple where all the entries are integers.
#            
#    Examples:
#        To do act the Permutation on the tuple use 
#        ``tuple_permutation(tuple)``.
#        
#        >>> a1 = (0,1,2,3,4)
#        >>> a2 = ((2,4),(1,5),(3,0))
#        >>> a3 = (((0,1),(2,4),(3,5)),((0,5),(1,3),(2,4)))
#        >>> tuple_permutation(a1,Permutation(0,1,2))
#        (1, 2, 0, 3, 4)
#        >>> tuple_permutation(a2,Permutation(1,3))
#        ((2, 4), (3, 5), (1, 0))
#        >>> tuple_permutation(a3,Permutation(0,1)(2,3))
#        (((1, 0), (3, 4), (2, 5)), ((1, 5), (0, 2), (3, 4)))
#
#        .. Note:: The function return other tuple that represent
#        how the Permutation is acting in a natural way in the origin 
#        tuple.
#    
#    """
##    k = int(k)
#    w=[]
#    for i in v:
#        u=[]
#        for j in i:
#            u.append(j*k)
#        w.append(u) 
#    print(w)
#    c=1
#    for i in w:
#        for j in i:
#            c=c*j
#            if (j%1 != 0):
##                if ((j*k).is_integer() == False):
##                print(type(j),type(k),type(j*k))
#                print(j,solve(x*j*c + np.sign(j)*(-1))[0])
#                return convert_to_int(w,solve(x*j*c + np.sign(j)*(-1))[0])
#    return w
def nullspace(A):
    """Returns a  ``list`` of column vectors that span the nullspace of the matrix.
    Args:
        A (Matrix): The matrix which we will find the nullspace.
        p (<class 'sympy.combinatorics.permutations.Permutation'>): The Permutation.
    
    Returns:
        (list): A list of list with the generators of the kernel.
            
    Examples:
        To find the nullspace of a matrix, use ``nullspace(A)``. 
        
        >>> M1 = Matrix([[2, 4, 6, 6], [8, 20, 0, 1], [5, 0, 3, 2]])
        >>> M2 = Matrix([[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0,-1,0],[-1,0,0]])
        >>> print(nullspace(M1))
        [[3/16, -1/8, -47/48, 1]]
        >>> print(nullspace(M2))
        [array([0, 0, 0])]
        
        .. Note:: Essentially the function only obtain the nullspace
        with the function ``A.nullspace()`` and returns the trivial kernel
        if ``A.nullspace()`` is a emtpy list.
    
    """
    u = A.nullspace()
    w= []
    for g in u:
        v=[]
        for i in g:
            v.append(i)
        w.append(v)
    if (w == []):
        return [np.zeros((A.shape[1],), dtype = int)]
    else:
        return w
    
def columnspace(A):
    """Returns a ``list`` of column vectors that span the columnspace of the matrix.
    
    Args:
        A (Matrix): The matrix which we will find the columnspace.
        p (<class 'sympy.combinatorics.permutations.Permutation'>): The Permutation.
    
    Returns:
        (list): A list of list with the generators of the columnspace (image).
            
    Examples:
        To find the columnspace of a matrix, use ``columnspace(A)``.
        
        >>> M1 = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        >>> M2 = Matrix([[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0,-1,0],[-1,0,0]])
        >>> print(columnspace(M1))
        [array([0, 0, 0])]
        >>> print(nullspace(M2))
        [array([0, 0, 0])]
        >>> print(columnspace(M2))
        [[1, 0, 0, 0, 0, -1], [0, 1, 0, 0, -1, 0], [0, 0, 1, -1, 0, 0]]
        
        .. Note:: Essentially the function only obtain the columnspace
        with the function ``A.columnspace()`` and returns the trivial image
        if ``A.columnspace()`` is a emtpy list. In the example ``M2`` is noted
        that is right that:
        ``dimension(nullspace(A))``+``dimension(columnspace(A))`` = ``number of columns``.
    
    """
    u = A.columnspace()
    w= []
    for g in u:
        v=[]
        for i in g:
            v.append(i)
        w.append(v)
    if (w == []):
        return [np.zeros((A.shape[0],), dtype = int)]
    else:
        return w
    
def permutation_in_simplex_test(vec, P):
    """Returns a simplex under a permutation.
    
    Args:
        vec ( __main__.P_chains): A p-chain which the permutation will act.
        P ( sympy.combinatorics.permutations.Permutation): The permutation.
    
    Returns:
        (__main__.P_chains): A new p-chain that is the result of the
        permutation acting on the original p-chain ``vec``.
            
    Examples:
        To see how a permutation act on a p-simplex, use 
        ``permutation_in_simplex_test(SimplicialComplex, Permuation)``.
        Also we must check that the boundary operator on a p-simplex
        (\partial_{p}) is well-defined and that (if ``p-simplex`` := \sigma)
        \partial_{p}(-\sigma) = - \partial_{p}(\sigma). For this purpose, it
        suffices to show that the right-hand side of:
            
            \partial_{p}(\sigma) = 
                                 = \partial_{p}([v_{0},...,v_{p}]) =
                                 = \sum_{i=0}^{p}(-1)^{i}[v_{0},...,v_{i},...v_{p}].
               
                (where v_{i} means that the vertex v_{i} is to be deleted
                from the array)
        
        changes sign if we exchange two adjacent vertices in the array
        [v_{0},...,v_{p}] (important step will be explain according with the 
        theory):
        
        >>> u1 = P_chains([(0,1,2,3)],[1])
        >>> u2 = P_chains([(0,2,1,3)],[1])
        ..Note:: The p-simplex in u1 and u2 differ by a sign.
        >>> bu1 = boundary_op_n(u1).dic
        >>> bu2 = boundary_op_n(u2).dic
        >>> print(bu1)
        {(1, 2, 3): 1, (0, 2, 3): -1, (0, 1, 3): 1, (0, 1, 2): -1}
        >>> print(bu2)
        {(1, 2, 3): -1, (0, 1, 3): -1, (0, 2, 3): 1, (0, 1, 2): 1}
        ..Note:: We could see that the result changes sign, like is wanted.
        
        Now se must check that \partial_{p}(\rho(\sigma)) = \rho(\partial_{p}(\sigma))
        (where \rho = Permutation ``P``). For this we will use some p-simplices
        associated with a graph.
        
        >>> n=5
        >>> G = nx.complete_graph(n)
        >>> sc = SimplicialComplex(G)
        >>> sigma = sc.basis_group_oriented_p_chains(1)
        >>> print(sigma.dic)
        {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1, (1, 2): 1, (1, 3): 1, (1, 4): 1, (2, 3): 1, (2, 4): 1, (3, 4): 1}
        >>> bo_sigma=boundary_op_n(sigma)
        >>> rho_bo_sigma=permutation_in_simplex_test(bo_sigma,Permutation(0,1))
        >>> print(rho_bo_sigma.dic)
        {(0,): -2, (1,): -4, (2,): 0, (3,): 2, (4,): 4}
        >>> rho_sigma=permutation_in_simplex_test(sigma,Permutation(0,1))
        >>> bo_rho_sigma=boundary_op_n(rho_sigma)
        >>> print(bo_rho_sigma.dic)
        {(1,): -4, (0,): -2, (2,): 0, (3,): 2, (4,): 4}
        >>> print(rho_bo_sigma == bo_rho_sigma)
        True
        ..Note:: Then for this example the result is the same.
        
        And for the second propertie:
        
        >>> sigma1 = P_chains([(0,1,2)],[1])
        >>> sigma2 = P_chains([(0,1,2)],[-1])
        ..Note:: The simplices differ by the sign.
        >>> w1 = boundary_op_n(sigma1)
        >>> w2 = boundary_op_n(sigma2)
        >>> print(w1.dic)
        {(1, 2): 1, (0, 2): -1, (0, 1): 1}
        >>> print(w2.dic)
        {(1, 2): -1, (0, 2): 1, (0, 1): -1}
        >>> print(w1 == w2.mul_esc(-1)) #Multiply by -1.
        True
        
        ..Note:: For this example is true that \partial_{p}(-\sigma) = - \partial_{p}(\sigma)
            like is wanted, and for all our cases the previous is true.
        
    """
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
                    s = s + P_chains([tuple(tuple_sorted(w))],[vec.dic[a]])
                else:
                    s = s - P_chains([tuple(tuple_sorted(w))],[vec.dic[a]])
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
def size_conjugacy_class(partition,n):
    """Returns the number of elements of a conjugacy class.

    Args:
        partition (list): Represents the partitions of a symmetric group (n).
        n (int): A integer to identify which the symmetric group.
    
    Returns:
        int: The number of elements of the conjugacy class.
        
    Examples:
        For find the number of elements of the conjugacy class of a
        symmetric group use ``size_conjugacy_class(partition,n)``.
        
        n = 4
        >>> print(size_conjugacy_class([4],n))
        6
        >>> print(size_conjugacy_class([1,1,1,1],n))
        1
        >>> print(size_conjugacy_class([2,1,1],n))
        6
        >>> print(size_conjugacy_class([2,2],n))
        3
        >>> print(size_conjugacy_class([3,1],n))
        8
        
        .. Note:: The examples showed are all the partition for the case
        4, and the sum of the results is 24 that is the cardinality of the 
        simmetric group of 4.
        
    """
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
    """Given a partition returns the a representate of a conjugacy class.

    Args:
        partition (list): Represents the partitions of a symmetric group (n).
    
    Returns:
        sympy.combinatorics.permutations.Permutation: A representate of the conjugacy class.
        
    Examples:
        For find a representate of a conjugacy class of s simmetric group
        use ``make_permutation(partition)``.
        
        >>> print(make_permutation([5]))
        (0 1 2 3 4)
        >>> print(make_permutation([1,1,1,1,1]))
        ()
        >>> print(make_permutation([2,1,1,1]))
        (4)(0 1)
        >>> print(make_permutation([2,2,1]))
        (4)(0 1)(2 3)
        >>> print(make_permutation([3,1,1]))
        (4)(0 1 2)
        >>> print(make_permutation([3,2]))
        (0 1 2)(3 4)
        >>> print(make_permutation([4,1]))
        (4)(0 1 2 3)
        
    """
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

def Reduce(N):
    """Returns a row reduced form of a matrix and a matrix that save the operations.

    Args:
        N (Matrix): The matrix which will be operated.
    
    Returns:
        tuple: The first element is the matrix which to be multiplied
        by the right to the original matrix, return the row reduced form
        and the other object is the row reduced form of the origin matrix.
        
    Examples:
        To use this functio use ``Reduce(Matrix)``. We will use the help of the 
        function ``rref`` to verify that the result is right.
        
        >>> M=Matrix([[-1, -1, -1, -1, 0, 0, 0, 0], 
        >>>   [ 1, 0, 0, 0, -1, -1, 0, 0], 
        >>>   [ 0, 1, 0, 0, 1, 0, -1, -1],
        >>>   [ 0, 0, 1, 0, 0, 1, 1, 0],
        >>>   [ 0, 0, 0, 1, 0, 0, 0, 1]])
        >>> print(M.rref())
        (Matrix([
        [1, 0, 0, 0, -1, -1,  0,  0],
        [0, 1, 0, 0,  1,  0, -1, -1],
        [0, 0, 1, 0,  0,  1,  1,  0],
        [0, 0, 0, 1,  0,  0,  0,  1],
        [0, 0, 0, 0,  0,  0,  0,  0]]), (0, 1, 2, 3))
        >>> P = Reduce(M)
        >>> print(P)
        (Matrix([
        [ 0,  1,  0,  0, 0],
        [ 0,  0,  1,  0, 0],
        [ 0,  0,  0,  1, 0],
        [-1, -1, -1, -1, 0],
        [ 1,  1,  1,  1, 1]]), Matrix([
        [1, 0, 0, 0, -1, -1,  0,  0],
        [0, 1, 0, 0,  1,  0, -1, -1],
        [0, 0, 1, 0,  0,  1,  1,  0],
        [0, 0, 0, 1,  0,  0,  0,  1],
        [0, 0, 0, 0,  0,  0,  0,  0]]))
    
        ..Note:: The first matrix is the row reduced form, and the second
            is a matrix which if is multiplied the left size to the 
            origin matrix, then we obtain the row reduced form, like below.
        print(P[0]*M == (M.rref())[0])
        True
        
    """
    M = N.copy()
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
    if elems[i]*g == elems[j]:
        return 1
    else:
        return 0


def regular_representation(G):
    """Builds the regular representation.
    
    Args:
        G (sympy.combinatorics.perm_groups.PermutationGroup): A symmetric group. 
        
    Returns:
        __main__.MatrixRepresentation: The matrix regular representation.
        
    """
    elems = list(G.elements)
    n = len(elems)
    mydict = {}
    for g in elems:
        mydict[g] = sp.ImmutableMatrix(sp.Matrix(n, n,
                                                 lambda i, j:
                                                 _char_f(G, g, i, j)))
    return MatrixRepresentation(mydict, G, n)
def MTS(A):
    """Create a non singular upper triangular matrix V such that V*AV=I. 
    
    Args:
        A (Matrix): A positive definite Hermitian matrix.
        
    Returns:
        V (Matrix): A non singular upper triangular matrix V that
            V*AV=I.
        
    Examples:
        To create this matrix use ``MTS(A)``.
       
        >>> M = Matrix([[1, 0, 1], [2, -1, 3], [4, 3, 2]])
        >>> N = Dagger(M)
        >>> A = N*M
        ..Note:: A is positive definite Hermitian matrix.
        >>> V = MTS(N*M) 
        >>> print(V)
        Matrix([[sqrt(21)/21, -sqrt(2310)/231, -12*sqrt(110)/11], [0, sqrt(2310)/110, 87*sqrt(110)/110], [0, 0, sqrt(110)]])
        >>> print(Dagger(V)*(A)*V)
        Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
    """
    A1=A
    n=A.shape[0]
    V=eye(n)
    for i in range(0,n):
        C=eye(n)
        C[i,i]=1/sqrt(A1[i,i])
        for j in range(i+1,n):
            C[i,j] = -(1/A1[i,i])*A1[i,j]
        V=V*C
        V.simplify()
        A1=Dagger(C)*A1*C
        A1.simplify()
    return V

def unitary_representation(G,d):
    """Becomes a matrix representation in a unitary matrix representation.
        
    Args:
        d (dict): A dict that must contains the map of a group into the group
            of nonsingular linear transformations of some finite dimensional
            vector space.
        G (sympy.combinatorics.perm_groups.PermutationGroup): The group. 
        
    Returns:
        __main__.MatrixRepresentation: A unitary matrix representation.

    """
    n=d.degree
    A=zeros(n,n)
    for g in d.map:
        J=(d.map[g].H)*d.map[g]
        J=expand(J)
        A=J+A
    A1=A
    V=eye(n)
    for i in range(0,n):
        C=eye(n)
        C[i,i]=1/sqrt(A1[i,i])
        for j in range(i+1,n):
            C[i,j] = -(1/A1[i,i])*A1[i,j]
        V=V*C
        V=expand(V)
        A1=(C.H)*A1*C
        A1=expand(A1)
    V=MTS(A)
    M = {}
    for g in list(G.elements):
        M[g]=sp.ImmutableMatrix((V.inv())*d.map[g]*V)
    return MatrixRepresentation(M, G, n)

def is_irreducible(G,d):
    """Determines if a representation is irreducible.
    
    Args:
        d (dict): A dict that must contains the map of a group into the group
            of nonsingular linear transformations of some finite dimensional
            vector space.
            
        G (sympy.combinatorics.perm_groups.PermutationGroup): The group. 
    
    Returns:
        True if the representation is irreducible, a matrix non 
        escalar that reduce the matrix representation in otherwise.

    Examples:
        To see if a representation is irreducible use
        ``is_irreducible(G,d)``.
        
        >>> G=SymmetricGroup(3)
        >>> rr=regular_representation(G)
        >>> M=is_irreducible(G,rr)
        >>> print(M)
        Matrix([[0, 1/3, 0, 0, 0, 0], [1/3, 0, 0, 0, 0, 0], [0, 0, 0, 1/3, 0, 0], [0, 0, 1/3, 0, 0, 0], [0, 0, 0, 0, 0, 1/3], [0, 0, 0, 0, 1/3, 0]])
            
    """
    n=d.degree
    N=eye(n)
    for r in range(0,n):
        for s in range(0,n):
            H=zeros(n)        
            if (n-1-r==n-1-s):
                H[n-1-r,n-1-r]=1
            else:
                if (n-1-r>n-1-s):
                    H[n-1-r,n-1-s]=1
                    H[n-1-s,n-1-r]=1
                else:
                    H[n-1-r,n-1-s]=1*I
                    H[n-1-s,n-1-r]=-1*I
            M=zeros(n,n)
            R=unitary_representation(G,d)
            for g in R.map:
                M=M+(R.map[g].H*H*R.map[g])
            M=(sympify(1)/n)*M
            M=expand(M)
            if (M!=M[0,0]*N):
                return M
    else:
        return True


def block(M):
    """A function that return where end the blocks of a matrix.
   
    Args:
        M (Matrix): The matrix which will be find their blocks.
            
    Returns:
        v (list): A list that indicated where end the blocks 
            of a matrix.
            
    Examples:
        To find the blocks of a matrix use ``block(M)``.
        
        >>> M=Matrix([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        >>>  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0], 
        >>>  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        >>>  [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        >>>  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
        >>>  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        >>>  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        >>>  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        >>>  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        >>>  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        >>> print(block(M))
        [0, 4, 5, 6, 7, 8, 9]        
    """   
    v=[]
    c1=0
    i=0
    n=M.shape[0]
    while (c1<n):
        c=0
        for j in range(c1,n):
            if (M[i,j]!=0 or M[j,i]!=0):
                if (Abs(i-j)>c):
                    c=Abs(i-j)
        if (c==0):
            v.append(c1)
            c1=c1+1
            i=c1
        else:
            bloques=False
            while (bloques==False):
                bloques=True
                for j in range(c1,c1+c+1):
                    for k in range(c1+c+1,n):
                         if (M[j,k]!=0 or M[k,j]!=0):
                            if (Abs(i-k)>c):
                                c=Abs(i-k)
            v.append(c1+c)
            c1=c1+c+1
            i=c1
    return v

def blockI(M,n,i):
    """A function that given a matrix, put it since the entry (i,i) of a 
    identity matrix of degree n.
   
    Args:
        M (Matrix): The matrix which will be put in the entry (i,i) of
            a identity matrix of degree n.
        
        n (int): Determine the size of the identity matrix.
        
        i (int): A integer that will indicated the entry (i,i) of the
            identity matrix.
    Returns:
        N (Matrix): The identity matrix with contains the matrix M
            in the entry (i,i).
            
    Raises:
        IndexError: If the number of the columns or the raws plus i are bigger
            to n.
            
    Examples:
        To use this function use ``blockI(M, n, i)``.
        
        >>> M=Matrix([[1, 1, 1], 
        >>>           [1, 1, 1]])
        >>> print(blockI(M,4,0))
        Matrix([[1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        
    """
#    a=M.shape[0]
    N=eye(n)
    for j in range(0,M.shape[0]):
        for k in range(0,M.shape[1]):
            N[j+i,k+i]=M[j,k]
    return N

def reduce(G,d):
    """Descompose a representation into irreudibles
    
    Args:
        G (Group): The group.
        
        d (dict): The representation.
            
    Returns:
        U (Matrix): A matrix wich descompose the representation.
    """
    M=is_irreducible(G,d)
    b=d.degree
    if (M==True):
        return(eye(b))
    else:
        (P, J) = M.jordan_form()
        P=expand(P)
        w=[]
        for g in d.map:
            w.append(block(P.inv()*d.map[g]*P))
        l=len(w[0])
        au=w[0]
        for g in w:
            if (len(g)<l):
                l=len(g)
                au=g
        e=0
        U=P
        for a in au:
            d1={}
            for g in list(G.elements):
                d1[g]=sp.ImmutableMatrix((P.inv()*d.map[g]*P)[e:a+1,e:a+1])
            U=U*blockI(reduce(G,MatrixRepresentation(d1,G,(a+1-e))),b,e)
            e=a+1
        return U
    
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
