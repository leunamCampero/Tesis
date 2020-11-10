# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:54:13 2020

@author: Hp
"""


from sympy import *
init_printing(use_unicode=True)
#import sympy 
from sympy import sympify
from sympy import solve
from sympy.abc import x
import networkx as nx
import numpy.matlib 
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
import timeit
from sympy.matrices import Matrix
from sympy.polys.domains import ZZ
from sympy.matrices.normalforms import smith_normal_form
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import null_space
from scipy.linalg import orth
from scipy.linalg import lu
from scipy import linalg
from numpy.linalg import matrix_rank
import numpy
from numpy import asarray_chkfinite, zeros, r_, diag
import numpy as np
from scipy.sparse import csr_matrix
from numpy import genfromtxt


__all__ = ['svd', 'svdvals', 'diagsvd', 'orth']


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

class P_chains:
    """
    A class used to do and operate p-chains.
    
    ----------
    Attributes:    
    ----------
    keys: list
        A list with the elements of the p-chains.
    values: list
        A list with the coefficient for every element in the p-chains.
    """
    
    def __init__(self, keys, values):
        '''
        Makes a p-chain.
        
        ----------
        Parameters
        ----------
        keys: list
            A list with the elements of the p-chains.
        values: list
            A list with the coefficient for every element in the p-chains.
        
        ------
        Raises:
        ------
        IndexError: 
            If the number of p-simplexs is not equal to the number of coefficients. 
        TypeError: 
            If the p-simplex given is not immutable data types like a tuple.
        
        ---------
        Examples:
        ---------
        To make a p-chain, use the ``P_chains(list1,list2)' ' class.  A p-chain is 
        constructed by providing a list_1 of p-simplex and a list_2 with 
        their coefficients, i.e.
            >>> P = P_chains([(0,1,2,3)],[1])
            >>> Q = P_chains([(0,1,2),(0,1,3)],[-1,2])
            >>> print(P.dic)
            {(0, 1, 2, 3): 1}
            >>> print(Q.dic)
            {(0, 1, 2): -1, (0, 1, 3): 2}
            
        One important thing to note about P_chains is that the p-simplex must be 
        a immutable data types like a tuple, and to see the P-chains, you need 
        use ``.dic''.
        
        '''
        self.keys = keys
        self.values = values
        self.dic = {}
        c = 0
        for x in self.keys:
            self.dic[x] = self.values[c]
            c = c+1
            
    def __add__(self, other):
        """
        Sums two p-chains.
        
        ----------
        Parameters
        ----------
        other: ( __main__.P_chains)
            Other p-chain.
            
        ------
        Return
        ------
        __main__.P_chains:
            A new p-chains that is the sum of the two p-chains given.
            
        ---------
        Examples:
        ---------
        To sum two p-chains, use ``+''.
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
        p_simplexes = []
        values = []
        for h in list(D.keys()):
            p_simplexes.append(h)
            values.append(D[h])
        return P_chains(p_simplexes,values)    
    
    def __sub__(self, other):
        """
        Subtracts two p-chains.
        
        ----------
        Parameters
        ----------
        other: ( __main__.P_chains)
            Other p-chain.
            
        ------
        Return
        ------
        __main__.P_chains:
            A new p-chains that is the sum of the two p-chains given.
            
        ---------
        Examples:
        ---------
        To subtract two p-chains, use ``-''.
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
        p_simplexes = []
        values = []
        for h in list(D.keys()):
            p_simplexes.append(h)
            values.append(D[h])
        return P_chains(p_simplexes,values)     
    
    def __eq__(self, other):
        '''
        Returns if the two P_chains are equal
        
        ----------
        Parameters
        ----------
        other: ( __main__.P_chains)
            Other p-chain.
            
        ------
        Return
        ------
        bool:
            The return value. True for success, False otherwise.
            
        ---------
        Examples:
        ---------
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
        '''
        Returns if the two P_chains are not equal
        
        ----------
        Parameters
        ----------
        other: ( __main__.P_chains)
            Other p-chain.
            
        ------
        Return
        ------
        bool:
            The return value. True for success, False otherwise.
            
        ---------
        Examples:
        ---------
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
        '''
        Returns if the two P_chains are not equal
        
        ----------
        Parameters
        ----------
        other: ( __main__.P_chains)
            Other p-chain.
            
        ------
        Return
        ------
        bool:
            The return value. True for success, False otherwise.
            
        ---------
        Examples:
        ---------
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
    
class SCFromFacets:
    """
    A class used to do and operate p-chains.
    
    ----------
    Attributes    
    ----------
    v: list
        A list of the maximum faces of the simplicial complex.
    """
    def __init__(self, v):
        '''
        Generate the nodes of a simplicial complex.
        
        ----------
        Attributes
        ----------
        v: list
            A list of the maximum faces of the simplicial complex.
            
        -------    
        Raises
        -------
        AttributeError:
            If the maximal faces are not immutable objects.
            
        ---------
        Examples:
        ---------
        To make a simplicial complex asociated with a graph, 
        use the ``SimplicialComplex`` class. We need a graph G.
            >>> sc = SCFromFacets([[0,1,2,3],[3,4,5]])
            >>> print(sc.SCnodes)
            ￼￼[0, 1, 2, 3, 4, 5]
        
        '''
        self.v = v
        self.SCFaset = []
        self.SCnodes = set()
        for x in self.v:
            self.SCFaset.append(x)
            for y in x:
                self.SCnodes.add(y)
        self.SCnodes = list(self.SCnodes)
        
    def faces(self):
        """
        Makes the faces of a simplicial complex. 
        
           A simplicial complex SC is uniquely determined by its maximum faces,
           and it must contains every face of a simplex, also the intersection 
           of any two simplexes of SC is a face of each of them.
           
        ----------
        Returns
        ----------
        list:
            A list of the faceset of a simplicial complex.
            
        ---------
        Examples:
        ---------
        To create the faces of a simplical complex use, 
        ``SimplicialComplex.faces()''.
            >>> sc = SCFromFacets([[0,1,2,3],[3,4,5]])
            >>> print(sc.faces())
            [(1, 3), (0, 1, 2), (0, 1, 3), (0, 3), (1, 2), (1,), (0, 2, 3), (3,), (5,), (0, 1, 2, 3), (4, 5), (1, 2, 3), (2, 3), (3, 5), (0, 1), (0,), (2,), (4,), (), (3, 4, 5), (3, 4), (0, 2)]
            
        .. Note:: The faces are sorted by their dimension.
            
            """
        
        faceset = set()
        for faset in self.SCFaset:
            for face in sub_lists(faset):
                faceset.add(tuple(face))
        return list(faceset)

    def p_simplex(self, p):
        """
        Creates a list of the faces of a simplex with dimension p.
        
        ----------
        Parameters
        ----------
        p: int
            The dimension of the desired faces.
          
        --------    
        Returns:
        --------
        list: 
            A list of the faces of a simplex with dimension p.
        
        ---------
        Examples:
        ---------
        The p-simplices are done with 
        "SimplicialComplex.p_simplex(p)".
            >>> sc = SCFromFacets([[0,1,2]])
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
        ``sc.p_simplex(5)''.
            
        """
        return list(filter(lambda face: (len(face) == p+1) , self.faces()))
    
    def dimension(self):
        """
        Gives the dimension of a simplicial complex.

        ----------
        Parameters
        ----------
        p: int
            The dimension of the faces.
          
        --------    
        Returns:
        --------
        (a - 1): int 
            The dimension of the simplicial complex.
        -------    
        Raises:
        -------
            Return ``-1'' if the simplicial compolex is empty.
        
        Examples:
        To use the method dimension write 
        ``SimplicialComplex.dimension()''.
            >>> sc = SCFromFacets([[0,1,2,4]])
            >>> print(sc.dimension())
            3
        """
        a = 0
        for x in self.faces():
            if (len(x) > a):
                a = len(x) 
        return a-1
    def basis_group_oriented_p_chains(self, p):
        """
        Gives a basis for the group of oriented p-chains.
        
        ----------
        Parameters
        ----------
        p: int
            Indicated the dimension of the p-simplex.
          
        --------    
        Returns:
        --------
        __main__.P_chains: 
            A new p-chains that contains the basis of the group.
        
        .. Note:: To every element is given the coefficiente ``1'' 
        by default, this is because the p-simplex are sorted with the
        lexicographical order, i.e, this orientation is taken positive.
        
        -------
        Raises:
        -------
        AttributeError: 
            If p is lower than zero or bigger than the dimension of the
            simplicial complex dimension.
            
        ---------
        Examples:
        ---------
        To create a basis for the group of oriented p-chains, use 
        ``SimplicialComplex.basis_group_oriented_p_chains(p)``.
            >>> sc = SCFromFacets([[0,1,2,4]])
            >>> print(sc.basis_group_oriented_p_chains(0).dic)
            {(0,): 1, (1,): 1, (2,): 1, (4,): 1}
        """
        if ((p<0) or (p>self.dimension())):
            return 0
        else:
            c_p = P_chains([], [])
            faces = []
            values = []
            for x in self.p_simplex(p):
                if (is_int(x) == True):
                    faces.append(tuple(tuple_sorted(x)))
                    values.append(1)
                else:
                    faces.append(tuple(x))
                    values.append(1)
            c_p = P_chains(faces, values)
            return c_p
    def matrix_simmetric_representate(self, p):
        """
        Give the matrix associated to the boundary operator.

        ----------
        Parameters
        ----------
        p: int
            Determine which basis of the p-simplex will be used.
          
        --------    
        Returns:
        --------
        A matrix if p is bigger than -1, and lower than the dimension
        of the simplicial complex, return False otherwise.
           
        ---------
        Examples:
        ---------
        To compute the matrix associated to the boundary operator, use
        ``SimplicialComplex.matrix_simmetric_representate(p)''.
            >>> sc = SCFromFacets([[0,1,2,4]])
            >>> print(sc.matrix_simmetric_representate(3))
            [[-1]
             [ 1]
             [ 1]
             [-1]]
            .. Note:: The previous is right becuase there is only 1 
            3-simplex and 4 2-simplex.
        """
        if (p >0 and (p <= self.dimension()) ):
            v = self.basis_group_oriented_p_chains(p)
            p = p - 1
            ve = self.basis_group_oriented_p_chains(p)
            M = csr_matrix((len(ve.dic), len(v.dic)), dtype=np.int8).toarray()
            j = 0
            for u1 in list(v.dic.keys()):
                d =  P_chains([u1],[v.dic[u1]])
                l = boundary_op_n(d).dic
                for u2 in list(l.keys()):
                    i = 0
                    for w in list(ve.dic.keys()):
                        if (w == u2):
                                M[i,j] = int((l)[u2])
                        i = i + 1
                j = j + 1
            return M 
        else:
            if (p == 0):
                return np.identity(len(list(self.basis_group_oriented_p_chains(0).dic.keys())))
            else:
                return False

    def decomposition_into_s_n_irreducibles(self, n):
        """
        Returns a dictionary showing the descomposition into irreducibles
        of the ith reduced homology S_n module of a simplicial complex.
        
        -----
        Args:
        -----
        n: int 
            It indicates which is the symmetrical group that acts in the homology.
        
        --------
        Returns:
        --------
        dict: 
            A dictionary that shows the decomposition into irreducible 
            submodules of the ith homology, where the number that 
            appears in front of each partition of n is the multiplicity
            of the irreducible representation in the homology.
        
        Examples:
            To get that, use 
            ``SimplicialComplex.decomposition_into_s_n_irreducibles(n)''.
            
            >>> n=4
            >>> G1 = matching_graph(n)
            >>> sc_m = SCG(G1).decomposition_into_s_n_irreducibles(n)
            >>> print(sc_m)
            {0: {(4,): 1.0, (1, 1, 1, 1): 0.0, (2, 1, 1): 0.0, (2, 2): 1.0, (3, 1): 0.0}, 1: {(4,): 0.0, (1, 1, 1, 1): 0.0, (2, 1, 1): 0.0, (2, 2): 0.0, (3, 1): 0.0}}
        
        """
        w5 = partitions_list(n)
        M5 = form_matrix_yt(w5)
        card = math.factorial(n)
        vec_dic = {}
        for k in range(self.dimension()+1):
            D = {}
            uu = []
            vv = []
            p = k 
            A = self.matrix_simmetric_representate(p)
            if (p >0 and (p <= self.dimension())):
                null = nullspace(A)
                w3 = []
                for i in range(len(null[0])):
                    w = []
                    for j in range(len(null)):
                        w.append(null[j][i])
                    w3.append(w)   
                null = w3
                M = np.matrix(w3, dtype= np.float64).transpose()
                Mi = np.linalg.pinv(M)
            else:
                if (p == 0):
                    M = A
                    null = []
                    for i in range(A.shape[0]):
                        aux = []
                        for j in range(A.shape[1]):
                            aux.append(M[i,j])
                        null.append(aux)
                    M = np.matrix(null, dtype=np.float64)
                    Mi = M
            p = k + 1
            if (p>0 and (p <= self.dimension())):
                A1=self.matrix_simmetric_representate(p)
                col = columnspace(A1)
                w4 = []
                for i in range(len(col[0])):
                    w = []
                    for j in range(len(col)):
                        w.append(col[j][i])
                    w4.append(w)
                col = w4
                M1 = np.matrix(w4, dtype=np.float64).transpose()
                Mii = np.linalg.pinv(M1)
            for h in w5:
                p = k 
                if (p >0 and (p <= self.dimension())):
                    if (all(elem == 0 for elem in null[0])):
                        l1 = 0
                    else:
                        he = self.basis_group_oriented_p_chains(p)   
                        on1 = np.ones(len(list(he.dic.keys())), dtype=np.float64) 
                        v = P_chains([],[])
                        v = P_chains(list(he.dic.keys()),on1)
                        v1 = permutation_in_simplex_test(v, make_permutation(h))
                        D1={}
                        c1 = 0
                        for i in list(v1.dic.keys()):
                            c2 = 1
                            for j in list(he.dic.keys()):
                                if (i == j):
                                    if (v1.dic[i] == he.dic[j]):
                                        D1[c1] = c2
                                    else:
                                        D1[c1] = -c2
                                c2 = c2 + 1
                            c1 = c1 + 1
                        rr = M.shape[0]
                        cc = M.shape[1]
                        Ma  = np.zeros([rr,cc],dtype=np.float64)
                        for i in range(rr):
                            Ma[i,:] = (M[(abs(D1[i])-1),:]*(np.sign(D1[i])))
                        l1 = 0
                        for j in range(cc):
                            l1 = np.dot(Mi[j,:],Ma[:,j])[0,0] + l1
                else:
                    if (p == 0):
                        he = self.basis_group_oriented_p_chains(p)   
                        on1 = np.ones(len(list(he.dic.keys())), dtype=np.float64) 
                        v = P_chains([],[])
                        v = P_chains(list(he.dic.keys()),on1)
                        v1 = permutation_in_simplex_test(v, make_permutation(h))
                        D1={}
                        c1 = 0
                        for i in list(v1.dic.keys()):
                            c2 = 1
                            for j in list(he.dic.keys()):
                                if (i == j):
                                    if (v1.dic[i] == he.dic[j]):
                                        D1[c1] = c2
                                    else:
                                        D1[c1] = -c2
                                c2 = c2 + 1
                            c1 = c1 + 1
                        rr = M.shape[0]
                        cc = M.shape[1]
                        Ma  = np.zeros([rr,cc],dtype=np.float64)
                        for i in range(rr):
                            Ma[i,:] = (M[(abs(D1[i])-1),:]*(np.sign(D1[i])))
                        l1 = 0
                        for j in range(cc):
                            l1 = np.dot(Mi[j,:],Ma[:,j])[0,0] + l1
                    else:
                        l1 = 0
                p = k + 1
                if (p>0 and (p <= self.dimension())):
                    hi = self.basis_group_oriented_p_chains(p-1)   
                    on1i = np.ones(len(list(hi.dic.keys())), dtype=np.float64) 
                    vi = P_chains([],[])
                    vi = P_chains(list(hi.dic.keys()),on1i)
                    v1i = permutation_in_simplex_test(vi, make_permutation(h))
                    D1i={}
                    c1 = 0
                    for i in list(v1i.dic.keys()):
                        c2 = 1
                        for j in list(hi.dic.keys()):
                            if (i == j):
                                if (v1i.dic[i] == hi.dic[j]):
                                    D1i[c1] = c2
                                else:
                                    D1i[c1] = -c2
                            c2 = c2 + 1
                        c1 = c1 + 1
                    rr = M1.shape[0]
                    cc = M1.shape[1]
                    Mai  = np.zeros([rr,cc],dtype=np.float64)
                    for i in range(rr):
                        Mai[i,:] = (M1[(abs(D1i[i])-1),:]*(np.sign(D1i[i])))
                    l2 = 0
                    for j in range(cc):
                        l2 = np.dot(Mii[j,:],Mai[:,j])[0,0] + l2
                else:
                    l2 = 0
                uu.append(l1-l2) 
                vv.append(size_conjugacy_class(h,n))
            for i in range(M5.shape[0]):
                Ip = 0
                for j in range(M5.shape[1]):
                    Ip = Ip + M5[i,j]*uu[j]*vv[j]
                Ip = Ip/card
                D[tuple(w5[i])]=abs(round(Ip))
            vec_dic[k] = D
        return vec_dic 
    def decomposition_into_s_n_irreducibles_chain_sp(self, n):
        """
        Returns a dictionary showing the descomposition into irreducibles
        of the ith chain space like a S_n module of a simplicial complex.
        
        -----
        Args:
        -----
        n: int 
            It indicates which is the symmetrical group that acts in the chain space.
        
        --------
        Returns:
        --------
        dict: 
            A dictionary that shows the decomposition into irreducible 
            submodules of the ith chain space, where the number that 
            appears in front of each partition of n is the multiplicity
            of the irreducible representation in the space.
        
        Examples:
            To get that, use 
            ``SimplicialComplex.decomposition_into_s_n_irreducibles(n)''.
            
            >>> n=4
            >>> G1 = matching_graph(n)
            >>> G = clique_graph(G1)
            >>> a =SCG(G1).decomposition_into_s_n_irreducibles_chain_sp(n)
            >>> b= SCG(G).decomposition_into_s_n_irreducibles_chain_sp(n)
            >>> print(a)
            >>> print(b)
            {0: {(4,): 1.0, (1, 1, 1, 1): 0.0, (2, 1, 1): 0.0, (2, 2): 1.0, (3, 1): 1.0}, 1: {(4,): 0.0, (1, 1, 1, 1): 0.0, (2, 1, 1): 0.0, (2, 2): 0.0, (3, 1): 1.0}}
            {0: {(4,): 1.0, (1, 1, 1, 1): 0.0, (2, 1, 1): 0.0, (2, 2): 1.0, (3, 1): 0.0}}
        """
        w5 = partitions_list(n)
        M5 = form_matrix_yt(w5)
        card = math.factorial(n)
        vec_dic = {}
        for k in range(self.dimension()+1):
            D = {}
            uu = []
            vv = []
            he = self.basis_group_oriented_p_chains(k) 
            for h in w5:
                v1 = P_chains([],[])
                v1 = permutation_in_simplex_test(he, make_permutation(h))
                rr = len(list(he.dic.keys()))
                Ma  = np.zeros([rr,rr],dtype=np.float64)
                c1 = 0
                for i in list(he.dic.keys()):
                    c2 = 0
                    for j in list(v1.dic.keys()):
                        if (i == j):
                            Ma[c1,c2] = v1.dic[i]
                        c2 = c2 + 1
                    c1 = c1 + 1
                Ma = np.matrix(Ma, dtype='float64')
                uu.append(np.trace(Ma)) 
                vv.append(size_conjugacy_class(h,n))
            for i in range(M5.shape[0]):
                Ip = 0
                for j in range(M5.shape[1]):
                    Ip = Ip + M5[i,j]*uu[j]*vv[j]
                Ip = Ip/card
                D[tuple(w5[i])]=Ip
            vec_dic[k] = D
        return vec_dic
    def dimension_homology_sc(self):
        """
        Returns a dictionary showing the dimension of the simplicial complex who
        are not trivially zeros

        --------
        Returns:
        --------
        dict: 
            A dictionary that contains the ith homolgy and its corresponding dimension.
        
        ---------
        Examples:
        ---------
        To get the ith homology dimension of the simplicial complex,
        use ``SimplicialComplex.dimension_homology_sc(n)``.
            >>> sc = SCFromFacets([[0,1,2,3]])
            >>> print(sc.dimension_homology_sc())
            {0: 1, 1: 0, 2: 0, 3: 0}
        
        """
        vec_dic = {}
        for k in range(self.dimension()+1):
            p = k 
            A = self.matrix_simmetric_representate(p)
            dn = 0
            dc = 0
            if (p == 0):
                dn = A.shape[1]
            if (p > 0 and (p <= self.dimension())):
                null = null_space(A)
                if (null.size != 0):
                    dn = len(null[0])
                if (all(elem == 0 for elem in null[0])):
                    dn = 0 
            p = k + 1
            if (p>0 and (p <= self.dimension())):
                A1=self.matrix_simmetric_representate(p)
                col = orth(A1)
                if (col.size != 0):
                    dc = len(col[0])
                else: 
                    dc = 0
            vec_dic[k] = dn - dc
        return vec_dic 

class YoungTableaux:
    """
    A class to compute irreducible character values of a symmetric group.

        ----------
        Parameters
        ----------
        p_lambda: list
            A list that represent the first partition.
        p_rho: list
            A list that represent the second partition.
    """
    
    def __init__(self, p_lambda, p_rho):
        '''
        ----------
        Parameters
        ----------
        p_lambda: list
            A list that represent the first partition.
        p_rho: list
            A list that represent the second partition.
        
        '''
        self.p_lambda = p_lambda
        self.p_rho = p_rho
        
    def choose_tableaux(self, v):
        """
        A method to identify if a given list is a border-strip tableaux or not.
        
        ----------
        Parameters
        ----------
        v: list
            A list of list that is the candidate to be a border-strip tableaux.
        
        --------
        Returns:
        --------
        bool:
            True if the list is a border-strip tableaux, False otherwise.
        
        -------
        Raises:
        -------
            IndexError: If the list given is not consistent with the partitions given.
        
        ---------
        Examples:
        ---------
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
        ￼
        """
        for i in v:
            for j in range(0,len(i)-1):
                if (i[j]>i[j+1]):
                    return False
        for i in range(1,len(v)):
            for j in range(0,len(v[i])):
                if (v[i][j]<v[i-1][j]):
                    return False
        for u in range(len(self.p_rho)):
            ir = []
            ic = []
            for i in range(0,len(v)):
                for j in range(0,len(v[i])):
                    c=0
                    c1 = 0
                    if (j != 0):
                        if (v[i][j] == v[i][j-1]):
                            c=1
                    if (j != (len(v[i])-1)):
                        if (v[i][j] == v[i][j+1]):
                            c=1
                            c1 = c1 + 1
                    if (i != 0):
                        if (v[i][j] == v[i-1][j]):
                            c=1
                    if (i != (len(v)-1)):
                        if (j < len(v[i+1])):
                            if ((v[i][j] == v[i+1][j-1]) and (v[i][j] != v[i][j-1]) and (v[i][j] != v[i+1][j])):
                                return False
                            if (v[i][j] == v[i+1][j]):
                                c=1
                                c1 = c1 + 1
                        if (j < (len(v[i+1]) - 1)):
                            if (v[i][j] == v[i+1][j+1]):
                                c1 = c1 + 1
                        if (j == len(v[i+1])):
                            if ((v[i][j] == v[i+1][j-1]) and (v[i][j] != v[i][j-1])):
                                return False
                    if ((c == 0) and (self.p_rho[v[i][j]-1]>1)):
                        return False
                    if (c1 == 3):
                        return False
                    if (v[i][j] == (u+1)):
                        ir.append(i)
                        ic.append(j)
            ir = np.unique(ir)
            ic = np.unique(ic)
            ir = sorted(ir)
            ic = sorted(ic)
            for i in range(len(ir)-1):
                if ((ir[i+1]-ir[i]) > 1):
                    return False
            for j in range(len(ic)-1):
                if ((ic[j+1]-ic[j]) > 1):
                    return False
        else:
            return True
    def MNR(self):
        """
        A method to generate all the border-strip tableauxs according to the partitions given.
        
        ----------
        Returns
        ----------
        D1: list
            A list that contains the border-strip tableauxs.
            
        ---------   
        Examples:
        ---------
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
    ￼    And for the third example there are not any border-strip
        tableaux.
        """
        p=[]
        i=1
        for h in self.p_rho:
            for j in range(0,h):
                p.append(i)
            i=i+1
        perm = perm_unique(p)
        D=[]
        for i in list(perm):
            v=[]   
            for g in i:
                v.append(g)
            c=0
            w=[]
            for r in self.p_lambda:
                u=[]
                for j in range(c,c+r):
                    u.append(v[j])
                w.append(u)
                c=c+r
            if (self.choose_tableaux(w) == True):
                D.append(w)
        return D
    
    def heights(self):
        """
        A method to calculate the heights i.e the sum of the heights of the border strips.
        
        --------
        Returns:
        --------
        He: list
            A list that contains heights.
         
        ---------
        Examples:
        ---------
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
        """
        A method to calculate irreducible character values ​​through the Murnaghan-Nakayama rule.
        
        --------
        Returns:
        --------
        s: int
            A irreducible character value of a symmetric group according to the given partition.
        
        ---------
        Examples:
        ---------
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

def sub_lists(my_list):
    """
    A function that find all the sublists of a list given.
    """
    subs = []
    for i in range(0, len(my_list)+1):
        temp = [list(x) for x in combinations(my_list, i)]
        if len(temp)>0:
           subs.extend(temp)
    return subs
def boundary_op_n(v):
    """
    Returns the action of the boundary operator on p-chains.
    
    ----------
    Parameters
    ----------
    v:  __main__.P_chains
        A p-chain chain.
        
    --------
    Returns:
    --------
    v:  __main__.P_chains
        The p-chain under the boundary operator.
         
    ---------
    Examples:
    ---------
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
    h = list(v.dic.keys())[0]
    p = len(h) - 1
    s = P_chains([],[])
    if (p != 0) and (isinstance(h, str) != True) and (isinstance(h, frozenset) != True) and (isinstance(h, ImmutableMatrix) != True):
        if (is_int(list(v.dic.keys())) == True):
            for u in v.dic.keys():
                c = 0
                for i in u:  
                    w = list(u)[:]
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
            aux = P_chains([],[])
            D = {}
            ct = 0
            st = []
            for u in v.dic.keys():
                for x in u:
                    if x not in st:
                        st.append(x)
            for i in st:
                D[tuple([ct])] = i
                ct = ct + 1
            for u in v.dic.keys():
                w2 = []
                for x in u:
                    for y in list(D.keys()):
                        if (x == D[y]):
                            w2.append(y)
                aux = aux + P_chains([tuple(w2)],[v.dic[u]])     
            v = aux
            for u in v.dic.keys():
                c = 0
                for i in u:  
                    w = list(u)[:]
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
            s2 = P_chains([],[])
            for u in s.dic.keys():
                w2=[]
                for i in u:
                    w2.append(D[i])
                s2 = s2 + P_chains([tuple(w2)],[s.dic[u]])
                    
            return s2
    else:
        return s

def partitions_list(n):
    """
    Returns a list of the partitions of n.

    ----------
    Parameters
    ----------
    n: int
        A integer that determine the partitions.
    
    --------
    Returns:
    --------
    w: list
        A list of list that are the different partitions of n.
    
    -------
    Raises:
    -------
    ValueError: If ``n`` is not bigger than zero.
    
    ---------
    Examples:
    ---------
    To form all the partitions for the integer ``n'', use 
    ``list_partitions''.
        
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
    """
    Returns the a matrix that represent the character table of the symmetric group.
    
    ----------
    Parameters
    ----------
    w: list
        A list with the partitions for certain symmetric group.
    
    --------
    Returns:
    --------
    M: <class 'sympy.matrices.dense.MutableDenseMatrix'>
        A matrix with the characters of the character
        table of the symmetric group.
    
    ---------
    Examples:
    ---------
    To form the matrix that represent the character table of the
    symmetric group, use ``form_matrix_yt''.
        >>> v = partitions_list(3)
        >>> print(form_matrix_yt(v))
        Matrix([[1, 1, 1], [1, 1, -1], [-1, 2, 0]])
        
    .. Note:: The function need a list of the partitions for
    ``n``, then is used the function ``partitions_list''.
    """
    M = np.zeros((len(w),len(w)))
    for i in range(len(w)):
        for j in range(len(w)):
            M[i,j] = YoungTableaux(w[i],w[j]).CMNR()
    return M

def eq_elements(a, b):
    """
    A function that identify when tuples are equal except by orientation.
    
    ----------
    Parameters
    ----------
    a: tuple
        The first tuple.
    b: tuple
        The second tuple.
    
    --------
    Returns:
    --------
    bool: 
        True if the tuples are equal except by orientation, 
        False otherwise.
    
    -------
    Raises:
    -------
    TypeError: 
        If the tuples don't have the same structure, for 
        example: 
            a = ((0,1),(2,3),(5,6)) 
            b = ((1,0))
    ---------
    Examples:
    --------
    To see if two tuples are equal use ``eq_elements''.
        >>> a1 = ((0,1),(2,3),(5,6)) 
        >>> b1 = ((0,3),(2,1),(5,6)) 
        >>> a2 = ((0,1),(2,3),(5,6)) 
        >>> b2 = ((6,5),(1,0),(3,2)) 
        >>> print(eq_elements(a1,b1))
        False
        >>> print(eq_elements(a2,b2))
        True

    """
    if ((isinstance(a, int) == True) or (isinstance(a, str) == True)):
        return a == b
    if ((isinstance(a[0], int) == True) or (isinstance(a[0], str) == True)):
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
    """
    Determines the orientation of ``b'' taken the orientation of ``a'' positive.
    
    ----------
    Parameters
    ----------
    a: tuple
        The first tuple.
    b: tuple
        The second tuple.
    p: tuple
        The dimension of the simplex.
    
    --------
    Returns:
    --------
    ValueError:
        If the tuples are not equal under the
        function ``eq_elements''.
    
    ---------
    Examples:
    ---------
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
    """
    Sorted tuples of tuples.
    
    ----------
    Parameters
    ----------
    a: tuple
        A tuple the which will be sorted.
     
    --------
    Returns:
    --------
    tuple: 
        The tuple sorted. 
        
    ---------
    Examples:
    ---------
    The function ``sorted'' don't sort tuples of tuples, but
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
    if ((isinstance(a, int) == True) or (isinstance(a, str) == True)):
        return a
    if ((isinstance(a[0], int) == True) or (isinstance(a[0], str) == True)):
        return sorted(a)
    else:
        w = []
        for b in a:
            w.append(tuple(tuple_sorted(b)))
        return tuple(sorted(tuple(w)))
def tuple_permutation(v,P):
    """
    Determines the orientation of ``b`` taken the orientation of ``a`` positive.

    ----------
    Parameters
    ----------
    a: tuple
        The tuple which will under the Permunation ``p''.
    p: <class 'sympy.combinatorics.permutations.Permutation'>
        The Permutation.
     
    --------
    Returns:
    --------
    tuple: 
        The tuple with their elements permutated under the permutation ``p``.
            
    ---------
    Examples:
    ---------
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
    w = list(v)[:]
    test = True
    for i in range(len(v)):
        if ((isinstance(v[i], int) == True) or (isinstance(v[i], str) == True)):
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
    
    ----------
    Parameters
    ----------
    n: int
        A integer that to do the complete graph.

    --------
    Returns:
    --------
    networkx.classes.graph.Graph: 
        The matching graph. 
    -------
    Raises:
    -------
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
def clique_graph(g, cmax=float('inf')):
    """
    Makes a clique graph since a matching graph.
    
    ----------
    Parameters
    ----------
    n: int
        A integer that to do the matching graph.

    --------
    Returns:
    --------
    networkx.classes.graph.Graph: 
        The clique graph. 
    
    -------
    Raises:
    -------
    NetworkXError: 
        If n is a negative number.

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
def nullspace(A):
    """
    Returns a  ``list'' of column vectors that span the nullspace of the matrix.
    
    ----------
    Parameters
    ----------
    A: Matrix
        The matrix which we will find the nullspace.
    p: <class 'sympy.combinatorics.permutations.Permutation'>
        The Permutation.
    
    --------
    Returns:
    --------
    u: list
        A list of list with the generators of the kernel.
          
    ---------
    Examples:
    ---------
    To find the nullspace of a matrix, use ``nullspace(A)''. 
        >>> M1 = np.array([[2, 4, 6, 6], [8, 20, 0, 1], [5, 0, 3, 2]])
        >>> M2 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0,-1,0],[-1,0,0]])
        >>> print(nullspace(M1))
        [[ 0.13226716]
         [-0.0881781 ]
         [-0.69072848]
         [ 0.70542483]]
        >>> print(nullspace(M2))
        [array([0, 0, 0])]
        
    .. Note:: Essentially the function only obtain the nullspace
    with the function ``A.nullspace()`` and returns the trivial kernel
    if ``A.nullspace()`` is a emtpy list.
    
    """
    u = null_space(A)
    if (u.size == 0):
        return [np.zeros((A.shape[1],), dtype = int)]
    else:
        return u

def columnspace(M):
    """
    Returns a ``list`` of column vectors that span the columnspace of the matrix.
    
    ----------
    Parameters
    ----------
    A: Matrix
        The matrix which we will find the columnspace.
    p: <class 'sympy.combinatorics.permutations.Permutation'>
        The Permutation.
    
    --------
    Returns:
    --------
    list:
        A list of list with the generators of the columnspace (image).
    
    ---------
    Examples:
    ---------
    To find the columnspace of a matrix, use ``columnspace(A)``.
        >>> M1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        >>> M2 = np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,-1],[0,-1,0],[-1,0,0]])
        >>> print(columnspace(M1))
        [array([0, 0, 0])]
        >>> print(columnspace(M2))
        [[-0.70710678  0.          0.        ]
         [ 0.         -0.70710678  0.        ]
         [ 0.          0.         -0.70710678]
         [ 0.          0.          0.70710678]
         [ 0.          0.70710678  0.        ]
         [ 0.70710678  0.          0.        ]]
    """
    v = orth(M)
    if (v.size == 0):
        return [np.zeros((M.shape[0],), dtype = int)]
    else:
        return v
def permutation_in_simplex_test(vec, P):
    """
    Returns a simplex under a permutation.
    
    ----------
    Parameters
    ----------
    vec:  __main__.P_chains
        A p-chain which the permutation will act.
    P: sympy.combinatorics.permutations.Permutation
        The permutation.
    
    --------
    Returns:
    --------
    __main__.P_chains: 
        A new p-chain that is the result of the
        permutation acting on the original p-chain ``vec``.
            
    ---------
    Examples:
    ---------
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
        >>> n=5
        >>> G1 = matching_graph(n)
        >>> G = clique_graph(G1)
        >>> sc = SCG(G)
        >>> sigma = sc.basis_group_oriented_p_chains(1)
        >>> print(sigma.dic)
        {(((0, 3), (1, 4)), ((1, 4), (2, 3))): 1, (((0, 3), (1, 2)), ((0, 4), (1, 2))): 1, (((0, 2), (3, 4)), ((1, 2), (3, 4))): 1, (((0, 1), (2, 4)), ((1, 3), (2, 4))): 1, (((0, 4), (1, 2)), ((0, 4), (1, 3))): 1, (((0, 4), (2, 3)), ((1, 4), (2, 3))): 1, (((0, 2), (1, 4)), ((0, 2), (3, 4))): 1, (((0, 3), (2, 4)), ((1, 3), (2, 4))): 1, (((0, 3), (1, 4)), ((0, 3), (2, 4))): 1, (((0, 4), (1, 2)), ((1, 2), (3, 4))): 1, (((0, 2), (1, 4)), ((1, 4), (2, 3))): 1, (((0, 3), (1, 2)), ((0, 3), (2, 4))): 1, (((0, 1), (2, 3)), ((0, 1), (3, 4))): 1, (((0, 2), (1, 3)), ((0, 2), (1, 4))): 1, (((0, 1), (2, 3)), ((0, 1), (2, 4))): 1, (((0, 1), (3, 4)), ((1, 2), (3, 4))): 1, (((0, 1), (3, 4)), ((0, 2), (3, 4))): 1, (((0, 4), (1, 2)), ((0, 4), (2, 3))): 1, (((0, 1), (2, 3)), ((0, 4), (2, 3))): 1, (((0, 4), (1, 3)), ((1, 3), (2, 4))): 1, (((0, 2), (1, 3)), ((0, 4), (1, 3))): 1, (((0, 2), (1, 4)), ((0, 3), (1, 4))): 1, (((0, 4), (1, 3)), ((0, 4), (2, 3))): 1, (((0, 1), (2, 4)), ((0, 1), (3, 4))): 1, (((0, 2), (1, 3)), ((1, 3), (2, 4))): 1, (((0, 1), (2, 3)), ((1, 4), (2, 3))): 1, (((0, 2), (1, 3)), ((0, 2), (3, 4))): 1, (((0, 3), (1, 2)), ((0, 3), (1, 4))): 1, (((0, 3), (1, 2)), ((1, 2), (3, 4))): 1, (((0, 1), (2, 4)), ((0, 3), (2, 4))): 1}
        >>> bo_sigma=boundary_op_n(sigma)
        >>> rho_bo_sigma=permutation_in_simplex_test(bo_sigma,Permutation(0,1))
        >>> print(rho_bo_sigma.dic)
        {(((0, 4), (2, 3)),): 4, (((0, 4), (1, 3)),): 0, (((0, 2), (1, 4)),): -2, (((0, 2), (1, 3)),): -4, (((0, 2), (3, 4)),): 4, (((1, 2), (3, 4)),): 2, (((0, 3), (2, 4)),): 4, (((0, 1), (2, 4)),): -2, (((0, 3), (1, 4)),): 0, (((1, 4), (2, 3)),): 2, (((0, 4), (1, 2)),): -2, (((1, 3), (2, 4)),): 2, (((0, 1), (3, 4)),): 0, (((0, 1), (2, 3)),): -4, (((0, 3), (1, 2)),): -4}
        >>> rho_sigma=permutation_in_simplex_test(sigma,Permutation(0,1))
        >>> bo_rho_sigma=boundary_op_n(rho_sigma)
        >>> print(bo_rho_sigma.dic)
        {(((0, 4), (2, 3)),): 4, (((0, 4), (1, 3)),): 0, (((0, 2), (1, 4)),): -2, (((0, 2), (1, 3)),): -4, (((1, 2), (3, 4)),): 2, (((0, 2), (3, 4)),): 4, (((0, 3), (2, 4)),): 4, (((0, 1), (2, 4)),): -2, (((0, 3), (1, 4)),): 0, (((1, 4), (2, 3)),): 2, (((0, 4), (1, 2)),): -2, (((1, 3), (2, 4)),): 2, (((0, 1), (3, 4)),): 0, (((0, 1), (2, 3)),): -4, (((0, 3), (1, 2)),): -4}
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
        p = len(v[0]) - 1
        faces = []
        values = []
        for a in v:
            if (isinstance(a, int) == True): 
                return vec
            else:
                w = tuple_permutation(a,P)
                w1 = tuple_sorted(w)
                if (orientation_function(w1,w,p) == True):
                    faces.append(tuple(w1))
                    values.append(vec.dic[a])
                else:
                    faces.append(tuple(w1))
                    values.append((-1)*vec.dic[a])
        s = P_chains(faces,values)
        return s
    else:
        return s
    
def is_int(v):
    '''
    A function that determines when an element in a n-composition of tuples is a integer,
    fronzenset, str or ImmtalbeMatrix.
    '''
    if ((isinstance(v, int) == True) or (isinstance(v, str) == True)):
        return True
    if ((isinstance(v, frozenset) == True) or (isinstance(v, ImmutableMatrix) == True)):
        return False
    else:
        aux = True
        for x in v:
            aux = aux and is_int(x)
        return aux
def size_conjugacy_class(partition,n):
    """Returns the number of elements of a conjugacy class.

    ----------
    Parameters
    ----------
    partition: list
        Represents the partitions of a symmetric group (n).
    n: int
        A integer to identify which the symmetric group.
    
    --------
    Returns:
    --------
    int:
        The number of elements of the conjugacy class.
    
    ---------    
    Examples:
    ---------
    For find the number of elements of the conjugacy class of a
    symmetric group use ``size_conjugacy_class(partition,n)''.
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
    """
    Given a partition returns the a representate of a conjugacy class.

    ----------
    Parameters
    ----------
    partition: list
        Represents the partitions of a symmetric group (n).
    
    --------
    Returns:
    --------
    sympy.combinatorics.permutations.Permutation: 
        A representate of the conjugacy class.
    
    ---------
    Examples:
    ---------
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
    """
    Returns a row reduced form of a matrix and a matrix that save the operations.

    ----------
    Parameters
    ----------
    N: Matrix
        The matrix which will be operated.
    
    --------
    Returns:
    --------
    tuple:
        The first element is the matrix which to be multiplied
        by the right to the original matrix, return the row reduced form
        and the other object is the row reduced form of the origin matrix.
        
    ---------
    Examples:
    ---------
    To use this functio use ``Reduce(Matrix)''. We will use the help of the 
    function ``rref`` to verify that the result is right.
        >>> M = np.array([[-1, -1, -1, -1, 0, 0, 0, 0], 
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
    B1=eye(rowCount)
    for r in range(rowCount):    
        if (columnCount <= lead):
            return B1,M
        i = r
        while (M[i, lead] == 0):
            i = i + 1
            if (rowCount == i):
                i = r
                lead = lead + 1
                if (columnCount == lead):
                    return B1,M
        B1.row_swap(i, r)
        M.row_swap(i, r)
        a=M[r,lead]
        for k in range(columnCount):
            M[r,k]=S(M[r,k])/a
        for k in range(rowCount):
            B1[r,k]=S(B1[r,k])/a
        for i in range(0,rowCount):
            if (i != r):
                a=M[i,lead]
                for k in range(0,columnCount):
                    M[i,k]=M[i,k]-M[r,k]*a
                for k in range(rowCount):
                    B1[i,k]=B1[i,k]-B1[r,k]*a
        lead = lead + 1
    return B1,M

def SCG(G):
    '''
    A simplicial complex is associated with a graph by means of its maximal cliques.
    '''
    v = []
    for x in list(nx.find_cliques(G)):
        w = []
        for y in x:
            w.append(y)
        v.append(w)
    return SCFromFacets(v)

def do_square(M):
    '''
    A function that changes one not-square matrix into a square one, adding rows or columns.
    '''
    r = M.shape[0]   
    c = M.shape[1]
    if (r < c):
        for i in range(r,c):
            w = zeros(1,c)
            M = M.row_insert(i, w)
    if (c < r):
        for j in range(c,r):
            w = zeros(r,1)
            M = M.col_insert(j, w)
    return M
            
def matriz_H_in_terms_G(v1,v2):
    '''
    A function that gives a matrix which diagonal entries C_ii are used to find 
    the finite direct sum of cyclic groups of every finitely generated abelian 
    group G. The cyclic groups may be taken to be copies of Z and various 
    C_p^k with p prime, and in this case the cyclic groups are unique up to 
    order and to isomorphism.
    
    ----------
    Parameters
    ----------
    v1: list
        A list with the basis of the additive subgroup G.
        
    v2: list
        A list with the basis of the subgroup H of G.
        
    --------
    Returns:
    --------
    Matrix:
        A matrix which diagonal entries C_ii are used to find 
        the finite direct sum of cyclic groups of every finitely generated abelian 
        group G. The cyclic groups may be taken to be copies of Z and various 
        C_p^k with p prime, and in this case the cyclic groups are unique up to 
        order and to isomorphism.
        
    ---------
    Examples:
    ---------
    To use this functio use ``matriz_H_in_terms_G'':
        >>> v1 = [[1,0,0,0],[1,1,0,0],[1/2,1/2,1/2,1/2],[1/2,1/2,1/2,-1/2]]
        >>> v2 = [[1,-1,0,0],[0,1,-1,0],[0,0,1,-1],[0,0,1,1]]
        >>> print(matriz_H_in_terms_G(v1,v2))
        Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -2, 0], [0, 0, 0, 2]])
    '''
    M = []
    for i in range(len(v1[0])):
        w = []
        for j in range(len(v1)):
            w.append(v1[j][i])
        M.append(w)
    M = Matrix(M)
    w = []
    M1 = []
    for i in v2:
        N = []
        w=[]
        for j in i:
            w.append(np.array([j]))
        N = np.append(M, w, axis=1)
        N = Matrix(N)
        vect = linsolve(N)
        for h in vect:
            w3 = []
            for r in h:
                w3.append(int(r))
            M1.append(w3)
    M1 = Matrix(M1).T
    n = M1.shape[0]
    m = M1.shape[1]
    if (m == n):
        setattr(M1, "ring", ZZ)
        M1 = smith_normal_form(M1)
        return M1
    if (n < m):
        M1 = do_square(M1)
        setattr(M1, "ring", ZZ)
        M1 = smith_normal_form(M1)
        for i in range(n,m):
            M1.row_del(n)
        return M1
    else:
        M1 = do_square(M1)
        setattr(M1, "ring", ZZ)
        M1 = smith_normal_form(M1)
        for i in range(m,n):
            M1.row_del(m)
        return M1

def H_p_using_matrix_c(M):
    '''
    A function that uses the receive the matrix given by ``function matriz_H_in_terms_G''
    to find the finite direct sum of cyclic groups of every finitely generated abelian 
    group G. The cyclic groups may be taken to be copies of Z and various 
    C_p^k with p prime, and in this case the cyclic groups are unique up to 
    order and to isomorphism.
    
    ----------
    Parameters
    ----------
    M: Matrix
        The matrix given by ``function matriz_H_in_terms_G''.
        
    --------
    Returns:
    --------
    v: list
        A list who are the elements in the finite direct sum of cyclic groups who
        represent G.
        
    ---------
    Examples:
    ---------
    To use this functio use ``H_p_using_matrix_c'':
        >>> v1 = [[1,0,0,0],[1,1,0,0],[1/2,1/2,1/2,1/2],[1/2,1/2,1/2,-1/2]]
        >>> v2 = [[1,-1,0,0],[0,1,-1,0],[0,0,1,-1],[0,0,1,1]]
        >>> print(H_p_using_matrix_c(matriz_H_in_terms_G(v1,v2)))
        [('C_', 2), ('C_', 2)]
        
    ..Note:: That means there are a isomorphism between G\H and C_2 x C_2.
        
    '''
    n = M.shape[0]
    m = M.shape[1]
    v = []
    for i in range(n):
        for j in range(m):
            if (i == j):
                if ((abs(M[i,j]) != 1) and (M[i,j] != 0)):
                    v.append(('C_',abs(M[i,j])))
                if (M[i,j] == 0):
                    v.append(('Z',1))
    if (n < m):
        for i in range(n,m):
            v.append(('Z',1))
    return v

def pivotsr(M):
    '''
    Returns a ``list'' of column vectors that belongs to the original matrix and
    that span the columnspace of the matrix.
    '''
    
    c = matrix_rank(M)
    j = 1
    v = [0]
    c1 = 1
    while ((len(v)<c) and (j<M.shape[1])):
        v.append(j)
        u = matrix_rank(M[:,v])
        if (u == c1):
            print(5,j,len(v),c)
            v.remove(j)
            print(6,j,len(v),c)
        c1 = u
        j = j + 1
    return v
class unique_element:
    '''
    class to create combinations for the Young Tableaux.
    '''
    def __init__(self,value,occurrences):
        self.value = value
        self.occurrences = occurrences

def perm_unique(elements):
    '''
    A function use in ``unique_element''.
    '''
    eset=set(elements)
    listunique = [unique_element(i,elements.count(i)) for i in eset]
    u=len(elements)
    return perm_unique_helper(listunique,[0]*u,u-1)

def perm_unique_helper(listunique,result_list,d):
    '''
    A function use in ``unique_element''.
    '''
    if d < 0:
        yield tuple(result_list)
    else:
        for i in listunique:
            if i.occurrences > 0:
                result_list[d]=i.value
                i.occurrences-=1
                for g in  perm_unique_helper(listunique,result_list,d-1):
                    yield g
                i.occurrences+=1
def conjugate(v):
    '''
    A function that conjugate a partition v.
    
    ----------
    Parameters
    ----------
    v: list
        A partition of an integer n.
        
    --------
    Returns:
    --------
    vc: list
        A partition of n that is the conjugate of the partition v.
    
    ---------    
    Examples:
    ---------
    For find the conjugate of v use ``conjugate(v)''.
        >>> v = [5,4,4,1]
        >>> print(conjugate(v))
        [4, 3, 3, 3, 1]
    '''
    vc = []
    c = 1
    j = 1
    while (c != 0):
        c = 0
        for i in v:
            if (i >= j):
                c = c + 1
        if (c != 0):
            vc.append(c)
        j = j + 1
    return vc

def Durfee_square(v):
    '''
    A function that find the lenght of a Durfee square of a partition v of the integer n.
    
    ----------
    Parameters
    ----------
    v: list
        A partition of an integer n.
        
    --------
    Returns:
    --------
    c-2: integer
        The lenght of a Durfee square of a partition v of the integer n.
    
    ---------    
    Examples:
    ---------
    To find the conjugate of v use ``Durfee_square(v)''.
        >>> v = [5,4,4,1]
        >>> v = [5,4,4,1]
        >>> print(Durfee_square(v))
        3
        >>> print(Durfee_square(conjugate(v)))
        3
    '''
    c = 1
    flag = True
    while(flag):
        if (len(v) < c):
            flag = False
        else:
            for i in range(c):
                if (v[i] < c):
                    flag = False
        c = c + 1
    return c-2
def Bouc_theorem(k,n):
    '''
    A function who gives who indicates the decomposition into
    S_n irreducibles of the H_k(M_n) like S_n-module.
    
    ----------
    Parameters
    ----------
    k: integer
        A integer who indicates the kth reduced homology.
        
    n: integer
        A integer who indicates who matching complex M_n will use.
        
    --------
    Returns:
    --------
    v: list
        A list that contains partitions of n, every partitions of n \lambda that appears in v 
        corresponding to the Specht module S^{\lambda} in the decompotition into irreducibles
        of H_k(M_n). If v is empty then H_k(M_n)=0.
        
    ---------    
    Examples:
    ---------
    Use ``Bouc_theorem''.
        >>> k = 1
        >>> n = 5
        >>> print(Bouc_theorem(k,n))
        [[3, 1, 1]]
    ..Note:: The previous means that H_k(M_n) is isomorphic to S^{(3,1,1)}.
    '''
    w =  partitions_list(n)
    v = []
    for h in w:
        if (h == conjugate(h)):
            if (Durfee_square(h) == n-2*(k+1)):
                v.append(h)
    return v
def Frobenius_notation(v):
    '''
    A function that changes a partition v of n into a the Frobenius notation.
    
    ----------
    Parameters
    ----------
    v: list
        A partition of an integer n.
        
    --------
    Returns:
    --------
    alpha: list
        A list that indicates the partition in the right side of the Frobenius notation.
        
    beta: list
        A list that indicates the partition in the left side of the Frobenius notation.
        
    ---------    
    Examples:
    ---------
    Use ``Frobenius_notation''.
        >>> v = [5,4,4,1]
        >>> print(Frobenius_notation(v))
        ([4, 2, 1], [3, 1, 0])
    ..Note:: The previous means that v = (4,2,1|3,1,0)
    '''
    alp = []
    bet = []
    vp = conjugate(v)
    for i in range(1, Durfee_square(v) +1):
        alp.append(v[i-1]-i)
        bet.append(vp[i-1]-i)
    return alp, bet
def almost_self_conjugate_partition(fv):
    '''
    Chech if a partition of an integer n is almost self conjugate.
    
    ----------
    Parameters
    ----------
    fv: tuple
        A partition fv in the Frobenius notation.
        
    --------
    Returns:
    --------
    bool:
        True if fv is almost self conjugate, False otherwise.
    ---------    
    Examples:
    ---------
    Use ``almost_self_conjugate_partition''.
        >>> v = [5,4,4,1]
        >>> fv = Frobenius_notation(v)
        >>> print(almost_self_conjugate_partition(fv))
    ..Note:: The previous means that v = (4,2,1|3,1,0)
    '''
    alp = fv[0]
    bet = fv[1]
    for i in range(0,len(alp)):
        if (alp[i]-1 != bet[i]):
            return False
    return True

def same_lenght(lamb,miu):
    '''
    A function that makes two partitions the same length.
    
    ----------
    Parameters
    ----------
    lamb: list
        A partition.
        
    miu: list
        A partition.
        
    --------
    Returns:
    --------
    lamb: list
        A partition.
        
    miu: list
        A partition.
        
    ---------    
    Examples:
    ---------
    Use ``same_lenght''.
        >>> lamb = [4,2,1]
        >>> miu = [3,1]
        >>> print(same_lenght(lamb,miu))
        ([4, 2, 1], [3, 1, 0])
    '''
    if (len(lamb) > len(miu)):
        for i in range(len(miu), len(lamb)):
            miu.append(0)
    if (len(miu) > len(lamb)):
        for i in range(len(lamb), len(miu)):
            lamb.append(0)
    return lamb, miu

def skew_diagram(lamb,miu):
    '''
    A function that determines if lamb-miu is a skew_diagram
    
    ----------
    Parameters
    ----------
    lamb: list
        A partition.
        
    miu: list
        A partition.
        
    --------
    Returns:
    --------
    bool:
        True if lamb-miu is a skew diagram, False otherwise.
        
    ---------    
    Examples:
    ---------
    Use ``skew_diagram''.
        >>> lamb = [5,4,4,1]
        >>> miu = [4,3,2]
        >>> print(skew_diagram(lamb,miu))
        True
    '''
    lamb, miu = same_lenght(lamb, miu)
    for i in range(len(lamb)):
        if (lamb[i] < miu[i]):
            return False
    return True
def horizontal_strip(lamb,miu):
    '''
    A function that determines if lamb-miu is a horizontal strip.
    
    ----------
    Parameters
    ----------
    lamb: list
        A partition.
        
    miu: list
        A partition.
        
    --------
    Returns:
    --------
    bool:
        True if lamb-miu is a horizontal strip, False otherwise.
        
    ---------    
    Examples:
    ---------
    Use ``horizontal_strip''.
        >>> lamb = [5,4,4,1]
        >>>  miu = [4,3,2]
        >>> print(horizontal_strip(lamb,miu))
        False
    '''
    if (skew_diagram(lamb,miu)):
        lamb, miu = same_lenght(lamb, miu)
        re = []
        for i in range(len(lamb)):
            re.append(lamb[i])
            re.append(miu[i])
        for i in range(len(re)-1):
            if (re[i] < re[i+1]):
                return False
        return True
    else:
        return False
    
    
def vertical_strip(lamb,miu):
    '''
    A function that determines if lamb-miu is a vertical strip.
    
    ----------
    Parameters
    ----------
    lamb: list
        A partition.
        
    miu: list
        A partition.
        
    --------
    Returns:
    --------
    bool:
        True if lamb-miu is a vertical strip, False otherwise.
        
    ---------    
    Examples:
    ---------
    Use ``horizontal_strip''.
        >>> lamb = [5,4,4,1]
        >>>  miu = [4,3,2]
        >>> print(vertical_strip(lamb,miu))
        False
    '''
    if (skew_diagram(lamb,miu)):
        lamb = conjugate(lamb)
        miu = conjugate(miu)
        lamb, miu = same_lenght(lamb, miu)
        re = []
        for i in range(len(lamb)):
            re.append(lamb[i])
            re.append(miu[i])
        for i in range(len(re)-1):
            if (re[i] < re[i+1]):
                return False
        return True
    else:
        return False    

def set_A(v):
    '''
    Implements Set A Proposition 3.3. pg 4. from Wachs and Dong "Combinatorial Laplacian of 
    the Matching Complex.
    '''
    fv = Frobenius_notation(v)
    if (len(fv[0]) < 1):
        return False
    for i in range(len(fv[0])):
        if (fv[0][i] < fv[1][i]):
            return False
    return True

def set_6_d_Mc(v, r):
    '''
    Set use in Example 6, Section 8, pg 137. From MACDONALD "Symmetric Functions and Hall Polynomials".
    '''
    fv = Frobenius_notation(v)
    print(fv)
    if (len(fv[0]) < 1):
        return False
    if (sum(fv[1]) != r):
        return False
    for i in range(len(fv[0])):
        if (fv[0][i] != fv[1][i]-1):
            return False
        if (i != (len(fv[0]) -1)):
            if (fv[1][i] <= fv[1][i+1]):
                return False
        else:
            if (fv[1][i] <= 0):
                return False
    return True


def Prop_6_d_Mc(r, n):
    '''
    Implements Example 6, Section 8, pg 137. From MACDONALD "Symmetric Functions and Hall Polynomials".
    '''
    w =  partitions_list(n)
    D = {}
    for i in w:
        if (set_6_d_Mc(i, r+1)):       
            D[tuple(conjugate(i))] = 1
    return D

def proposition_3_3(r, n):
    '''
    Implements Proposition 3.3. pg 4. from Wachs and Dong "Combinatorial Laplacian of 
    the Matching Complex".
    
    ----------
    Parameters
    ----------
    r: integer
        A integer who indicates the kth Chain space.
        
    n: integer
        A integer who indicates who matching complex M_n will use.
        
    --------
    Returns:
    --------
    D: dict
        A dictionary that contains partitions of n, every partitions of n \lambda that appears in D 
        corresponding to the Specht module S^{\lambda} in the decompotition into irreducibles
        of C_r(M_n). If every value in D is zero then C_r(M_n)=0.
        
    ---------    
    Examples:
    ---------
    Use ``proposition_3_3''.
        >>> r = 1
        >>> n = 5
        >>> print(proposition_3_3(r,n))
        
    ..Note:: The previous means that C_1(M_5) is isomorphic to 
    S^{(3,1,1) \oplus S^{(3,2,0)} \oplus S^{4,1,0}, where oplus is direct sum.
    '''
    w =  partitions_list(n)
    w1 = partitions_list(2*(r+1))
    D = {}
    alr = 0
    for i in w:
        alr = 0
        if (set_A(i)):
            alr = 0
            for j in w1:
                if (almost_self_conjugate_partition(Frobenius_notation(j))):
                    if (horizontal_strip(i,j)):
                        alr = alr + 1
        D[tuple(i)] = alr
    return D
def induction_product(lamb,miu):
    '''
    Computes the dimension of the induction product in the irreducible representations
    by couple of S_n and S_m and decompose them into irreducibles of S_nm.
    
    ----------
    Parameters
    ----------
    lamb: list
        A partition.
        
    miu: list
        A partition.
        
    --------
    Returns:
    --------
    D: dict
        A dictionary that contains partitions of nm which are the decomposition of the 
        induction product.
        
    dimS: integer
        the dimension of the induction product.
        
    ---------    
    Examples:
    ---------
    Use ``induction_product''.
        >>> lamb = [2,2]
        >>> miu = [3]
        >>> print(induction_product(lamb,miu))
        ({(3, 2, 2): 1.0, (4, 2, 1): 1.0, (5, 2): 1.0}, 70.0)
    '''
    n1 = sum(lamb)
    n2 = sum(miu)
    w1 = partitions_list(n1)
    w2 = partitions_list(n2)
    v = []
    v1 = []
    v2 = []
    for i in w1:
        for j in w2:
            v.append(YoungTableaux(lamb,i).CMNR()*YoungTableaux(miu,j).CMNR())
            v1.append((i,j))
            v2.append(size_conjugacy_class(i,n1)*size_conjugacy_class(j,n2))
    D = {}
    dimS = 0
    triv = []
    for i in range(n1 + n2):
        triv.append(1)
    for i in partitions_list(n1+n2):
        s = 0
        pr = []
        for j in range(len(v1)):
            cl = v1[j][0] + v1[j][1]
            cl = sorted(cl)
            cl = cl[::-1]
            s = s + YoungTableaux(i,cl).CMNR()*v2[j]*v[j]
            pr.append(YoungTableaux(i,cl).CMNR()*v2[j]*v[j])
        val = s/(math.factorial(n1)*math.factorial(n2))
        if (val != 0):
            D[tuple(i)] = val
#            dimS = dimS + YoungTableaux(i,triv).CMNR()
            dimS = dimS + dim_lamb(i)
    return D, dimS
def hook_product(lamb, lambconj, i, j):
    '''
    Implements Theorem 2.2.5 (Frame-Robinson-Thrall hook length formula). From Wachs
    "Poset Topology: Tools and Applications".
    '''
    return lamb[i-1] + lambconj[j-1] - i - j + 1
        
def product_hook_product(lamb):
    '''
    Implements Example 1, Section 1. From MACDONALD "Symmetric Functions and Hall Polynomials".
    '''
    miu = conjugate(lamb)
    lamb, miu = same_lenght(lamb, miu)
    p = 1
    for i in range(1,len(lamb)+1):
        for j in range(1,lamb[i-1]+1):
            p=p*hook_product(lamb, miu, i, j)
    return p
 
def dim_lamb(lamb):
    '''
    Implements Theorem 2.2.5 (Frame-Robinson-Thrall hook length formula). From Wachs
    "Poset Topology: Tools and Applications".
    '''
    return math.factorial(sum(lamb))/product_hook_product(lamb)


def pieri_rule_1_1(lamb, m):
    '''
    Implements the first part of the Theorem 2.2.11 (Pieri’s rule). From Wachs
    "Poset Topology: Tools and Applications".
    '''
    n = sum(lamb)
    w = partitions_list(n + m)
    s = 0
    D = {}
    for miu in w:
        miua = miu.copy()
        if (horizontal_strip(miu,lamb)):
            D[tuple(miua)] = 1
            s = s + dim_lamb(list(miua))  
    return D, s

def pieri_rule_1_2(lamb, m):
    '''
    Implements the second part of the Theorem 2.2.11 (Pieri’s rule). From Wachs
    "Poset Topology: Tools and Applications".
    '''
    n = sum(lamb)
    w = partitions_list(n + m)
    s = 0
    D = {}
    for miu in w:
        miua = miu.copy()
        if (vertical_strip(miu,lamb)):
            D[tuple(miua)] = 1
            s = s + dim_lamb(list(miua))  
    return D, s