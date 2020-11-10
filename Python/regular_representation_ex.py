# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 13:08:17 2020

@author: Hp
"""


import sympy as sp
from sympy.combinatorics.named_groups import SymmetricGroup
from sympy import *
init_printing(use_unicode=True)
from sympy.physics.quantum.dagger import Dagger
from sympy.combinatorics.named_groups import DihedralGroup
from sympy.matrices import Matrix
from sympy.matrices import GramSchmidt
from sympy import BlockMatrix
from sympy import Symbol, I
from sympy import radsimp
from sympy.simplify.radsimp import collect_sqrt


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
    