
import sympy as sp
import random
from sympy.combinatorics.named_groups import SymmetricGroup
init_printing()
from sympy.physics.quantum.dagger import Dagger
from sympy.combinatorics.named_groups import DihedralGroup
from sympy.matrices import Matrix
from sympy.matrices import GramSchmidt
from sympy import BlockMatrix
from sympy import Symbol, I
from sympy import radsimp
from sympy.simplify.radsimp import collect_sqrt

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


def unitary_representation(G,d):
    """
    A function that given a matrix representation become it 
    in a unitary matrix representation.
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
    """A function that determine if a matrix representation is irreducible, 
    if is reducible, return a matrix non escalar that reduce the 
    matrix representation in otherwise return True"""
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
    """A function that return where end the blocks of a matrix."""
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
    identity matrix of degree n"""
    a=M.shape[0]
    N=eye(n)
    for j in range(0,a):
        for k in range(0,a):
            N[j+i,k+i]=M[j,k]
    return N

def reduce(G,d):
    """This function give a matrix that reduce the matrix representation"""
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
