from sympy import *
import random
from sympy.combinatorics.named_groups import SymmetricGroup
init_printing()
from sympy.physics.quantum.dagger import Dagger
from sympy.combinatorics.named_groups import DihedralGroup
from sympy.matrices import Matrix
from sympy.matrices import GramSchmidt
from sympy import BlockMatrix
from sympy import Symbol, I
def ublock(M,N):
    """ublock hace una matriz diagonal por bloques con dos matrices dadas"""
    m=M.shape[0]
    n=N.shape[0]
    l=m+n
    L=zeros(l,l)
    for i in range(l):
        for j in range(l):
            if (i<m and j<m):
                L[i,j]=M[i,j]
            if (i>=m and j>=m):
                L[i,j]=N[i-m,j-m]
    return L
def MTS(A):
    """Calcula una matriz triangular superior"""
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
def regular_representation(G):
    elems = list(G.elements)
    n = len(elems)
    def char_function(i, j):
        if elems[i]*g == elems[j]:
            return 1
        else:
            return 0
    mydict = {}
    for g in elems:
        mydict[g] = Matrix(n, n, char_function)
    return mydict
def rho(G):
    D={}
    for g in list(G.elements):
        n=max(g)+1
        M=eye(n)
        N=eye(n)
        for i in range(0,n):
            for j in range(0,n):
                N[i,j]=M[g(i),j]
        D[g]=N
    return D
def unit(G,D):
    """Hace unitaria la representación"""
    n=D[G[0]].shape[0]
    A=zeros(n,n)
    for g in D:
        J=Dagger(D[g])*D[g]
        J.simplify()
        A=J+A
    C=MTS(A)
    M = {}
    for g in list(G.elements):
        M[g]=(C.inv())*D[g]*C
    return M
def ME(G,Ers,D):
    """Forma las matrices a las cuales el libro de Dixon llama "E" """
    """La matriz E conmuta con todo elemento de la representación"""
    a=D[G[0]].shape[0]
    E=zeros(a,a)
    R=unit(G,D)
    for g in R:
        E=E+(Dagger(R[g])*Ers*R[g])
    E.simplify()
    E=(sympify(1)/a)*E
    return E
def irreducible(G,D):
    """Determina si una representación es irreducible"""
    """En caso de no serlo, regresa la matriz no escalar que conmuta"""
    """con todos los elementos de la representación"""
    n=D[G[0]].shape[0]
    N=eye(n)
    L=zeros(n,n)
    v=True
    for r in range(0,n):
        for s in range(0,n):
            H=zeros(n)        
            if (r==s):
                H[r,r]=1
            else:
                if (r>s):
                    H[r,s]=1
                    H[s,r]=1
                else:
                    H[r,s]=1*I
                    H[s,r]=-1*I
            M=ME(G,H,D)
            if (M!=M[0,0]*N):
                v=False
                L=M
    if (v==True):
        return v
    else:
        return L
     
##########################################
def block(M):
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
    a=M.shape[0]
    N=eye(n)
    for j in range(0,a):
        for k in range(0,a):
            N[j+i,k+i]=M[j,k]
    return N
def reducir(D,G):
    M=irreducible(G,D)
    b=D[G[0]].shape[0]
    if (M==True):
        return(eye(b))
    else:
        (P, J) = M.jordan_form()
        w=[]
        for g in D:
            w.append(block(P.inv()*D[g]*P))
        l=len(w[0])
        au=w[0]
        for g in w:
            if (len(g)<l):
                l=len(g)
                au=g
        e=0
        U=P
        for a in g:
            D1={}
            for g in list(G.elements):
                D1[g]=(P.inv()*D[g]*P)[e:a+1,e:a+1]
            U=U*blockI(reducir(D1,G),b,e)
            e=a+1
        return U
        #v=block(P.inv()*D[G[0]]*P)
        #I=[]
        #s=0
        #for g in v:
        #    I.append(s)
        #    s=s+g.shape[0]
        #U=blockI(P,n,i)
        #for a in range(0,len(v)):
        #    D1={}
        #    for g in list(G.elements):
        #        if (g!=G.identity()):
        #            D1[g]=block(P.inv()*D[g]*P)[a]
        #        else:
        #            D1[g]=eye(v[a].shape[0])
        #    U=(U*blockI(reducir(D1,n,i+a,G))).simplify()
        #return U


G=SymmetricGroup(3)
D=regular_representation(G)
H=DihedralGroup(4)
K=rho(H)
#n=D[G[0]].shape[0]
#i=0
M=reducir(D,G)
for g in list(G.elements):
    display(M.inv()*D[g]*M)
