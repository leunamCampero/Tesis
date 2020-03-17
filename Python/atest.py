from sympy.combinatorics.named_groups import SymmetricGroup

S3 = SymmetricGroup(3)

rep = regular_representation(S3)

P = rep._matrix_to_reduce(1, 3)

(Q, J) = P.jordan_form()

Q = Q.applyfunc(simplifier_function)

repr = rep.equivalent_by(Q)
