using Base.Test

N = 10

C = rand(N, N)
A = C - C'

T, Z, λ = schur(A)
@test Z*T*Z' ≈ A

##########################

A = Antisymmetric(A)
T, Z, λ = schur(A)

@test Z*T*Z' ≈ A

#############################

A = AntisymTridiagonal(rand(N))
@show A

T, Z, λ = schur(A)
@test Z*T*Z' ≈ A


