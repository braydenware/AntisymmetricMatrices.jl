module AntisymmetricMatrices
using AutoHashEquals
#### More specialized matrix types ####

import Base.LinAlg: char_uplo, checksquare, HessenbergQ, @commutative
export isantisymmetric, Antisymmetric, AntisymTridiagonal, AntisymBlockdiagonal, pfaffian, pfaffiansign

function isantisymmetric(A::AbstractMatrix)
    n = checksquare(A)
    for i in 1:n, j in i:n
        if A[i,j]!=-ctranspose(A[j,i])
            return false
        end
    end
    return true
end

## (complex) antisymmetric tridiagonal matrices

"""
    AntisymTridiagonal(ev)

Construct a antisymmetric tridiagonal matrix from the first super-diagonal.
The result is of type `AntisymTridiagonal` and provides efficient specialized
eigensolvers, but may be converted into a regular matrix with
[`convert(Array, _)`](@ref) (or `Array(_)` for short).

# Example

```jldoctest
julia> ev = [7; 8; 9]
3-element Array{Int64,1}:
 7
 8
 9

julia> AntisymTridiagonal(ev)
4×4 AntisymTridiagonal{Int64}:
  ⋅   7   ⋅   ⋅
 -7   ⋅   8   ⋅
  ⋅  -8   ⋅   9
  ⋅   ⋅  -9   ⋅
```
"""
struct AntisymTridiagonal{T} <: AbstractMatrix{T}
    ev::Vector{T} # superdiagonal
    AntisymTridiagonal{T}(ev::Vector{T}) where T = new{T}(ev)
end

"""
    AntisymBlockdiagonal(λ)

Canonical form of an antisymmetric matrix as a vector of 2x2 antisymmetric blocks on the diagonal
followed by a number of 1x1 antisymmetric blocks (aka zeros on the diagonal).
The result is of type `AntisymBlockdiagonal` and provides efficient specialized
eigensolvers, but may be converted into a regular matrix with
[`convert(Array, _)`](@ref) (or `Array(_)` for short).

# Example

```jldoctest
julia> λ = [7; 8]
2-element Array{Int64,1}:
 7
 8

julia> AntisymBlockdiagonal(λ, 1)
5×5 AntisymBlockdiagonal{Int64}:
  ⋅   7   ⋅   ⋅   ⋅
 -7   ⋅   ⋅   ⋅   ⋅
  ⋅   ⋅   ⋅   8   ⋅
  ⋅   ⋅  -8   ⋅   ⋅
  ⋅   ⋅   ⋅   ⋅   ⋅
```
"""
struct AntisymBlockdiagonal{T} <: AbstractMatrix{T}
    λ::Vector{T}
    n::Int
    AntisymBlockdiagonal{T}(λ::Vector{T}, n::Int) where T = new{T}(λ, n)
end

AntisymTridiagonal(ev::Vector{T}) where T = AntisymTridiagonal{T}(ev)
AntisymBlockdiagonal(λ::Vector{T}, n::Int=0) where T = AntisymBlockdiagonal{T}(λ, n)

#######################################################################################################
# making the print function show the structure

function Base.replace_in_print_matrix(A::AntisymTridiagonal, i::Integer, j::Integer, s::AbstractString)
    i==j-1||i==j+1 ? s : Base.replace_with_centered_mark(s)
end

function Base.replace_in_print_matrix(A::AntisymBlockdiagonal, i::Integer, j::Integer, s::AbstractString)
    (i+1==j&&iseven(j))||(i==j+1&&iseven(i)) ? s : Base.replace_with_centered_mark(s)
end

function Base.show(io::IO, M::AntisymTridiagonal)
    println(io, summary(M), ":")
    print(io, "superdiag:")
    Base.print_matrix(io, (M.ev)')
end

function Base.show(io::IO, M::AntisymBlockdiagonal)
    println(io, summary(M), "with $(M.n) zero blocks and block values:")
    Base.print_matrix(io, (M.λ)')
end

#############################################################################################
# Creation and conversion

function AntisymTridiagonal(A::AbstractMatrix)
    if diag(A,1) == -1*diag(A,-1)
        AntisymTridiagonal(diag(A,1))
    else
        throw(ArgumentError("matrix is not antisymmetric; cannot convert to AntisymTridiagonal"))
    end
end

Base.convert(::Type{AntisymTridiagonal{T}}, S::AntisymTridiagonal) where {T} =
    AntisymTridiagonal(convert(Vector{T}, S.ev))
Base.convert(::Type{AbstractMatrix{T}}, S::AntisymTridiagonal) where {T} =
    AntisymTridiagonal(convert(Vector{T}, S.ev))
function Base.convert(::Type{Matrix{T}}, M::AntisymTridiagonal{T}) where T
    n = size(M, 1)
    Mf = zeros(T, n, n)
    @inbounds begin
        @simd for i = 1:n-1
            Mf[i+1,i] = -1*M.ev[i]
            Mf[i,i+1] = M.ev[i]
        end
    end
    return Mf
end
Base.convert(::Type{Matrix}, M::AntisymTridiagonal{T}) where {T} = convert(Matrix{T}, M)
Base.convert(::Type{Array}, M::AntisymTridiagonal) = convert(Matrix, M)
Base.full(M::AntisymTridiagonal) = convert(Array, M)

Base.convert(::Type{AntisymBlockdiagonal{T}}, S::AntisymBlockdiagonal) where {T} =
    AntisymBlockdiagonal(convert(Vector{T}, S.λ), S.n)
Base.convert(::Type{AbstractMatrix{T}}, S::AntisymBlockdiagonal) where {T} =
    AntisymBlockdiagonal(convert(Vector{T}, S.λ), S.n)
function Base.convert(::Type{Matrix{T}}, M::AntisymBlockdiagonal{T}) where T
    m = length(M.λ)
    n = M.n
    N = 2m+n
    Mf = zeros(T, N, N)
    for i = 1:m
        Mf[2*i,2*i-1] = -1*M.λ[i]
        Mf[2*i-1,2*i] = M.λ[i]
    end
    return Mf
end
Base.convert(::Type{Matrix}, M::AntisymBlockdiagonal{T}) where {T} = convert(Matrix{T}, M)
Base.convert(::Type{Array}, M::AntisymBlockdiagonal) = convert(Matrix, M)
Base.full(M::AntisymBlockdiagonal) = convert(Array, M)

Base.convert(::Type{Tridiagonal}, M::AntisymTridiagonal{T}) where T = Tridiagonal(-M.ev, zeros(T, size(M,1)), M.ev)

###################################################################################################
# Methods: size, similar, conj, copy, real, imag, transpose, ctranspose, diag, +, -, *, /, ==

Base.size(A::AntisymTridiagonal) = (length(A.ev)+1, length(A.ev)+1)
function Base.size(A::AntisymTridiagonal, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d<=2
        return length(A.ev)+1
    else
        return 1
    end
end

Base.size(A::AntisymBlockdiagonal) = (2*length(A.λ)+A.n, 2*length(A.λ)+A.n)
function Base.size(A::AntisymBlockdiagonal, d::Integer)
    if d < 1
        throw(ArgumentError("dimension must be ≥ 1, got $d"))
    elseif d<=2
        return 2*length(A.λ)+A.n
    else
        return 1
    end
end

Base.similar(S::AntisymTridiagonal, ::Type{T}) where {T} = AntisymTridiagonal{T}(similar(S.ev, T))
Base.similar(S::AntisymBlockdiagonal, ::Type{T}) where {T} = AntisymTridiagonal{T}(similar(S.λ, T), S.n)

for func in (:(Base.conj), :(Base.copy), :(Base.real), :(Base.imag))
    @eval ($func)(M::AntisymTridiagonal) = AntisymTridiagonal(($func)(M.ev))
    @eval ($func)(M::AntisymBlockdiagonal) = AntisymBlockdiagonal(($func)(M.λ), M.n)
end

Base.transpose(M::AntisymTridiagonal) = -1*M
Base.ctranspose(M::AntisymTridiagonal) = conj(-1*M)

Base.transpose(M::AntisymBlockdiagonal) = -1*M
Base.ctranspose(M::AntisymBlockdiagonal) = conj(-1*M)

function Base.diag(M::AntisymTridiagonal{T}, n::Integer=0) where T
    absn = abs(n)
    if n == 1
        return M.ev
    elseif n==-1
        return -1*M.ev
    elseif absn<size(M,1)
        return zeros(T,size(M,1)-absn)
    else
        throw(ArgumentError("$n-th diagonal of a $(size(M)) matrix doesn't exist!"))
    end
end

function Base.diag(M::AntisymBlockdiagonal{T}, n::Integer=0) where T
    absn = abs(n)
    if n == 1
        l = size(M.λ)
        d = zeros(T, 2*l+M.n-1)  
        d[1:2:2l-1] = M.λ
        return d
    elseif n==-1
        l = size(M.λ)
        d = zeros(T, 2*l+M.n-1)  
        d[1:2:2l-1] = -1*M.λ
        return d
    elseif absn<size(M,1)
        return zeros(T,size(M,1)-absn)
    else
        throw(ArgumentError("$n-th diagonal of a $(size(M)) matrix doesn't exist!"))
    end
end

Base.:+(A::AntisymTridiagonal, B::AntisymTridiagonal) = AntisymTridiagonal(A.ev+B.ev)
Base.:-(A::AntisymTridiagonal, B::AntisymTridiagonal) = AntisymTridiagonal(A.ev-B.ev)
@commutative Base.:*(A::AntisymTridiagonal, B::Number) = AntisymTridiagonal(A.ev*B)
Base.:/(A::AntisymTridiagonal, B::Number) = AntisymTridiagonal(A.ev/B)
Base.:(==)(A::AntisymTridiagonal, B::AntisymTridiagonal) = (A.ev==B.ev)

function Base.:+(A::AntisymBlockdiagonal, B::AntisymBlockdiagonal)
    if !(size(A, 1) == size(B, 1))    
        throw(DimensionMismatch("A has first dimension $(size(S,1)), B has $(size(B,1)); must match"))
    end
    n = min(A.n, B.n)
    if n==A.n
        λ = copy(A.λ)
        λ[1:length(B.λ)] += B.λ
    else
        λ = copy(B.λ)
        λ[1:length(A.λ)] += A.λ
    end 
    AntisymBlockdiagonal(λ, n)
end

Base.:-(A::AntisymBlockdiagonal, B::AntisymBlockdiagonal) = A+(-1*B)

@commutative Base.:*(A::AntisymBlockdiagonal, B::Number) = AntisymBlockdiagonal(A.λ*B, A.n)
Base.:/(A::AntisymBlockdiagonal, B::Number) = AntisymBlockdiagonal(A.λ/B, A.n)
Base.:(==)(A::AntisymBlockdiagonal, B::AntisymBlockdiagonal) = (A.λ==B.λ && A.n==B.n)

const tridiagtypes = Union{Diagonal, Bidiagonal, Tridiagonal, SymTridiagonal}
@commutative Base.:+(A::AntisymTridiagonal, B::tridiagtypes) = convert(Tridiagonal, A) + convert(Tridiagonal, B)
Base.:-(A::AntisymTridiagonal, B::tridiagtypes) = convert(Tridiagonal, A) - convert(Tridiagonal, B)
Base.:-(A::tridiagtypes, B::AntisymTridiagonal) = convert(Tridiagonal, A) - convert(Tridiagonal, B)

###################################################################################################
# Implementation of specialized matrix multiplication with A_mul_B

function Base.A_mul_B!(C::StridedVecOrMat, S::AntisymTridiagonal, B::StridedVecOrMat)
    m, n = size(B, 1), size(B, 2)
    if !(m == size(S, 1) == size(C, 1))
        throw(DimensionMismatch("A has first dimension $(size(S,1)), B has $(size(B,1)), C has $(size(C,1)) but all must match"))
    end
    if n != size(C, 2)
        throw(DimensionMismatch("second dimension of B, $n, doesn't match second dimension of C, $(size(C,2))"))
    end

    β = S.ev
    @inbounds begin
        @simd for j = 1:n
            C[1, j] = B[2, j]*β[1]
            @simd for i = 2:m - 1
                C[i, j] = B[i+1, j]*β[i] - B[i-1, j]*β[i-1]
            end
            C[m, j] = -B[m-1, j]*β[m-1]
        end
    end

    return C
end

###################################################################################################
# Implementation of specialized inv, det, and pfaffian

#Implements the inverse using the recurrence relation between principal minors
# special cased from inv_usmani in julia/base/linalg/tridiag.jl
#Reference:
#    R. Usmani, "Inversion of a tridiagonal Jacobi matrix",
#    Linear Algebra and its Applications 212-213 (1994), pp.413-414
#    doi:10.1016/0024-3795(94)90414-6

function inv_usmani(c::Vector{T}) where T
    n = length(c)+1
    θ = zeros(T, n+1) #principal minors of A
    θ[1] = 1
    n>=1 && (θ[2] = zero(T))
    for i=2:n
        θ[i+1] = (c[i-1]^2)*θ[i-1]
    end
    φ = zeros(T, n+1)
    φ[n+1] = 1
    n>=1 && (φ[n] = zero(T))
    for i=n-1:-1:1
        φ[i] = (c[i]^2)*φ[i+2]
    end
    α = Matrix{T}(n, n)
    for i=1:n, j=1:n
        sign = (i+j)%2==0 ? (+) : (-)
        if i<j
            α[i,j]=(sign)(prod(c[i:j-1]))*θ[i]*φ[j+1]/θ[n+1]
        elseif i==j
            α[i,i]=                       θ[i]*φ[i+1]/θ[n+1]
        else #i>j
            α[i,j]=(sign)(prod(-1*c[j:i-1]))*θ[j]*φ[i+1]/θ[n+1]
        end
    end
    α
end
Base.inv(A::AntisymTridiagonal) = inv_usmani(A.ev)

minv(q::Number) = -1./q

Base.inv(A::AntisymBlockdiagonal) = (A.n==0 || throw(error("Can't invert due to zero columns. Try pinv")); AntisymBlockdiagonal(minv.(A.λ)))
Base.pinv(A::AntisymBlockdiagonal) = AntisymBlockdiagonal(minv.(A.λ), A.n)

function pfaff(c::Vector{T}) where T
    n = length(c)+1
    if n == 0
        return one(T)
    elseif isodd(n)
        return zero(T)
    end
    return prod(c[1:2:n-1])
end

function pfaffsign(c::Vector{T}) where T
    n = length(c)+1
    if n == 0
        return one(T)
    elseif isodd(n)
        return zero(T)
    end
    return prod(sign.(c[1:2:n-1]))
end

pfaffian(A::AntisymTridiagonal) = pfaff(A.ev)
Base.det(A::AntisymTridiagonal) = pfaffian(A)^2
pfaffiansign(A::AntisymTridiagonal) = pfaffsign(A.ev)

pfaffian(A::AntisymBlockdiagonal{T}) where T = (A.n==0 ? prod(A.λ) : zero(T))
Base.det(A::AntisymBlockdiagonal) = pfaffian(A)^2
pfaffiansign(A::AntisymBlockdiagonal{T}) where T = (A.n==0 ? prod(sign.(A.λ)) : zero(T))


function Base.getindex(A::AntisymTridiagonal{T}, i::Integer, j::Integer) where T
    if !(1 <= i <= size(A,1) && 1 <= j <= size(A,2))
        throw(BoundsError(A, (i,j)))
    end
    if i == j + 1
        return -A.ev[j]
    elseif i + 1 == j
        return A.ev[i]
    else
        return zero(T)
    end
end

function Base.setindex!(A::AntisymTridiagonal, x, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    throw(ArgumentError("cannot set entry ($i, $j)"))
    return x
end

function Base.getindex(A::AntisymBlockdiagonal{T}, i::Integer, j::Integer) where T
    if !(1 <= i <= size(A,1) && 1 <= j <= size(A,2))
        throw(BoundsError(A, (i,j)))
    end
    if i == j + 1 && iseven(i) && i <= 2*length(A.λ)
        return -A.λ[i>>1]
    elseif i + 1 == j && iseven(j) && j <= 2*length(A.λ)
        return A.λ[j>>1]
    else
        return zero(T)
    end
end

function Base.setindex!(A::AntisymBlockdiagonal, x, i::Integer, j::Integer)
    @boundscheck checkbounds(A, i, j)
    throw(ArgumentError("cannot set entry ($i, $j)"))
    return x
end

###########################################################################################################################

@auto_hash_equals struct AntisymTridiagonalSchur{T,S<:AbstractMatrix{T}} <: Factorization{T}
    W::S
    values::Vector{T}
    Vt::S
    AntisymTridiagonalSchur{T,S}(W::AbstractMatrix{T}, values::Vector{T}, Vt::AbstractMatrix{T}) where {T,S} = new(W, values, Vt)
end
AntisymTridiagonalSchur(W::AbstractMatrix{T}, values::Vector{T}, Vt::AbstractMatrix{T}) where T = AntisymTridiagonalSchur{T, typeof(W)}(W, values, Vt)

function Base.schurfact(A::AntisymTridiagonal{T}) where T<:Real
    if iseven(size(A, 1))
        C = Bidiagonal(A.ev[1:2:end], -A.ev[2:2:end], false)
    else
        C = Bidiagonal(vcat(A.ev[1:2:end], T[zero(T)]), -A.ev[2:2:end], false)
    end
    U, s, Vt = svd(C)
    if isodd(size(A, 1))
        @assert s[end]==0
        s = s[1:end-1]
        Vt = Vt[1:end-1, 1:end-1]
    end
    return AntisymTridiagonalSchur(U, s, Vt)
end

function unpackZ(F::AntisymTridiagonalSchur{T}) where T
    # n = length(F.values)
    # N = 2n
    W = F.W
    Vt = F.Vt
    m = size(W, 1)
    n = size(Vt, 1)
    N = m+n
    if isodd(N)
        @assert m==n+1
    else
        @assert m==n
    end
    Z = zeros(T, N, N)
    for i in 1:m, j in 1:m
        Z[2i-1, 2j-1] = W[i, j]
    end
    for i in 1:n, j in 1:n
        Z[2i, 2j] = Vt[i, j]
    end
    return Z
end

unpack(F::AntisymTridiagonalSchur) = F.values, unpackZ(F)

function Base.schur(A::AntisymTridiagonal{T}) where T<:Real
    F = schurfact(A)
    values, vecs = unpack(F)
    A0 = AntisymBlockdiagonal(values, size(A, 1)%2)
    evals = Complex{T}[sgn*im*v for v in values for sgn in [1, -1]]
    return A0, vecs, evals
end

##########################################################################################################################
# Next: add generic antisymmetric matrix schur factorization using first a reduction to tridiagonal

struct Antisymmetric{T, S<:AbstractMatrix} <: AbstractMatrix{T}
    data::S
    uplo::Char
end

Antisymmetric(A::AbstractMatrix, uplo::Char='U') = (checksquare(A); Antisymmetric{eltype(A),typeof(A)}(A, uplo))
Antisymmetric(A::AbstractMatrix, uplo::Symbol) = Antisymmetric(A, char_uplo(uplo))

Base.size(A::Antisymmetric, d) = size(A.data, d)
Base.size(A::Antisymmetric) = size(A.data)

@inline function Base.getindex(A::Antisymmetric{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A, i, j)
    @inbounds r = (i==j) ? zero(T) : (A.uplo == 'U') == (i < j) ? A.data[i, j] : -A.data[j, i]
    r
end

function Base.setindex!(A::Antisymmetric, v, i::Integer, j::Integer)
    throw(ArgumentError("Cannot set an index in a antisymmetric matrix"))
end

function Base.replace_in_print_matrix(A::Antisymmetric, i::Integer, j::Integer, s::AbstractString)
    i!=j ? s : Base.replace_with_centered_mark(s)
end

Base.similar(A::Antisymmetric, ::Type{T}) where {T} = Antisymmetric(similar(A.data, T))

Base.convert(::Type{Matrix}, A::Antisymmetric) = copyatri!(convert(Matrix, copy(A.data)), A.uplo)

function copyatri!(A::AbstractMatrix{T}, uplo::Char) where T
    n = checksquare(A)
    for i =1:n
        A[i, i] = zero(T)
    end
    if uplo == 'U'
        for i = 1:(n-1), j = (i+1):n
            A[j,i] = -A[i,j]
        end
    elseif uplo == 'L'
        for i = 1:(n-1), j = (i+1):n
            A[i,j] = -A[j,i]
        end
    else
        throw(ArgumentError("uplo argument must be 'U' (upper) or 'L' (lower), got $uplo"))
    end
    A
end

Base.convert(::Type{Array}, A::Antisymmetric) = convert(Matrix, A)
Base.full(A::Antisymmetric) = convert(Array, A)
Base.parent(A::Antisymmetric) = A.data
Base.copy(A::Antisymmetric{T,S}) where {T,S} = (B = copy(A.data); Antisymmetric{T,typeof(B)}(B,A.uplo))

function Base.copy!(dest::Antisymmetric, src::Antisymmetric)
    if src.uplo == dest.uplo
        copy!(dest.data, src.data)
    else
        transpose!(dest.data, -1*src.data)
    end
    return dest
end

isantisymmetric(A::Antisymmetric) = true
Base.transpose(A::Antisymmetric) = -A
Base.ctranspose(A::Antisymmetric{<:Real}) = -A
Base.trace(A::Antisymmetric) = zero(eltype(A))

Base.:-(A::Antisymmetric{Tv,S}) where {Tv,S<:AbstractMatrix} = Antisymmetric{Tv,S}(-A.data, A.uplo)
Base.:*(A::Antisymmetric, B::StridedMatrix) = full(A)*B
Base.:*(A::StridedMatrix, B::Antisymmetric) = A*full(B)

Base.:*(A::Antisymmetric, x::Number) = Antisymmetric(A.data*x, A.uplo)
Base.:/(A::Antisymmetric, x::Number) = Antisymmetric(A.data/x, A.uplo)

##############################################################

@auto_hash_equals struct AntisymmetricSchur{T,S<:AbstractMatrix{T}} <: Factorization{T}
    Q::HessenbergQ{T, S}
    F::AntisymTridiagonalSchur{T, S}
    AntisymmetricSchur{T,S}(Q::HessenbergQ{T, S}, F::AntisymTridiagonalSchur{T, S}) where {T,S} = new{T, S}(Q, F)
end
AntisymmetricSchur(Q::HessenbergQ{T, S}, F::AntisymTridiagonalSchur{T, S}) where {T, S} = AntisymmetricSchur{T, S}(Q, F)

function Base.schurfact(A::Antisymmetric{T}) where T<:Real
    F = hessfact!(copy(Array(A))) # converts A into a tridiagonal form using a unitary rotation F[:Q]
    Atri = AntisymTridiagonal(diag(F.factors, 1))
    G = schurfact(Atri) # diagonalizes the antisymmetric tridiagonal matrix
    return AntisymmetricSchur(F[:Q], G)
end

function Base.schur(A::Antisymmetric{T}) where T<:Real
    F = schurfact(A)
    Z = unpackZ(F.F)

    A0 = AntisymBlockdiagonal(F.F.values, size(A, 1)%2)
    
    evals = Complex{T}[sgn*im*v for v in F.F.values for sgn in [1, -1]]
    if isodd(size(A, 1))
        evals = vcat(evals, zero(T))
    end
    Q = F.Q
    return  A0, Q*Z, evals
end

unpack(F::AntisymmetricSchur) = (F.F.values, F.Q*unpackZ(F.F))

function pfaffian(A::Antisymmetric{T}) where T<:Real
    n = size(A, 1)
    if n == 0
        return one(T)
    elseif isodd(n)
        return zero(T)
    end
    F=hessfact!(copy(Array(A)))
    Atri = AntisymTridiagonal(diag(F.factors, 1))
    return det(F[:Q])*pfaffian(Atri)
end

function pfaffiansign(A::Antisymmetric{T}) where T<:Real
    n = size(A, 1)
    if n == 0
        return one(T)
    elseif isodd(n)
        return zero(T)
    end
    F=hessfact!(Array(A))
    Atri = AntisymTridiagonal(diag(F.factors, 1))
    return sign(det(F[:Q]))*pfaffiansign(Atri)
end

end # module
