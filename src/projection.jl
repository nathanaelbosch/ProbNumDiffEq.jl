function projection(d::Integer, q::Integer, ::Val{elType}=Val(typeof(1.0))) where {elType}
    # Proj(deriv) = kron(diagm(0 => ones(elType, d)), [i==(deriv+1) ? 1 : 0 for i in 1:q+1]')

    # Slightly faster version of the above:
    D = d * (q + 1)
    Proj(deriv) = begin
        P = zeros(elType, d, D)
        @simd ivdep for i in deriv*d+1:D+1:d*D
            @inbounds P[i] = 1
        end
        return P
    end
    return Proj
end


"""Projection matrices with non-allocating multiplication by using views."""
struct ProjMatGenerator{d,q} end
ProjMatGenerator(d,q) = ProjMatGenerator{d,q}()
(P::ProjMatGenerator{d,q})(p::Int) where {d,q} = ProjMat(d,q,p)
struct ProjMat{d,q,p} <: LinearMaps.LinearMap{Bool} end
ProjMat(d,q,p) = ProjMat{d,q,p}()
Base.size(P::ProjMat{d,q,p}) where {d,q,p} = (d, d*(q+1))
*(P::ProjMat{d,q,p}, v::AbstractVector) where {d,q,p} = view(v, p+1:q+1:d*(q+1))
*(P::ProjMat{d,q,p}, M::AbstractMatrix) where {d,q,p} = view(M, p+1:q+1:d*(q+1), :)
mul!(out::AbstractVector, P::ProjMat, v::AbstractVector) = out .= P*v
mul!(out::AbstractVector, P::ProjMat, v::AbstractVector, a::Number, b::Number) =
    (out .*= b; out .+= b .* (P*v); out)
mul!(out::AbstractMatrix, P::ProjMat, M::AbstractMatrix) = out .= P*M
mul!(out::AbstractMatrix, P::ProjMat, M::AbstractMatrix, a::Number, b::Number) =
    (out .*= b; out .+= b .* (P*M); out)
LinearMaps.MulStyle(P::ProjMat) = LinearMaps.FiveArg()
Base.Matrix(P::ProjMat{d,q,p}) where {d,q,p} =
    kron(diagm(0 => ones(d)), [i==(p+1) ? 1 : 0 for i in 1:q+1]')
# LinearAlgebra.adjoint(P::ProjMat{d,q,p}) where {d,q,p} =

*(P::LinearMaps.TransposeMap{<:Any, <:ProjMat}, v::AbstractVector) =
    mul!(similar(v, size(P)[1]), P, v)
*(P::LinearMaps.TransposeMap{<:Any, <:ProjMat}, M::AbstractMatrix) =
    mul!(similar(M, size(P)[1], size(M)[2]), P, M)
*(M::AbstractMatrix, P::LinearMaps.TransposeMap{<:Any, <:ProjMat}) = (P.lmap * M')'
# const PT{d,q,p} = LinearMaps.TransposeMap{Bool, ProjMat{d,q,p}}
function mul!(
    out::AbstractVector,
    P::LinearMaps.TransposeMap{<:Any, ProjMat{d,q,p}},
    v::AbstractVector,
    a::Number,
    b::Number) where {d,q,p}
    out .*= a
    out[p+1:q+1:d*(q+1)] .+= b .* v
    out
end
function mul!(
    out::AbstractVector,
    P::LinearMaps.TransposeMap{<:Any, ProjMat{d,q,p}},
    v::AbstractVector,
    ) where {d,q,p}
    out .*= 0
    out[p+1:q+1:d*(q+1)] .= v
    out
end




"""
Special operator to handle the H resulting from ODEs

H = E1 - J * E0
"""
struct ODEHMat{T,E0T,E1T,JT} <: LinearMaps.LinearMap{T}
    E0::E0T
    E1::E1T
    J::JT
end
ODEHMat(E0, E1, J::AbstractMatrix{T}) where {T} =
    ODEHMat{T,typeof(E0),typeof(E1),typeof(J)}(E0,E1,J)
ODEHMat(E0, E1, J::T) where {T<:Number} =
    ODEHMat{T,typeof(E0),typeof(E1),typeof(J)}(E0,E1,J)
Base.size(H::ODEHMat) = size(H.E0)
*(H::ODEHMat, v::AbstractVector) = (H.E1 * v) .- H.J * (H.E0 * v)
*(H::ODEHMat, M::AbstractMatrix) = (H.E1 * M) .- H.J * (H.E0 * M)
LinearMaps.MulStyle(H::ODEHMat) = FiveArg()
mul!(out::AbstractVector, H::ODEHMat, v::AbstractVector) =
    (matmul!(out, H.J, H.E0*v, -1, 0); out .+= (H.E1*v); out)
mul!(out::AbstractVector, H::ODEHMat, v::AbstractVector, a::Number, b::Number) =
    (matmul!(out, H.J, H.E0*v, -a, b); out .+= a .* (H.E1*v); out)
mul!(out::AbstractMatrix, H::ODEHMat, M::AbstractMatrix) = begin
    matmul!(out, H.J, H.E0*M, -1, 0); out .+= (H.E1*M); out
end
mul!(out::AbstractMatrix, H::ODEHMat, M::AbstractMatrix, a::Number, b::Number) =
    (matmul!(out, H.J, H.E0*M, -a, b); out .+= a .* (H.E1*M); out)
*(M::AbstractMatrix, HT::LinearMaps.TransposeMap{<:Any, <:ODEHMat}) = (HT.lmap * M')'
# Works great with J::Matrix, but also J=0!
# mul!(out::AbstractMatrix, M::AbstractMatrix, HT::LinearMaps.TransposeMap{<:Any, <:ODEHMat}) = mul!(out', HT.lmap, M')
