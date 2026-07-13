"""
ExplicitImports.jl checks, following the convention that SciML is rolling out across its
ecosystem: every name is imported explicitly and from its owner module, so that upstream
`using`-to-explicit-import cleanups (which can remove re-exported bindings in patch
releases) cannot silently break this package.

Names listed in `NONPUBLIC_ACCESSES` are known, deliberate uses of non-public API for
which no public counterpart exists (yet). If one of these starts failing, the upstream
package either promoted it (remove it from the list) or removed it (find a replacement).
"""

using ExplicitImports, ProbNumDiffEq, Test

# Deliberate uses of non-public API, kept qualified so they are easy to find:
NONPUBLIC_ACCESSES = (
    # OrdinaryDiffEqDifferentiation Jacobian machinery; no public interface exists.
    :calc_J!, :build_jac_config, :prepare_ADType, :prepare_user_sparsity,
    # OrdinaryDiffEqCore integrator internals used by our save/smooth loop.
    :_postamble!, :_savevalues!, :update_uprev!,
    # SciMLBase remake helpers used in DiffEqBase.remake(::EK1).
    :remaker_of, :struct_as_namedtuple,
    # SciMLBase re-exports this module; used for function wrapper type checks.
    :FunctionWrappersWrappers,
    # ForwardDiff/DiffResults internals; flagged non-public but stable de-facto API.
    :Dual, :value, :derivative!, :jacobian, :jacobian!, :JacobianResult,
    # Base/LinearAlgebra internals used in the fast linear algebra routines.
    :ReshapedArray, :_reshape, :BlasFloat, :geqrt!,
    # Kronecker.jl internals used by IsometricKroneckerProduct.
    :getallfactors, :ldiv_vec_trick!,
    # FiniteHorizonGramians workspace allocation.
    :alloc_mem,
    # TaylorIntegration internal used for Taylor-mode initialization.
    :jetcoeffs!,
)

@test check_no_implicit_imports(ProbNumDiffEq) === nothing
@test check_no_stale_explicit_imports(ProbNumDiffEq) === nothing
@test check_no_self_qualified_accesses(ProbNumDiffEq) === nothing
@test check_all_explicit_imports_via_owners(ProbNumDiffEq) === nothing
# `postamble!` moved from SciMLBase to OrdinaryDiffEqCore between the OrdinaryDiffEqCore
# versions in our compat range, so we cannot access it via a fixed owner.
@test check_all_qualified_accesses_via_owners(
    ProbNumDiffEq;
    ignore=(:postamble!,),
) === nothing

# SciML only started declaring its downstream-facing API `public` in mid-2026
# (SciMLBase#1401, OrdinaryDiffEq#3787). Test-dependency version constraints can still
# resolve older versions without those declarations, so only enforce publicness when
# the resolved versions have them.
SCIML_DECLARES_PUBLIC_API =
    Base.ispublic(ProbNumDiffEq.SciMLBase, :AbstractODEProblem) &&
    Base.ispublic(ProbNumDiffEq.OrdinaryDiffEqCore, :ODEIntegrator) &&
    Base.ispublic(parentmodule(ProbNumDiffEq.solve), :solve)

# ProbNumDiffEq's own internal names accessed by its package extensions (same repo,
# so these can never break unexpectedly; consider declaring them `public` once the
# lower Julia compat bound is 1.11).
OWN_INTERNALS_USED_BY_EXTENSIONS = (
    :AbstractGaussMarkovProcess, :dim, :marginalize, :num_derivatives, :projection,
    :sample,
)

if SCIML_DECLARES_PUBLIC_API
    @test check_all_explicit_imports_are_public(ProbNumDiffEq) === nothing
    @test check_all_qualified_accesses_are_public(
        ProbNumDiffEq;
        ignore=(NONPUBLIC_ACCESSES..., OWN_INTERNALS_USED_BY_EXTENSIONS...),
    ) === nothing
else
    @info "Resolved SciML dependency versions predate their `public` API declarations; " *
          "skipping the publicness checks."
end
