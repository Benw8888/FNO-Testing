using ApproxFun, OrdinaryDiffEq, Sundials
using DiffEqDevTools
using LinearAlgebra
using Plots; gr()

function spectral_output(input_data, viscosity)
    # Uses spectral method to get an output for a Burgers PDE input
    # Viscosity is a constant in the Burgers equation. Should be 0.1

    S = Fourier()
    n = 512
    x = points(S, n)
    D2 = Derivative(S,2)[1:n,1:n]
    D  = (Derivative(S) → S)[1:n,1:n]
    T = ApproxFun.plan_transform(S, n)
    Ti = ApproxFun.plan_itransform(S, n)

    û₀ = input_data
    A = viscosity*D2
    tmp = similar(û₀)
    p = (D,D2,T,Ti,tmp,similar(tmp))
    function burgers_nl(dû,û,p,t)
        D,D2,T,Ti,u,tmp = p
        mul!(tmp, D, û)
        mul!(u, Ti, tmp)
        mul!(tmp, Ti, û)
        @. tmp = tmp*u
        mul!(u, T, tmp)
        @. dû = - u
    end

    prob = SplitODEProblem(DiffEqArrayOperator(Diagonal(A)), burgers_nl, û₀, (0.0,5.0), p)
    sol  = solve(prob, Rodas5(autodiff=false); reltol=1e-12,abstol=1e-12)
    test_sol = TestSolution(sol)

    return sol
end



tslices=[0.0 1.0 2.0 3.0 5.0]
ys=hcat((Ti*sol(t) for t in tslices)...)
labels=["t=$t" for t in tslices]
plot(x,ys,label=labels)