using Burgers
using Test
using BSON: @save

@testset "Burgers" begin
    xs, ys = Burgers.get_data(n=1000)

    @test size(xs) == (2, 1024, 1000)
    @test size(ys) == (1, 1024, 1000)

    learner = Burgers.train(epochs=1) # usually 10 epochs
    #loss = learner.cbstate.metricsepoch[ValidationPhase()][:Loss].values[end]
    #@test loss < 0.1

    #@save "Burgers-learner.bson" learner

    # include("deeponet.jl")
end
