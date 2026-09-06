using Test, TOML
include(joinpath(@__DIR__,"../code/smm/finite_differences.jl"))
@testset "Bounded differences in level and log search coordinates" begin
    for (lo,hi) in ((0.0,1.0),(log(0.05),log(20.0)))
        width=hi-lo
        for frac in (0.0,0.001,0.25,0.5,0.999,1.0), step in (0.005,0.01,0.02)
            z=lo+frac*width
            st=bounded_stencil(z,lo,hi,step)
            @test all(lo .<= st.points .<= hi)
            @test sum(st.weights .* st.points) ≈ 1.0 atol=1e-10
            @test sum(st.weights .* (st.points.^2)) ≈ 2z atol=1e-10
            @test abs(sum(st.weights)) < 1e-10
            @test st.scheme == (frac < step ? "forward" : frac > 1-step ? "backward" : "central")
        end
    end
    @test_throws ArgumentError bounded_stencil(-0.1,0.0,1.0,0.01)
    @test_throws ArgumentError bounded_stencil(0.5,0.0,1.0,0.0)
    @test_throws ArgumentError bounded_stencil(0.5,0.0,1.0,NaN)
    @test_throws ArgumentError bounded_stencil(0.5,0.0,1.0,1e-320)
end
module Fixture
    using Main: bounded_stencil
    const COLUMNS=[(name=:x,lo=0.0,hi=1.0,link=:level)]
    const THETA0=Dict(:x=>1.0)
    const NP_=1; const NM_=1; const GRID=30; const SIM_N=2000; const SEED=1234
    to_s(v,c)=v; from_s(v,c)=v
    pmap(f,jobs)=map(f,jobs)
    const INVALID=Ref(false)
    residuals_at(vals;kwargs...)=(r=[vals[:x]^2],nviol=INVALID[] ? 1 : 0,nbad=0)
    source=joinpath(@__DIR__,"../code/smm/jacobian.jl")
    for ex in Meta.parseall(read(source,String)).args
        if ex isa Expr && ex.head==:function && ex.args[1].args[1]==:jacobian_at
            Core.eval(@__MODULE__,ex)
        end
    end
end
@testset "Jacobian integration rejects invalid samples and records its stencil" begin
    J,nbad,stencils,nev=Fixture.jacobian_at(0.01)
    @test J[1,1] ≈ 2.0 atol=1e-10
    @test nbad==0 && nev==3
    @test stencils[1].scheme=="backward"
    io=IOBuffer()
    TOML.print(io,Dict("schemes"=>[s.scheme for s in stencils],"points"=>[s.points for s in stencils],"weights"=>[s.weights for s in stencils]))
    raw=TOML.parse(String(take!(io)))
    @test raw["points"][1]==stencils[1].points
    Fixture.INVALID[]=true
    @test_throws ErrorException Fixture.jacobian_at(0.01)
end
