using Test, Dates, TOML
include(joinpath(@__DIR__,"../code/src/paths.jl"))
@testset "Timestamped targets and immutable run snapshots" begin
    mktempdir() do root
        @test_throws ErrorException smm_targets_file(;root)
        runs=joinpath(root,"output","smm_runs")
        old=joinpath(runs,"2026-09-06_183119")
        new=joinpath(runs,"2026-09-07_114138")
        mkpath(old);mkpath(new)
        write(joinpath(old,"targets.toml"),"value = 1\n")
        write(joinpath(new,"targets.toml"),"value = 2\n")
        symlink(old,joinpath(runs,"latest"))
        @test smm_targets_file(;root)==joinpath(new,"targets.toml")
        @test smm_targets_file(;at=joinpath(old,"baseline.toml"),root)==joinpath(old,"targets.toml")
        @test smm_targets_file(joinpath(old,"targets.toml");at=joinpath(new,"candidate.toml"),root)==joinpath(old,"targets.toml")
        @test_throws ErrorException smm_targets_file(joinpath(root,"missing.toml");root)
        dest=joinpath(runs,"2026-09-08_120000")
        saved=freeze_smm_targets(dest;root)
        @test read(saved,String)=="value = 2\n"
        write(joinpath(new,"targets.toml"),"value = 3\n")
        @test read(saved,String)=="value = 2\n"
        @test freeze_smm_targets(dest;resume=true,root)==saved
        @test_throws ErrorException freeze_smm_targets(dest;source=joinpath(new,"targets.toml"),resume=true,root)
        @test read(saved,String)=="value = 2\n"
        @test_throws ErrorException freeze_smm_targets(dest;at=joinpath(old,"baseline.toml"),root)
        @test_throws ErrorException freeze_smm_targets(joinpath(runs,"missing");resume=true,root)
        @test TOML.parsefile(saved)["value"]==2
    end
end
