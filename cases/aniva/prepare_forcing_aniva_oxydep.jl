# Forcing for the Aniva region from Copernicus Marine, in the format `prepare_forcing` writes,
# with the OxyDep tracers from <aniva data root>/Sea_boundary added.
#
#   julia --project=/home/eya/src/FjordSim.jl prepare_forcing_aniva_oxydep.jl
#
# Run download_copernicus.jl first. This reads aniva_bathymetry.nc and the files in copernicus/, and
# writes forcing_aniva_oxydep.nc next to the bathymetry.

using FjordSim
using Oceananigans: CPU

# The Aniva grid, forcing source and OxyDep sea-boundary tracers.
module AnivaOxyDep
include(joinpath(@__DIR__, "aniva_oxydep.jl"))
end

function prepare_forcing_aniva_oxydep()
    config = AnivaOxyDep.oxydep_config
    forcing = config.forcing_config
    AnivaOxyDep.combine_copernicus(forcing.base)

    grid = ImmersedBoundaryGrid(bathymetry_path(config.bathymetry_config), CPU(), (7, 7, 7), Float32)
    result = prepare_forcing(grid, forcing; coverage = coverage_window(config.simulation_config))
    @info "Forcing file saved to $(result.output_file)"
    return result.output_file
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    prepare_forcing_aniva_oxydep()
end
