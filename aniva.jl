using FjordSim
using FjordSim.Forcing: ProjectedSourceGrid, SourceRecord
using Oceananigans
using Oceananigans.Units: days, hours, minutes, second
using NCDatasets
using Dates: DateTime
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using NumericalEarth: FreezingLimitedOceanTemperature

include(joinpath(@__DIR__, "jra55_atmosphere.jl"))

# Copernicus variable => FjordSim forcing name; the currents come from the T-points as is.
const COPERNICUS_VARIABLES = Dict("thetao" => "T", "so" => "S", "uo" => "u", "vo" => "v")
const COPERNICUS_PROJ4 = "+proj=longlat +datum=WGS84 +no_defs"

# Variable => the download_copernicus.jl file holding it.
const COPERNICUS_SOURCE_FILES = Dict(
    "thetao" => "copernicus_thetao.nc",
    "so" => "copernicus_so.nc",
    "uo" => "copernicus_cur.nc",
    "vo" => "copernicus_cur.nc",
)

"""
    CopernicusConfig

The combined Copernicus file as a forcing source on a regular lon/lat grid, which a
`ProjectedSourceGrid` in degrees describes, so the generic `prepare_forcing` regrids it unchanged.

`combine_copernicus` builds `combined_file` from the files `download_copernicus.jl` leaves in
`output_directory`, and has to run before `prepare_forcing`.
"""
Base.@kwdef mutable struct CopernicusConfig <: AbstractForcingConfig
    data_root::String
    output_directory::String
    output_file::String
    plot_file::String
    combined_file::String
    architecture::Symbol = :auto
    parameters::Vector{String} = collect(keys(COPERNICUS_VARIABLES))
    rivers::Nothing = nothing
end

FjordSim.forcing_variable_names(::CopernicusConfig) = COPERNICUS_VARIABLES

function FjordSim.forcing_time_steps(config::CopernicusConfig)
    filepath = joinpath(forcing_directory(config), config.combined_file)
    isfile(filepath) || error("$filepath does not exist; run `combine_copernicus` to build it.")
    return NCDataset(filepath) do ds
        return [SourceRecord(DateTime(date), filepath, index) for (index, date) in enumerate(ds["time"][:])]
    end
end

function FjordSim.forcing_source_grid(::CopernicusConfig, filepath)
    return NCDataset(filepath) do ds
        longitude = Array{Float64}(ds["longitude"][:])
        latitude = Array{Float64}(ds["latitude"][:])
        depths = Array{Float64}(ds["depth"][:])
        regular(axis) = all(difference -> isapprox(difference, axis[2] - axis[1]; rtol = 1e-3), diff(axis))
        length(longitude) >= 2 && length(latitude) >= 2 && regular(longitude) && regular(latitude) ||
            error("Copernicus longitude/latitude in $filepath are not a regular grid of at least 2x2.")
        return ProjectedSourceGrid(longitude, latitude, depths, COPERNICUS_PROJ4)
    end
end

# The downloads share one grid and time axis; `prepare_forcing` reads one file per time record.
function combine_copernicus(config::CopernicusConfig)
    directory = forcing_directory(config)
    output_path = joinpath(directory, config.combined_file)
    rm(output_path; force = true)

    NCDataset(output_path, "c") do output
        coordinates = nothing
        for (name, filename) in sort!(collect(COPERNICUS_SOURCE_FILES))
            NCDataset(joinpath(directory, filename)) do ds
                current = (ds["longitude"][:], ds["latitude"][:], ds["depth"][:], ds["time"][:])
                if isnothing(coordinates)
                    coordinates = current
                    for (dimension, values) in zip(("longitude", "latitude", "depth", "time"), current)
                        defDim(output, dimension, length(values))
                        defVar(output, dimension, values, (dimension,))
                    end
                end
                current == coordinates || error("$filename is on a different grid or time axis than the other downloads")

                variable = defVar(output, name, Float32, ("longitude", "latitude", "depth", "time");
                    deflatelevel = 5, attrib = ["_FillValue" => NaN32])
                variable[:, :, :, :] = coalesce.(ds[name][:, :, :, :], NaN32)
            end
        end
    end
    return output_path
end

"""
    AnivaBathymetryConfig

A bathymetry file prepared outside FjordSim, in the `lon`/`lat`/`h`/`z_faces` layout
`ImmersedBoundaryGrid` reads. `prepare_bathymetry_aniva.jl` derives it from `aniva_bathymetry.nc`.
There is no `prepare_bathymetry` for it.
"""
Base.@kwdef mutable struct AnivaBathymetryConfig <: AbstractBathymetryConfig
    data_root::String
    output_file::String
    plot_file::String
end

"""
    aniva()

The Aniva Bay setup: a 210 x 216 x 13 grid of 500 m cells covering 142.07-143.44°E, 45.80-46.77°N
from a bathymetry file prepared outside FjordSim, Copernicus GLOBAL_ANALYSISFORECAST daily forcing,
and closed lateral boundaries. It runs the calendar year 2023.

The atmosphere is the JRA55-do repeat year (1990-91) laid onto the 2023 calendar, so the surface
forcing is climatological rather than 2023 weather.
"""
function aniva()
    data_root = fjord_data_root("aniva")
    FT = Float32

    return FjordConfig(
        grid_config = EvenGrid(
            size = (210, 216, 13),
            halo = (7, 7, 7),
            longitude = (142.06936, 143.43577),
            latitude = (45.79955, 46.77082),
            z_faces = [-120.0, -100.0, -80.0, -60.0, -40.0, -24.0, -15.0, -10.0, -7.5, -5.0, -3.0, -2.0, -1.0, 0.0],
            float_type = FT,
        ),
        bathymetry_config = AnivaBathymetryConfig(
            data_root = data_root,
            output_file = "bathymetry.nc",
            plot_file = "bathymetry.png",
        ),
        forcing_config = CopernicusConfig(
            data_root = data_root,
            output_directory = "copernicus",
            output_file = "forcing_aniva.nc",
            plot_file = "forcing_aniva.png",
            combined_file = "copernicus_2023.nc",
        ),
        boundary_config = nothing,
        atmosphere_config = JRA55RYFConfig(data_root = data_root),
        simulation_config = SimulationConfig(
            results_root = fjord_results_root("aniva"),
            architecture = :auto,
            model = CoupledHydrostaticSimulation(
                buoyancy = SeawaterBuoyancy(FT, equation_of_state = TEOS10EquationOfState(FT)),
                closure = (
                    CATKEVerticalDiffusivity(FT; minimum_tke = 5e-6, maximum_tracer_diffusivity = 1e-3),
                    HorizontalScalarDiffusivity(FT; ν = 2.0, κ = 10.0),
                ),
                tracer_advection = WENO(FT),
                momentum_advection = WENOVectorInvariant(FT),
                tracers = (:T, :S),
                coriolis = HydrostaticSphericalCoriolis(FT),
                sea_ice = FreezingLimitedOceanTemperature(FT),
                biogeochemistry = nothing,
                free_surface = SplitExplicitFreeSurfaceConfig(cfl = 0.7),
                extra_kwargs = (;),
            ),
            boundary_conditions = MergedBoundaryConditions(
                AirSeaFluxes(),
                QuadraticBottomDrag(coefficient = 0.003),
            ),
            writers = (
                SnapshotWriter(
                    name = :ocean,
                    output_file = "snapshots_ocean.nc",
                    variables = (:T, :S, :u, :v),
                    interval = 3hours,
                    overwrite_existing = true,
                ),
                CheckpointWriter(interval = 12hours, cleanup = true),
            ),
            callbacks = (ProgressCallback(name = :progress, interval = 24hours, report = progress),),
            time_stepping = AdaptiveTimeStep(
                initial_time_step = 1second,
                cfl = 0.3,
                max_time_step = 3minutes,
                max_time_step_change = 1.01,
            ),
            initial_conditions = FromForcing(),
            start_date = DateTime(2023, 1, 1),
            stop_time = 365days,
            loops = 1,
            pickup = false,
        ),
    )
end

aniva()
