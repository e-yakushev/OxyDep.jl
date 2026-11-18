# The Aniva Bay setup: the `bathymetry_aniva.nc` grid (prepare_bathymetry_aniva.jl), Copernicus
# GLOBAL_ANALYSISFORECAST_PHY daily means for 2023 along the open southern
# edge, and the JRA55-do repeat year as the atmosphere (jra55_atmosphere.jl). It runs the calendar
# year 2023. River observations from 2024 are an explicit month/day seasonal proxy for 2023,
# with 29 February omitted; re-run add_rivers with this setup to prepare that time axis.
#
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim prepare_bathymetry     --config aniva.jl   # checks the file
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim download_boundary_data --config aniva.jl   # combines local files
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim prepare_boundary_data  --config aniva.jl
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim prepare_atmosphere     --config aniva.jl
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim run_simulation         --config aniva.jl
#
# Nothing is downloaded: the Copernicus files are already in `~/FjordSim_data/aniva/copernicus_1day/`
# and the JRA55 files in `JRA55/`.
# No prepared initial-state file is required. Start uniformly, or warm-start from the last ocean
# snapshot of an archived run; either way the new run replays the 2023 forcing cycle on the GPU.

using FjordSim
using FjordSim: Bathymetry, BoundaryData, Grids, InitialState
using FjordSim.Configs: AbstractBathymetryConfig, AbstractBoundaryDataConfig, AbstractInitialStateConfig
using FjordSim.Regridding: ProjectedSourceGrid, SourceRecord
using Oceananigans:
    CATKEVerticalDiffusivity,
    HydrostaticSphericalCoriolis,
    SeawaterBuoyancy,
    WENO,
    WENOVectorInvariant
using Oceananigans.TurbulenceClosures: HorizontalScalarBiharmonicDiffusivity
using Oceananigans.Units: days, hour, hours, minutes, second
using NCDatasets
using Dates: Date, DateTime, Day
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using NumericalEarth: FreezingLimitedOceanTemperature

include(joinpath(@__DIR__, "jra55_atmosphere.jl"))
if !isdefined(@__MODULE__, :AnivaCoastalSources)
    include(joinpath(@__DIR__, "prepare_coastal_sources_aniva.jl"))
end

# --- Copernicus files ---------------------------------------------------------------------------

# Copernicus is on a regular longitude/latitude grid, which a `ProjectedSourceGrid` in degrees
# describes, so FjordSim's regridding reads it unchanged.
const COPERNICUS_PROJ4 = "+proj=longlat +datum=WGS84 +no_defs"

"""
    copernicus_source_grid(filepath)

The `ProjectedSourceGrid` of a Copernicus file on `longitude`, `latitude` and `depth` axes, checked
to be regular in the horizontal.
"""
function copernicus_source_grid(filepath)
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

"""
    copernicus_records(filepath)

One `SourceRecord` per time index of `filepath`.
"""
function copernicus_records(filepath)
    isfile(filepath) || error("Copernicus file $filepath does not exist.")
    return NCDataset(filepath) do ds
        [SourceRecord(DateTime(date), filepath, index) for (index, date) in enumerate(ds["time"][:])]
    end
end

# --- Bathymetry ---------------------------------------------------------------------------------

"""
    AnivaBathymetryConfig

A bathymetry file prepared outside FjordSim by `prepare_bathymetry_aniva.jl`, in the
`lon`/`lat`/`h`/`z_faces` layout `ImmersedBoundaryGrid` reads. `prepare_bathymetry` only checks that
it matches the setup's `EvenGrid`; it does not write it.
"""
Base.@kwdef mutable struct AnivaBathymetryConfig <: AbstractBathymetryConfig
    data_root::String
    output_file::String
    plot_file::String
end

function Bathymetry.process_bathymetry(grid_config, config::AnivaBathymetryConfig, boundary_data_config)
    filepath = Bathymetry.resolve_bathymetry_file(config)
    Nx, Ny, Nz = grid_config.size
    centers(range, n) = [range[1] + (i - 1 // 2) * (range[2] - range[1]) / n for i in 1:n]

    NCDataset(filepath) do ds
        size(ds["h"]) == (Nx, Ny) || error("$filepath is $(size(ds["h"])) cells, the grid ($Nx, $Ny).")
        ds["z_faces"][:] ≈ grid_config.z_faces || error("$filepath has other z_faces than the grid.")
        ds["lon"][:] ≈ centers(grid_config.longitude, Nx) && ds["lat"][:] ≈ centers(grid_config.latitude, Ny) ||
            error("$filepath covers another box than the grid's $(grid_config.longitude), $(grid_config.latitude).")
        wet = coalesce.(ds["h"][:, :], 0.0) .< 0
        for edge in Grids.open_edges(boundary_data_config)
            row = edge === :south ? wet[:, 1] : edge === :north ? wet[:, end] : edge === :west ? wet[1, :] : wet[end, :]
            any(row) || @warn "The open $edge edge of $filepath has no water."
        end
        @info "$filepath matches the grid: $Nx x $Ny x $Nz, $(count(wet)) wet columns"
    end

    return (; output_file = filepath)
end

# --- Initial state ------------------------------------------------------------------------------

"""
    CopernicusInitialStateConfig(; data_root, output_directory, source_file, output_file, variable_names, parameters)

The initial state from one daily record of the Copernicus file `source_file` in `output_directory`:
the record of the day `start_date` falls in. `variable_names` maps each source variable to the state
name it becomes; only those in `parameters` are prepared. Nothing is downloaded.
"""
Base.@kwdef mutable struct CopernicusInitialStateConfig <: AbstractInitialStateConfig
    data_root::String
    output_directory::String
    source_file::String
    output_file::String
    variable_names::Dict{String,String}
    parameters::Vector{String}
end

InitialState.initial_state_variable_names(config::CopernicusInitialStateConfig) = config.variable_names

InitialState.initial_state_source_grid(config::CopernicusInitialStateConfig, filepath) =
    copernicus_source_grid(filepath)

copernicus_initial_state_file(config) = joinpath(InitialState.initial_state_directory(config), config.source_file)

function InitialState.initial_state_record(config::CopernicusInitialStateConfig, date)
    records = copernicus_records(copernicus_initial_state_file(config))
    index = findfirst(record -> Date(record.date) == Date(date), records)
    isnothing(index) && error(
        "$(config.source_file) has no record on $(Date(date)); it covers " *
        "$(first(records).date) to $(last(records).date).",
    )
    return records[index]
end

function InitialState.fetch_initial_state(target_grid, config::CopernicusInitialStateConfig, date)
    InitialState.initial_state_record(config, date)
    @info "Initial state for $(Date(date)) is in $(copernicus_initial_state_file(config)); nothing to download"
    return copernicus_initial_state_file(config)
end

# --- Open southern boundary ---------------------------------------------------------------------

const COPERNICUS_BOUNDARY_VARIABLES = Dict(
    "thetao" => "T",
    "so" => "S",
    "uo" => "u",
    "vo" => "v",
    "zos" => "eta",
    "ubar" => "ubar",
    "vbar" => "vbar",
)

"""
    CopernicusBoundaryDataConfig(; data_root, output_directory, open_edges, kw...)

The exterior state along the open edges from the Copernicus daily means in `output_directory`:
`source_file` (thetao, so, uo, vo) and `sea_level_file` (zos). `download_boundary_data` combines
them into `combined_file`, adding the depth-mean currents `ubar` and `vbar` the Flather condition
needs, and `prepare_boundary_data` regrids that onto the open edges. The daily records are
interpolated linearly to the hourly axis the boundary file is written on.
"""
mutable struct CopernicusBoundaryDataConfig <: AbstractBoundaryDataConfig
    data_root::String
    output_directory::String
    source_file::String
    sea_level_file::String
    combined_file::String
    output_file::String
    plot_file::String
    open_edges::Vector{Symbol}
    architecture::Symbol
    parameters::Vector{String}
    start_date::DateTime       
    stop_date::DateTime
end

function CopernicusBoundaryDataConfig(;
    data_root,
    output_directory,
    source_file = "copernicus_2023.nc",
    sea_level_file = "copernicus_zos.nc",
    combined_file = "copernicus_boundary_2023.nc",
    output_file = "boundaries_aniva.nc",
    plot_file = "boundaries_aniva.png",
    open_edges,
    architecture = :auto,
    parameters = collect(keys(COPERNICUS_BOUNDARY_VARIABLES)),
    start_date = DateTime(2023, 1, 1),   
    stop_date  = DateTime(2024, 1, 1),
)
    edges = Grids.lateral_edges(open_edges)
    isempty(edges) && throw(ArgumentError("CopernicusBoundaryDataConfig names no `open_edges`."))
    return CopernicusBoundaryDataConfig(
        data_root, output_directory, source_file, sea_level_file, combined_file, output_file,
        plot_file, edges, architecture, parameters, start_date, stop_date, 
    )
end

copernicus_boundary_file(config) = joinpath(BoundaryData.boundary_data_directory(config), config.combined_file)

BoundaryData.boundary_variable_names(::CopernicusBoundaryDataConfig) = COPERNICUS_BOUNDARY_VARIABLES

BoundaryData.boundary_source_grid(::CopernicusBoundaryDataConfig, filepath) = copernicus_source_grid(filepath)

function BoundaryData.boundary_time_steps(config::CopernicusBoundaryDataConfig)
    filepath = copernicus_boundary_file(config)
    isfile(filepath) || error("$filepath does not exist; run `download_boundary_data` for this setup first.")
    return copernicus_records(filepath)
end

"""
    copernicus_layer_thicknesses(depths)

The thickness of each Copernicus level, its faces halfway between the level centres `depths`, the
top one at the surface and the bottom one as far below the deepest centre as the face above it.
"""
function copernicus_layer_thicknesses(depths)
    faces = [0.0; (depths[1:end-1] .+ depths[2:end]) ./ 2; depths[end] + (depths[end] - depths[end-1]) / 2]
    return diff(faces)
end

"""
    depth_mean(values, thicknesses)

The thickness-weighted mean over the finite levels of each `(longitude, latitude)` column of
`values`, `NaN` for a column with none.
"""
function depth_mean(values, thicknesses)
    mean = fill(NaN32, size(values, 1), size(values, 2))
    for j in axes(values, 2), i in axes(values, 1)
        total = 0.0
        weight = 0.0
        for k in axes(values, 3)
            value = values[i, j, k]
            isfinite(value) || continue
            total += value * thicknesses[k]
            weight += thicknesses[k]
        end
        weight > 0 && (mean[i, j] = total / weight)
    end
    return mean
end

"""
    fetch_boundary_data(target_grid, config::CopernicusBoundaryDataConfig)

Combine `source_file` (thetao, so, uo, vo) and `sea_level_file` (zos) into `combined_file`, adding
`ubar` and `vbar`, the depth means of `uo` and `vo` over each column's valid levels, and repeating the
last daily record one day later. Nothing is downloaded; an existing combined file is rebuilt.
"""
function BoundaryData.fetch_boundary_data(target_grid, config::CopernicusBoundaryDataConfig)
    directory = BoundaryData.boundary_data_directory(config)
    source_path = joinpath(directory, config.source_file)
    sea_level_path = joinpath(directory, config.sea_level_file)
    output_path = copernicus_boundary_file(config)
    temporary_path = output_path * ".tmp"
    rm(temporary_path; force = true)
    @info "Combining $(config.source_file) and $(config.sea_level_file) into $(config.combined_file)"

    NCDataset(source_path) do source
        NCDataset(sea_level_path) do sea_level
            longitude = source["longitude"][:]
            latitude = source["latitude"][:]
            depth = Float64.(source["depth"][:])
            times = source["time"][:]
            sea_level["longitude"][:] ≈ longitude && sea_level["latitude"][:] ≈ latitude &&
                sea_level["time"][:] == times ||
                error("$(config.sea_level_file) is on another grid or time axis than $(config.source_file)")
            thicknesses = copernicus_layer_thicknesses(depth)
            # The daily means end on 31 December, a day before the run does, and the boundary axis is
            # hourly, so it can be padded by one hour only: the last day is repeated a day later.
            output_times = [times; last(times) + Day(1)]
            @info "Repeating the $(last(times)) record at $(last(output_times)) to reach the end of the run"

            NCDataset(temporary_path, "c") do output
                for (name, values) in (("longitude", longitude), ("latitude", latitude), ("depth", depth), ("time", output_times))
                    defDim(output, name, length(values))
                    defVar(output, name, values, (name,))
                end
                output.attrib["source"] = "$(config.source_file), $(config.sea_level_file)"
                output.attrib["history"] = "Record $(last(output_times)) repeats $(last(times))"
                full_depth = ("longitude", "latitude", "depth", "time")
                surface = ("longitude", "latitude", "time")
                variable(name, dimensions) = defVar(output, name, Float32, dimensions;
                    chunksizes = [length(longitude), length(latitude), (length(dimensions) == 4 ? (length(depth),) : ())..., 1],
                    deflatelevel = 5, attrib = ["_FillValue" => NaN32])
                source_index = [eachindex(times); lastindex(times)]

                for name in ("thetao", "so", "uo", "vo")
                    variable(name, full_depth)[:, :, :, :] = Float32.(coalesce.(source[name][:, :, :, source_index], NaN32))
                end
                variable("zos", surface)[:, :, :] = Float32.(coalesce.(sea_level["zos"][:, :, source_index], NaN32))
                ubar = variable("ubar", surface)
                vbar = variable("vbar", surface)
                for (index, record) in enumerate(source_index)
                    ubar[:, :, index] = depth_mean(Float32.(coalesce.(source["uo"][:, :, :, record], NaN32)), thicknesses)
                    vbar[:, :, index] = depth_mean(Float32.(coalesce.(source["vo"][:, :, :, record], NaN32)), thicknesses)
                end
            end
        end
    end

    mv(temporary_path, output_path; force = true)
    @info "Boundary source saved to $output_path"
    return output_path
end

# --- The setup ----------------------------------------------------------------------------------

ANIVA_INITIAL_CONDITIONS_SOURCE = :uniform
ANIVA_INITIAL_RESULTS_DIRECTORY = fjord_results_root("aniva_init")
const ANIVA_UNIFORM_INITIAL_CONDITIONS = (T = 5f0, S = 33f0, u = 0f0, v = 0f0)

"""
    aniva_initial_conditions_for_run(source, uniform; results_directory)

Choose a uniform state or the last record of the newest snapshots_ocean_*.nc in an archived
results directory. A results warm start resets the clock and does not restore the free surface
or CATKE turbulence; checkpoint pickup is the separate exact-continuation mechanism.
"""
function aniva_initial_conditions_for_run(source::Symbol, uniform; results_directory)
    source === :uniform && return uniform
    source === :results || throw(ArgumentError("initial_conditions_source must be :uniform or :results"))
    isdir(results_directory) || error("No archived results directory $results_directory to start from")
    snapshots = sort(filter(
        name -> startswith(name, "snapshots_ocean_") && endswith(name, ".nc") &&
                isfile(joinpath(results_directory, name)),
        readdir(results_directory),
    ))
    isempty(snapshots) && error("No snapshots_ocean_*.nc in $results_directory to start from")
    return FromResults(abspath(joinpath(results_directory, last(snapshots))))
end

"""
    aniva(; architecture = :gpu, initial_conditions_source, initial_results_directory)

Aniva Bay hydrophysics for the calendar year 2023 on a 210 x 160 x 17 grid of ~500 m cells covering
142.08-143.45°E, 46.03-46.75°N, open to La Perouse Strait along its southern edge.
"""
function aniva(;
    architecture = :gpu,
    river_config = AnivaCoastalSources.AnivaRiversConfig(target_year = 2023),
    initial_conditions_source = ANIVA_INITIAL_CONDITIONS_SOURCE,
    initial_results_directory = ANIVA_INITIAL_RESULTS_DIRECTORY,
)
    data_root = fjord_data_root("aniva")
    FT = Float32

    return FjordConfig(
        # Must match `bathymetry_aniva.nc`, which `prepare_bathymetry` checks: Δx is 503 m on the
        # southern edge and 496 m on the northern, Δy 500 m.
        grid_config = EvenGrid(
            size = (210, 160, 17),
            halo = (7, 7, 7),
            longitude = (142.08, 143.45),
            latitude = (46.03, 46.75),
            z_faces = [
                -200.0, -180.0, -160.0, -140.0, -120.0, -100.0, -80.0, -60.0,
                -40.0, -24.0, -15.0, -10.0, -7.5, -5.0, -3.0, -2.0, -1.0, 0.0,
            ],
            float_type = FT,
        ),
        bathymetry_config = AnivaBathymetryConfig(
            data_root = data_root,
            output_file = "bathymetry_aniva.nc",
            plot_file = "bathymetry_aniva.png",
        ),
        boundary_data_config = CopernicusBoundaryDataConfig(
            data_root = data_root,
            output_directory = "copernicus_1day",
            open_edges = :south,
            start_date = DateTime(2023, 1, 1), 
            stop_date  = DateTime(2024, 1, 1),   
        ),
        river_config = river_config,
        atmosphere_config = JRA55RYFConfig(data_root = data_root),
        # The mixing and advection of FjordSim's `examples/oslofjorden.jl`, with its
        # grid-spacing-dependent coefficients scaled from that setup's 193 m cell to this one's 500 m
        # (`docs/setups.md`).
        model_config = CoupledHydrostaticModel(
            buoyancy = SeawaterBuoyancy(FT, equation_of_state = TEOS10EquationOfState(FT)),
            # A harmonic viscosity and diffusivity ramping to zero over 16 cells (8 km) inward from
            # the open southern edge, damping what the open boundary radiates. It is explicit, so its
            # 2D limit under Oceananigans' Adams-Bashforth stepping is λΔt ≤ 0.91 with
            # λ = ν(4/Δx² + 4/Δy²): Δt ≤ 0.91·Δx²/8ν, 280 s at 100 m² s⁻¹ on the narrowest 496 m
            # cell, clear of the 180 s `max_time_step`. At 200 (Oslofjord's 30 scaled by
            # (500/193)²) the limit is 140 s, and the boundary rows blew up to 5.7 m/s within an hour
            # of the adaptive step reaching 180 s, at day 2.1.
            closure = BoundarySponge(
                base = (
                    # CATKE takes `w★ = sqrt(max(minimum_tke, e))`, so this floor, not the
                    # prognostic TKE, sets κ = 0.098·e_min/N over most of a stratified column. 5e-6
                    # and the 1e-3 m² s⁻¹ cap on the tracer diffusivity are this setup's own.
                    CATKEVerticalDiffusivity(FT; minimum_tke = 5e-6, maximum_tracer_diffusivity = 1e-3),
                    # Oslofjord's 2e4 m⁴ s⁻¹ scaled by (500/193)⁴ = 45: a 70 min e-folding of the
                    # 2Δx mode (Δx⁴/16ν₄) and an explicit limit Δt ≤ Δx⁴/32ν₄ of 2100 s that
                    # `AdaptiveTimeStep` never measures. κ = 0: on Oslofjord biharmonic tracer
                    # diffusion overshot at fronts.
                    HorizontalScalarBiharmonicDiffusivity(FT; ν = 9e5, κ = 0),
                ),
                width_cells = 16,
                viscosity = 100.0,
                diffusivity = 50.0,
            ),
            # Salinity's WENO is bounded to (0, 40), with horizontal upwinding within two cells
            # of the river columns to avoid high-order overshoots across the freshwater fronts.
            # A tracer this NamedTuple omits falls back to Oceananigans' unbounded `Centered()`, so
            # it names CATKE's `e`, and a setup adding tracers must name those too.
            tracer_advection = (
                T = isnothing(river_config) ? WENO(FT) :
                    RiverMouthAdvection(interior = WENO(FT), width_cells = 2, narrow_cells = true),
                S = isnothing(river_config) ? WENO(FT; bounds = (0, 40)) :
                    RiverMouthAdvection(interior = WENO(FT; bounds = (0, 40)), width_cells = 2, narrow_cells = true),
                e = WENO(FT),
            ),
            momentum_advection = WENOVectorInvariant(FT),
            tracers = (:T, :S),
            coriolis = HydrostaticSphericalCoriolis(FT),
            sea_ice = FreezingLimitedOceanTemperature(FT),
            free_surface = SplitExplicitFreeSurfaceConfig(cfl = 0.7),
            extra_kwargs = (;),
        ),
        # Tuple order is merge precedence. The open-boundary timescales are Marchesiello et al.
        # (2001): nudge hard towards the data on inflow, let radiation do the work on outflow.
        boundary_condition_configs = (
            AirSeaFluxes(),
            QuadraticBottomDrag(coefficient = 0.003),
            OpenLateralBoundaryFromData(
                inflow_timescale = 3hours,
                outflow_timescale = 360days,
            ),
        ),
        run_config = RunConfig(
            results_root = fjord_results_root("aniva"),
            architecture = architecture,
            writers = (
                SnapshotWriter(
                    name = :ocean,
                    output_file = "snapshots_ocean.nc",
                    variables = (:T, :S, :u, :v, :e),
                    interval = 3hours,
                    split_times = Float64[],
                    overwrite_existing = true,
                ),
                FieldSnapshotWriter(
                    name = :surface,
                    output_file = "snapshots_surface.jld2",
                    variables = (:η,),
                    interval = 3hours,
                    overwrite_existing = true,
                ),
                CheckpointWriter(interval = 12hours, cleanup = true),
            ),
            callbacks = (ProgressCallback(name = :progress, interval = 24hours, report = progress),),
            time_stepping = AdaptiveTimeStep(
                initial_time_step = 1second,
                cfl = 0.3,
                max_time_step = 60second,
                max_time_step_change = 1.01,
            ),
            initial_conditions = aniva_initial_conditions_for_run(
                initial_conditions_source,
                ANIVA_UNIFORM_INITIAL_CONDITIONS;
                results_directory = initial_results_directory,
            ),
            # The JRA55 repeat year is laid onto this calendar by `jra55_atmosphere.jl`, whose
            # START_DATE and STOP_DATE must match.
            start_date = DateTime(2023, 1, 1),
            stop_date = DateTime(2024, 1, 1),
            loops = 1,
            pickup = false,
        ),
    )
end

aniva()
