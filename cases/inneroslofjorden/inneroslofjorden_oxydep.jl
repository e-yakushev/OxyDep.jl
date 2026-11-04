"""
The `inneroslofjorden.jl` hydrophysics with OxyDep biogeochemistry (NUT, P, HET, POM, DOM, O₂).

There is no interior forcing (no `forcing_oxydep.nc`). The model reads only:

- `bathymetry.nc` (`prepare_bathymetry`),
- `forcing_rivers_oxydep.nc`, the CSV coastal sources (`add_rivers`, or
  `prepare_coastal_sources_oxydep.jl`),
- `boundaries_oxydep.nc`, NorKyst along the southern edge plus the `bgc_open_boundary/` profiles
  (`prepare_boundary_data`, or `prepare_open_boundary.jl`),
- `atmosphere.nc` (`prepare_atmosphere`),

all under `fjord_data_root("inneroslofjorden")`.

    julia --project=/path/to/FjordSim.jl -m FjordSim run_simulation \\
        --config cases/inneroslofjorden/inneroslofjorden_oxydep.jl

The initial state is set by `INITIAL_CONDITIONS_SOURCE` below: `:uniform` starts from
`inneroslofjorden.jl`'s T and S and the January 1 boundary profiles at 0.5 m for NUT, DOM and O₂;
`:results` starts from the last record of the newest `snapshots_ocean_*.nc` in
`INITIAL_RESULTS_DIRECTORY`. Either way the run starts on `start_date`, 1 January 2020, and
repeats the 2020 boundary, river and atmosphere cycle.
"""

using FjordSim
using FjordSim: Simulations
using FjordSim.Configs: AbstractBiogeochemistryConfig, AbstractModelConfig
using Oceananigans.Units: days, hours
using OceanBioME.Light: TwoBandPhotosyntheticallyActiveRadiation
using OceanBioME: ScaleNegativeTracers
using Oceananigans.BoundaryConditions: FieldBoundaryConditions

include(joinpath(@__DIR__, "inneroslofjorden.jl"))
if !isdefined(@__MODULE__, :OXYDEPModel)
    include(joinpath(@__DIR__, "..", "..", "src", "Oxydep.jl"))
end
if !isdefined(@__MODULE__, :CoastalRiverSources)
    include(joinpath(@__DIR__, "prepare_coastal_sources_oxydep.jl"))
end
if !isdefined(@__MODULE__, :InnerOslofjordOpenBoundary)
    include(joinpath(@__DIR__, "prepare_open_boundary.jl"))
end
using .OXYDEPModel: OXYDEP, bgh_oxydep_boundary_conditions, oxydep_sediment_forcings
using .CoastalRiverSources: CSVRiversConfig

const OXYDEP_TRACERS = (:NUT, :P, :HET, :POM, :DOM, :O₂)

# The initial-conditions switch: `:uniform` for the uniform state, `:results` for the last record of
# a previous run's ocean snapshot in `INITIAL_RESULTS_DIRECTORY`. The new run writes to
# `fjord_results_root("inneroslofjorden_oxydep")` in both cases, so archive what is there first.
INITIAL_CONDITIONS_SOURCE = :results
INITIAL_RESULTS_DIRECTORY = fjord_results_root("inneroslofjorden_oxydep_init")

# `true` resumes the run in `fjord_results_root("inneroslofjorden_oxydep")` from its newest
# checkpoint, superseding the initial conditions above; with no checkpoint there it warns and starts
# fresh. `false` starts fresh and refuses a directory that still holds an earlier run's checkpoints.
PICKUP = true

"""
    OxyDepModel(base, tracers)

`base`, a `CoupledHydrostaticModel`, with OxyDep's sediment forcing added to the forcing and its
surface condition replacing the top boundary condition of each tracer it names.
"""
struct OxyDepModel{M,T} <: AbstractModelConfig
    base::M
    tracers::T
end

struct OxyDepBiogeochemistry{P} <: AbstractBiogeochemistryConfig
    parameter_file::String
    surface_PAR::P
end

Simulations.model_tracers(model::OxyDepModel) = model.tracers

# FjordSim's forcing (the rivers) stays, and OxyDep's sediment term is added to each tracer it names.
function oxydep_forcing(base, bgc)
    sediment = oxydep_sediment_forcings(bgc)
    combined = map(keys(sediment)) do name
        haskey(base, name) ? (base[name], sediment[name]) : sediment[name]
    end
    return merge(base, NamedTuple{keys(sediment)}(combined))
end

with_top_condition(conditions, top) = FieldBoundaryConditions(
    conditions.west, conditions.east, conditions.south, conditions.north,
    conditions.bottom, top, conditions.immersed,
)

# FjordSim's open-edge conditions stay, and OxyDep's surface condition replaces the top.
function oxydep_boundary_conditions(base, bgc, grid)
    oxydep = bgh_oxydep_boundary_conditions(bgc, grid.Nz)
    combined = map(keys(oxydep)) do name
        haskey(base, name) ? with_top_condition(base[name], oxydep[name].top) : oxydep[name]
    end
    return merge(base, NamedTuple{keys(oxydep)}(combined))
end

function Simulations.model_biogeochemistry(config::OxyDepBiogeochemistry, grid)
    light_attenuation = TwoBandPhotosyntheticallyActiveRadiation(grid, config.surface_PAR)
    # Over the six OxyDep tracers only, with `invalid_fill_value = 0` rather than `NaN`: a `NaN` in
    # one cell spreads to the whole domain within a few advection steps.
    negative_tracer_modifier = ScaleNegativeTracers(OXYDEP_TRACERS; invalid_fill_value = 0)
    return OXYDEP(
        grid,
        config.parameter_file;
        scale_negatives = false,
        modifiers = negative_tracer_modifier,
        light_attenuation_model = light_attenuation,
    )
end

function Simulations.coupled_model(
    model::OxyDepModel, grid; forcing, boundary_conditions, biogeochemistry, kwargs...,
)
    return Simulations.coupled_model(
        model.base,
        grid;
        forcing = oxydep_forcing(forcing, biogeochemistry),
        boundary_conditions = oxydep_boundary_conditions(boundary_conditions, biogeochemistry, grid),
        biogeochemistry = biogeochemistry,
        kwargs...,
    )
end

# Clear-sky PAR at the surface, W m⁻². `time` is seconds from `start_date`, 1 January, so
# day-of-year needs no offset; moving `start_date` means adding one here.
@inline function surface_PAR(longitude, latitude, time)
    day_of_year = time / days
    latitude_radians = deg2rad(latitude)
    solar_declination = deg2rad(23.44) *
                        sin(2π * (day_of_year - 81) / 365)
    solar_time = mod(time / hours + longitude / 15, 24)
    hour_angle = deg2rad(15) * (solar_time - 12)
    solar_elevation = sin(latitude_radians) * sin(solar_declination) +
                      cos(latitude_radians) * cos(solar_declination) * cos(hour_angle)
    earth_orbit_factor = 1 + 0.033 * cos(2π * (day_of_year - 3) / 365)
    return max(0, 1361 * 0.43 * 0.5 * earth_orbit_factor * solar_elevation)
end

function inneroslofjorden_oxydep(;
    architecture = :auto,
    initial_conditions_source = INITIAL_CONDITIONS_SOURCE,
    initial_results_directory = INITIAL_RESULTS_DIRECTORY,
)
    base = inneroslofjorden(; architecture)
    physics = base.model_config
    run = base.run_config
    tracers = (:T, :S, OXYDEP_TRACERS...)
    uniform_initial_conditions = merge(
        INNEROSLOFJORDEN_UNIFORM_INITIAL_CONDITIONS,
        (
            NUT = 11.365169f0,
            P = 0.05f0,
            HET = 0.015f0,
            POM = 0.01f0,
            DOM = 0.782484f0,
            :O₂ => 307.45563f0,
        ),
    )
    model = CoupledHydrostaticModel(
        buoyancy = physics.buoyancy,
        closure = physics.closure,
        # Temperature's scheme for the six OxyDep tracers: upwind within three cells of a river
        # column, WENO elsewhere. The river file carries NUT, POM and DOM, which make the same front
        # at a mouth as salinity does, and a tracer the NamedTuple omits would fall back to the
        # unbounded `Centered()`.
        tracer_advection = merge(
            physics.tracer_advection,
            NamedTuple{OXYDEP_TRACERS}(map(_ -> physics.tracer_advection.T, OXYDEP_TRACERS)),
        ),
        momentum_advection = physics.momentum_advection,
        tracers = tracers,
        coriolis = physics.coriolis,
        sea_ice = physics.sea_ice,
        free_surface = physics.free_surface,
        extra_kwargs = physics.extra_kwargs,
    )
    return FjordConfig(
        grid_config = base.grid_config,
        bathymetry_config = base.bathymetry_config,
        # Writes and reads `forcing_rivers_oxydep.nc`.
        river_config = CSVRiversConfig(base.river_config),
        # Writes and reads `boundaries_oxydep.nc`.
        boundary_data_config = InnerOslofjordOpenBoundary.boundary_config(
            data_root = base.bathymetry_config.data_root,
            years = base.boundary_data_config.years,
            open_edges = base.boundary_data_config.open_edges,
        ),
        atmosphere_config = base.atmosphere_config,
        model_config = OxyDepModel(model, tracers),
        biogeochemistry_config = OxyDepBiogeochemistry(
            normpath(joinpath(@__DIR__, "..", "..", "src", "oxydep_bgc_params.toml")),
            surface_PAR,
        ),
        boundary_condition_configs = base.boundary_condition_configs,
        run_config = RunConfig(
            results_root = fjord_results_root("inneroslofjorden_oxydep"),
            architecture = run.architecture,
            writers = (
                SnapshotWriter(
                    name = :ocean,
                    output_file = "snapshots_ocean.nc",
                    variables = (:T, :S, :u, :v, :e, OXYDEP_TRACERS...),
                    interval = 6hours,
                    split_times = Float64[],
                    overwrite_existing = true,
                ),
                FieldSnapshotWriter(
                    name = :surface,
                    output_file = "snapshots_surface.jld2",
                    variables = (:η,),
                    interval = 6hours,
                    overwrite_existing = true,
                ),
                CheckpointWriter(interval = 24hours, cleanup = true),
            ),
            callbacks = run.callbacks,
            time_stepping = run.time_stepping,
            initial_conditions = initial_conditions_for_run(
                initial_conditions_source,
                uniform_initial_conditions;
                results_directory = initial_results_directory,
            ),
            start_date = run.start_date,
            stop_time = run.stop_time,
            loops = run.loops,
            pickup = PICKUP,
        ),
    )
end

inneroslofjorden_oxydep()
