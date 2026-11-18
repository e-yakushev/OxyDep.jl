"""
The `inneroslofjorden.jl` hydrophysics with OxyDep biogeochemistry (NUT, P, HET, POM, DOM, O₂).

There is no interior forcing file. The model adds bottom-layer point sources of NUT, DOM and Ci_free
at grid indices (16, 80) and (55, 91), each at 10 mmol/s per tracer. It reads:

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
`INITIAL_RESULTS_DIRECTORY`. Any missing Ci fields in a prior snapshot start at zero. Either way
the run starts on `start_date`, 1 January 2020, and repeats the 2020 boundary, river and atmosphere
cycle.
"""

using FjordSim
using FjordSim: Simulations
using FjordSim.Configs: AbstractBiogeochemistryConfig, AbstractModelConfig
using Oceananigans.Units: days, hours
using Oceananigans.Grids: bottommost_active_node, Center
using Oceananigans.Operators: Ax
using OceanBioME.Light: TwoBandPhotosyntheticallyActiveRadiation
using OceanBioME: ScaleNegativeTracers
using Oceananigans.BoundaryConditions: FieldBoundaryConditions

include(joinpath(@__DIR__, "inneroslofjorden.jl"))
# This case needs the Ci carbon tracers: set `const Ci_ = true` in src/Oxydep.jl.
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

const OXYDEP_TRACERS = (
    :NUT, :P, :HET, :POM, :DOM, :O₂, :Ci_free, :Ci_PHY, :Ci_HET, :Ci_POM, :Ci_DOM,
)
const POINT_SOURCE_TRACERS = (:NUT, :DOM, :Ci_free)
const POINT_SOURCE_LOCATIONS = ((16, 80), (55, 91))
# Each location receives these tracer loads in mmol/s.
const POINT_SOURCE_RATES = (NUT = 5f0, DOM = 10f0, Ci_free = 8f0)

struct BottomPointSourceForcing{L,R}
    locations::L
    rate::R
end

@inline function (forcing::BottomPointSourceForcing)(i, j, k, grid, clock, fields)
    bottom = bottommost_active_node(i, j, k, grid, Center(), Center(), Center())
    source_column =
        ((i == forcing.locations[1][1]) & (j == forcing.locations[1][2])) |
        ((i == forcing.locations[2][1]) & (j == forcing.locations[2][2]))
    rate = ifelse(source_column & bottom, forcing.rate, zero(forcing.rate))
    flux_density = rate / Ax(i, j, k, grid, Center(), Center(), Center())
    return FjordSim.Forcing.forcing_term_x_flux(2, flux_density, i, j, k, grid)
end

# The initial-conditions switch: `:uniform` for the uniform state, `:results` for the last record of
# a previous run's ocean snapshot in `INITIAL_RESULTS_DIRECTORY`. The new run writes to
# `fjord_results_root("inneroslofjorden_oxydep")` in both cases, so archive what is there first.
#INITIAL_CONDITIONS_SOURCE = :results
INITIAL_CONDITIONS_SOURCE = :uniform
INITIAL_RESULTS_DIRECTORY = fjord_results_root("inneroslofjorden_oxydep_init")

# `true` resumes the run in `fjord_results_root("inneroslofjorden_oxydep")` from its newest
# checkpoint, superseding the initial conditions above; with no checkpoint there it warns and starts
# fresh. `false` starts fresh and refuses a directory that still holds an earlier run's checkpoints.
PICKUP = true

"""
    OxyDepModel(base, tracers)

`base`, a `CoupledHydrostaticModel`, with OxyDep's sediment forcing added to the forcing and its
bottom-layer point sources added to NUT, DOM and Ci_free, and its surface condition replacing the
top boundary condition of each tracer it names.
"""
struct OxyDepModel{M,T} <: AbstractModelConfig
    base::M
    tracers::T
end

struct OxyDepBiogeochemistry{P} <: AbstractBiogeochemistryConfig
    parameter_file::String
    ci_parameter_file::String
    surface_PAR::P
end

Simulations.model_tracers(model::OxyDepModel) = model.tracers

# FjordSim's river forcing stays unchanged. OxyDep sediment forcing and the configured point sources
# are added independently to their tracer tendencies.
function oxydep_forcing(base, bgc)
    sediment = oxydep_sediment_forcings(bgc)
    combined = map(keys(sediment)) do name
        haskey(base, name) ? (base[name], sediment[name]) : sediment[name]
    end
    with_sediment = merge(base, NamedTuple{keys(sediment)}(combined))
    point_sources = NamedTuple{POINT_SOURCE_TRACERS}(map(
        name -> BottomPointSourceForcing(POINT_SOURCE_LOCATIONS, POINT_SOURCE_RATES[name]),
        POINT_SOURCE_TRACERS,
    ))
    combined = map(keys(point_sources)) do name
        haskey(with_sediment, name) ?
            append_forcing(with_sediment[name], point_sources[name]) : point_sources[name]
    end
    return merge(with_sediment, NamedTuple{keys(point_sources)}(combined))
end

append_forcing(existing::Tuple, addition) = (existing..., addition)
append_forcing(existing, addition) = (existing, addition)

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
    # Conserve nitrogen, carbon and oxygen independently, retaining the zero fallback for a
    # negative group total.
    negative_tracer_modifier = ScaleNegativeTracers(
        (
            nitrogen = (:NUT, :P, :HET, :POM, :DOM),
            oxygen = (:O₂,),
            carbon = (:Ci_free, :Ci_PHY, :Ci_HET, :Ci_POM, :Ci_DOM),
        );
        invalid_fill_value = 0,
    )
    return OXYDEP(
        grid,
        config.parameter_file;
        ci_file = config.ci_parameter_file,
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
            Ci_free = 0f0,
            Ci_PHY = 0f0,
            Ci_HET = 0f0,
            Ci_POM = 0f0,
            Ci_DOM = 0f0,
        ),
    )
    model = CoupledHydrostaticModel(
        buoyancy = physics.buoyancy,
        closure = physics.closure,
        # Temperature's scheme for every OxyDep tracer: upwind within three cells of a river
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
            normpath(joinpath(@__DIR__, "..", "..", "src", "oxydep_ci_params.toml")),
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
