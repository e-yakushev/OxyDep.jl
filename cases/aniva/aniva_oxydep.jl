# The `aniva.jl` hydrophysics with OxyDep biogeochemistry (NUT, P, HET, POM, DOM, O₂).
#
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim prepare_bathymetry     --config aniva_oxydep.jl   # checks the file
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim download_boundary_data --config aniva_oxydep.jl   # combines local files
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim prepare_boundary_data  --config aniva_oxydep.jl
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim prepare_atmosphere     --config aniva_oxydep.jl
#   julia --project=/home/eya/src/FjordSim.jl -m FjordSim run_simulation         --config aniva_oxydep.jl
#
# The initial state (1 January 2023) and the open southern boundary carry, besides the Copernicus
# hydrophysics `aniva.jl` reads:
#
# - NUT, P and DOM from Copernicus GLOBAL_ANALYSISFORECAST_BGC daily means in `copernicus/`: nitrate
#   (`no3`), total phytoplankton (`phyc`) and dissolved organic nitrogen (`don`), all mmol m⁻³. OxyDep
#   counts P in nitrogen and Copernicus phytoplankton in carbon, so P is `phyc` x 16/106 (Redfield).
#   `download_initial_state` and `download_boundary_data` combine the three files into
#   `copernicus_1day/copernicus_bgc_2023.nc`, and the prepare steps regrid it with FjordSim's own
#   interpolation onto the same cells and times as the hydrophysics.
# - O₂, which Copernicus does not provide here, from the daily `Sea_boundary/O2.csv` profiles.
# - HET and POM at constant concentrations.
#
# run_simulation reads boundaries_aniva.nc from prepare_open_boundary_aniva.jl, whose oxygen
# is at 100% air saturation using boundary T/S and JRA55 pressure, not the legacy profile-based
# boundary preparation below. The river_config writes forcing_rivers_oxydep.nc on the 2023
# time axis, explicitly using 2024 river observations as a month/day seasonal proxy.
# No prepared initial-state file is used. INITIAL_CONDITIONS_SOURCE selects an editable uniform
# spin-up state or the last snapshot of an archived run. The GPU is required by default.

using FjordSim
using FjordSim: BoundaryData, Grids, InitialState, Simulations
using FjordSim.Configs:
    AbstractBiogeochemistryConfig, AbstractBoundaryDataConfig, AbstractInitialStateConfig, AbstractModelConfig
using Oceananigans: WENO
using Oceananigans.Units: days, hours
using OceanBioME.Light: TwoBandPhotosyntheticallyActiveRadiation
using OceanBioME: ScaleNegativeTracers
using Oceananigans.BoundaryConditions: FieldBoundaryConditions
using NCDatasets
using DelimitedFiles: readdlm
using Dates: Date, DateTime, Day, dayofyear

include(joinpath(@__DIR__, "aniva.jl"))
if !isdefined(@__MODULE__, :OXYDEPModel)
    include(joinpath(@__DIR__, "..", "..", "src", "Oxydep.jl"))
end
using .OXYDEPModel: OXYDEP, bgh_oxydep_boundary_conditions, oxydep_sediment_forcings

const OXYDEP_TRACERS = (:NUT, :P, :HET, :POM, :DOM, :O₂)

INITIAL_CONDITIONS_SOURCE = :uniform
INITIAL_RESULTS_DIRECTORY = fjord_results_root("aniva_oxydep_init")
PICKUP = false
const ANIVA_OXYDEP_UNIFORM_INITIAL_CONDITIONS = merge(
    ANIVA_UNIFORM_INITIAL_CONDITIONS,
    (NUT = 5f0, P = 0.1f0, HET = 0.015f0, POM = 0.01f0, DOM = 1f0, O₂ = 300f0),
)

# OxyDep tracer => Copernicus BGC variable, the file `download_copernicus.jl` left it in, and the
# factor converting it to OxyDep's units.
const COPERNICUS_BGC = Dict(
    "NUT" => (source = "no3", file = "copernicus_no3_daily.nc", scale = 1.0),
    "P" => (source = "phyc", file = "copernicus_phyc_daily.nc", scale = 16 / 106),
    "DOM" => (source = "don", file = "copernicus_don_daily.nc", scale = 1.0),
)
const COPERNICUS_BGC_DIRECTORY = "copernicus"
const COPERNICUS_BGC_COMBINED_FILE = "copernicus_bgc_2023.nc"

# The tracers Copernicus does not provide, at the concentrations the previous Aniva runs used.
const CONSTANT_TRACERS = Dict("HET" => 0.015, "POM" => 0.01)
const OXYGEN_PROFILE_FILE = joinpath("Sea_boundary", "O2.csv")

# --- Copernicus BGC source ----------------------------------------------------------------------

"""
    combine_copernicus_bgc(data_root, output_directory)

Combine the three Copernicus BGC files in `data_root/copernicus` into one file in
`data_root/output_directory`, repeating the last daily record a day later so the data reaches the
end of the run (see `aniva.jl`'s boundary source). Returns its path.
"""
function combine_copernicus_bgc(data_root, output_directory)
    output_path = joinpath(data_root, output_directory, COPERNICUS_BGC_COMBINED_FILE)
    temporary_path = output_path * ".tmp"
    rm(temporary_path; force = true)
    coordinates = nothing

    NCDataset(temporary_path, "c") do output
        for name in sort!(collect(keys(COPERNICUS_BGC)))
            entry = COPERNICUS_BGC[name]
            NCDataset(joinpath(data_root, COPERNICUS_BGC_DIRECTORY, entry.file)) do source
                current = (source["longitude"][:], source["latitude"][:], Float64.(source["depth"][:]), source["time"][:])
                if isnothing(coordinates)
                    coordinates = current
                    longitude, latitude, depth, times = current
                    output_times = [times; last(times) + Day(1)]
                    for (dimension, values) in (("longitude", longitude), ("latitude", latitude), ("depth", depth), ("time", output_times))
                        defDim(output, dimension, length(values))
                        defVar(output, dimension, values, (dimension,))
                    end
                    output.attrib["history"] = "Record $(last(output_times)) repeats $(last(times))"
                end
                all(map((a, b) -> a ≈ b, current[1:3], coordinates[1:3])) && current[4] == coordinates[4] ||
                    error("$(entry.file) is on another grid or time axis than the other Copernicus BGC files")

                records = [eachindex(current[4]); lastindex(current[4])]
                variable = defVar(output, entry.source, Float32, ("longitude", "latitude", "depth", "time");
                    deflatelevel = 5, attrib = ["_FillValue" => NaN32, "units" => source[entry.source].attrib["units"]])
                variable[:, :, :, :] = Float32.(coalesce.(source[entry.source][:, :, :, records], NaN32))
            end
        end
    end

    mv(temporary_path, output_path; force = true)
    @info "Copernicus BGC source saved to $output_path"
    return output_path
end

copernicus_bgc_variable_names() = Dict(entry.source => name for (name, entry) in COPERNICUS_BGC)

"""
    CopernicusBGCBoundaryDataConfig

The Copernicus BGC variables along the open edges, regridded from `copernicus_bgc_2023.nc` into an
intermediate file that `OxyDepBoundaryDataConfig` merges into the hydrophysical boundary file.
"""
Base.@kwdef mutable struct CopernicusBGCBoundaryDataConfig <: AbstractBoundaryDataConfig
    data_root::String
    output_directory::String
    output_file::String
    plot_file::String
    open_edges::Vector{Symbol}
    architecture::Symbol
    parameters::Vector{String} = [entry.source for entry in values(COPERNICUS_BGC)]
end

BoundaryData.boundary_variable_names(::CopernicusBGCBoundaryDataConfig) = copernicus_bgc_variable_names()
BoundaryData.boundary_source_grid(::CopernicusBGCBoundaryDataConfig, filepath) = copernicus_source_grid(filepath)
BoundaryData.boundary_time_steps(config::CopernicusBGCBoundaryDataConfig) =
    copernicus_records(joinpath(BoundaryData.boundary_data_directory(config), COPERNICUS_BGC_COMBINED_FILE))

# --- O₂ profiles --------------------------------------------------------------------------------

struct OxygenProfile
    depths::Vector{Float64}
    values::Matrix{Float64}
end

function read_oxygen_profile(filepath)
    isfile(filepath) || error("Missing oxygen profile file $filepath")
    table, header = readdlm(filepath, ';', Float64; header = true)
    labels = strip.(String.(vec(header)))
    first(labels) == "DayOfYear" && all(label -> startswith(label, "depth_"), labels[2:end]) ||
        error("Expected DayOfYear;depth_<metres>;... columns in $filepath")
    Int.(table[:, 1]) == collect(1:size(table, 1)) || error("Expected consecutive days from 1 in $filepath")
    return OxygenProfile(parse.(Float64, chopprefix.(labels[2:end], "depth_")), table[:, 2:end])
end

"""
    oxygen(profile, z, date)

O₂ at height `z` (negative down) on `date`: the profile of `date`'s day of year (the last day for a
day the file lacks), linear in depth and held at its end values beyond them.
"""
function oxygen(profile::OxygenProfile, z, date)
    depth = clamp(-Float64(z), first(profile.depths), last(profile.depths))
    day = min(dayofyear(Date(date)), size(profile.values, 1))
    lower = clamp(searchsortedlast(profile.depths, depth), 1, length(profile.depths) - 1)
    weight = (depth - profile.depths[lower]) / (profile.depths[lower + 1] - profile.depths[lower])
    return (1 - weight) * profile.values[day, lower] + weight * profile.values[day, lower + 1]
end

"""
    biogeochemical_value(name, regridded, profile, z, date)

The OxyDep tracer `name` at one cell: the regridded Copernicus value converted to OxyDep units, the
O₂ profile, or a constant.
"""
function biogeochemical_value(name, regridded, profile, z, date)
    haskey(COPERNICUS_BGC, name) && return COPERNICUS_BGC[name].scale * regridded
    name == "O₂" && return oxygen(profile, z, date)
    return CONSTANT_TRACERS[name]
end

# --- Initial state ------------------------------------------------------------------------------

"""
    OxyDepInitialStateConfig(hydrophysics; output_file)

`hydrophysics`' initial state, written to `output_file`, with the six OxyDep tracers added: NUT, P
and DOM regridded from the Copernicus BGC record of the same day, O₂ from the profile file, HET and
POM constant.
"""
struct OxyDepInitialStateConfig <: AbstractInitialStateConfig
    hydrophysics::CopernicusInitialStateConfig
    biogeochemistry::CopernicusInitialStateConfig
    data_root::String
    output_file::String
end

function OxyDepInitialStateConfig(base::CopernicusInitialStateConfig; output_file)
    hydrophysics = CopernicusInitialStateConfig(
        data_root = base.data_root,
        output_directory = base.output_directory,
        source_file = base.source_file,
        output_file = output_file,
        variable_names = base.variable_names,
        parameters = base.parameters,
    )
    biogeochemistry = CopernicusInitialStateConfig(
        data_root = base.data_root,
        output_directory = base.output_directory,
        source_file = COPERNICUS_BGC_COMBINED_FILE,
        output_file = replace(output_file, ".nc" => "_bgc_intermediate.nc"),
        variable_names = copernicus_bgc_variable_names(),
        parameters = [entry.source for entry in values(COPERNICUS_BGC)],
    )
    return OxyDepInitialStateConfig(hydrophysics, biogeochemistry, base.data_root, output_file)
end

function InitialState.fetch_initial_state(target_grid, config::OxyDepInitialStateConfig, date)
    InitialState.fetch_initial_state(target_grid, config.hydrophysics, date)
    combine_copernicus_bgc(config.data_root, config.biogeochemistry.output_directory)
    return InitialState.initial_state_record(config.biogeochemistry, date).filepath
end

function InitialState.regrid_initial_state(target_grid, config::OxyDepInitialStateConfig, date; edges = nothing)
    result = InitialState.regrid_initial_state(target_grid, config.hydrophysics, date; edges)
    regridded = InitialState.regrid_initial_state(target_grid, config.biogeochemistry, date; edges)
    profile = read_oxygen_profile(joinpath(config.data_root, OXYGEN_PROFILE_FILE))

    NCDataset(regridded.output_file) do bgc
        NCDataset(result.output_file, "a") do ds
            z = ds["Nz"][:]
            wet = isfinite.(coalesce.(ds["T"][:, :, :, 1], NaN32))
            for tracer in OXYDEP_TRACERS
                name = String(tracer)
                source = haskey(COPERNICUS_BGC, name) ? coalesce.(bgc[name][:, :, :, 1], NaN32) : zeros(Float32, size(wet))
                values = Float32[
                    wet[i, j, k] ? biogeochemical_value(name, source[i, j, k], profile, z[k], date) : NaN32
                    for i in axes(wet, 1), j in axes(wet, 2), k in axes(wet, 3)
                ]
                variable = defVar(ds, name, Float32, ("Nx", "Ny", "Nz", "time");
                    deflatelevel = 5, attrib = ["_FillValue" => NaN32])
                variable[:, :, :, 1] = values
                @info "Initial $name: $(extrema(filter(isfinite, values)))"
            end
        end
    end
    rm(regridded.output_file; force = true)

    return (; result..., variables = [result.variables; String.(collect(OXYDEP_TRACERS))])
end

# --- Open southern boundary ---------------------------------------------------------------------

"""
    OxyDepBoundaryDataConfig(hydrophysics; output_file, plot_file)

`hydrophysics`' boundary data, written to `output_file`, with the six OxyDep tracers added on every
open edge: NUT, P and DOM regridded from the Copernicus BGC daily means on the same hourly axis, O₂
from the profile file, HET and POM constant. Every model tracer needs a series along an open edge,
so `OpenLateralBoundaryFromData` would fail without them.
"""
struct OxyDepBoundaryDataConfig <: AbstractBoundaryDataConfig
    hydrophysics::CopernicusBoundaryDataConfig
    biogeochemistry::CopernicusBGCBoundaryDataConfig
    data_root::String
    output_file::String
    plot_file::String
    open_edges::Vector{Symbol}
    architecture::Symbol
    parameters::Vector{String}
end

function OxyDepBoundaryDataConfig(base::CopernicusBoundaryDataConfig; output_file, plot_file)
    hydrophysics = CopernicusBoundaryDataConfig(
        data_root = base.data_root,
        output_directory = base.output_directory,
        source_file = base.source_file,
        sea_level_file = base.sea_level_file,
        combined_file = base.combined_file,
        output_file = output_file,
        plot_file = plot_file,
        open_edges = base.open_edges,
        architecture = base.architecture,
        parameters = base.parameters,
    )
    biogeochemistry = CopernicusBGCBoundaryDataConfig(
        data_root = base.data_root,
        output_directory = base.output_directory,
        output_file = replace(output_file, ".nc" => "_bgc_intermediate.nc"),
        plot_file = replace(plot_file, ".png" => "_bgc_intermediate.png"),
        open_edges = base.open_edges,
        architecture = base.architecture,
    )
    return OxyDepBoundaryDataConfig(
        hydrophysics, biogeochemistry, base.data_root, output_file, plot_file, base.open_edges,
        base.architecture, base.parameters,
    )
end

BoundaryData.boundary_variable_names(config::OxyDepBoundaryDataConfig) = merge(
    BoundaryData.boundary_variable_names(config.hydrophysics),
    Dict(String(tracer) => String(tracer) for tracer in OXYDEP_TRACERS),
)
BoundaryData.boundary_source_grid(config::OxyDepBoundaryDataConfig, filepath) =
    BoundaryData.boundary_source_grid(config.hydrophysics, filepath)
BoundaryData.boundary_time_steps(config::OxyDepBoundaryDataConfig) =
    BoundaryData.boundary_time_steps(config.hydrophysics)

function BoundaryData.fetch_boundary_data(target_grid, config::OxyDepBoundaryDataConfig)
    BoundaryData.fetch_boundary_data(target_grid, config.hydrophysics)
    return combine_copernicus_bgc(config.data_root, config.biogeochemistry.output_directory)
end

function BoundaryData.regrid_boundary_data(target_grid, config::OxyDepBoundaryDataConfig; coverage = nothing)
    result = BoundaryData.regrid_boundary_data(target_grid, config.hydrophysics; coverage)
    regridded = BoundaryData.regrid_boundary_data(target_grid, config.biogeochemistry; coverage)
    profile = read_oxygen_profile(joinpath(config.data_root, OXYGEN_PROFILE_FILE))

    NCDataset(regridded.output_file) do bgc
        NCDataset(result.output_file, "a") do ds
            dates = DateTime.(ds["time"][:])
            dates == DateTime.(bgc["time"][:]) || error("The BGC boundary file is on another time axis")
            z = ds["Nz"][:]
            for edge in config.open_edges
                wet = isfinite.(coalesce.(ds[BoundaryData.boundary_variable_name(edge, "S")][:, :, 1], NaN32))
                for tracer in OXYDEP_TRACERS
                    name = String(tracer)
                    variable_name = BoundaryData.boundary_variable_name(edge, name)
                    source = haskey(COPERNICUS_BGC, name) ?
                        coalesce.(bgc[variable_name][:, :, :], NaN32) : zeros(Float32, size(wet)..., length(dates))
                    variable = defVar(ds, variable_name, Float32, BoundaryData.boundary_dimension_names(Val(edge), name);
                        chunksizes = [size(wet)..., 1], deflatelevel = 5, attrib = ["_FillValue" => NaN32])
                    for (index, date) in enumerate(dates)
                        variable[:, :, index] = Float32[
                            wet[i, k] ? biogeochemical_value(name, source[i, k, index], profile, z[k], date) : NaN32
                            for i in axes(wet, 1), k in axes(wet, 2)
                        ]
                    end
                end
            end
        end
    end
    rm(regridded.output_file; force = true)

    appended = [BoundaryData.boundary_variable_name(edge, String(tracer)) for edge in config.open_edges for tracer in OXYDEP_TRACERS]
    return (; result..., variables = [result.variables; appended])
end

# --- The model ----------------------------------------------------------------------------------

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

# FjordSim's forcing stays, and OxyDep's sediment term is added to each tracer it names.
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

# --- The config ---------------------------------------------------------------------------------

function aniva_oxydep(;
    architecture = :gpu,
    river_config = AnivaCoastalSources.AnivaRiversConfig(target_year = 2023),
    initial_conditions_source = INITIAL_CONDITIONS_SOURCE,
    initial_results_directory = INITIAL_RESULTS_DIRECTORY,
    pickup = PICKUP,
)
    base = aniva(; architecture, river_config)
    physics = base.model_config
    run = base.run_config
    tracers = (:T, :S, OXYDEP_TRACERS...)

    model = CoupledHydrostaticModel(
        buoyancy = physics.buoyancy,
        closure = physics.closure,
        # Temperature's scheme for the six OxyDep tracers; a tracer the NamedTuple omits would fall
        # back to the unbounded `Centered()`.
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
        river_config = base.river_config,
        boundary_data_config = OxyDepBoundaryDataConfig(
            base.boundary_data_config;
            output_file = "boundaries_aniva.nc",
            plot_file = "boundaries_aniva.png",
        ),
        atmosphere_config = base.atmosphere_config,
        model_config = OxyDepModel(model, tracers),
        biogeochemistry_config = OxyDepBiogeochemistry(
            normpath(joinpath(@__DIR__, "..", "..", "src", "oxydep_bgc_params.toml")),
            surface_PAR,
        ),
        boundary_condition_configs = base.boundary_condition_configs,
        run_config = RunConfig(
            results_root = fjord_results_root("aniva_oxydep"),
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
            initial_conditions = aniva_initial_conditions_for_run(
                initial_conditions_source,
                ANIVA_OXYDEP_UNIFORM_INITIAL_CONDITIONS;
                results_directory = initial_results_directory,
            ),
            start_date = run.start_date,
            stop_time = run.stop_time,
            loops = run.loops,
            pickup = pickup,
        ),
    )
end

aniva_oxydep()
