using FjordSim
using Oceananigans
using Oceananigans.Units: day, days, hour, hours, minutes, second
using NCDatasets
using DelimitedFiles
using Dates: Date, dayofyear
using OceanBioME.Light: TwoBandPhotosyntheticallyActiveRadiation
using OceanBioME: ScaleNegativeTracers
using Oceananigans.BoundaryConditions: FieldBoundaryConditions

include(joinpath(@__DIR__, "inneroslofjorden.jl"))
include(joinpath(@__DIR__, "src", "Oxydep.jl"))
using .OXYDEPModel: OXYDEP, bgh_oxydep_boundary_conditions, oxydep_sediment_forcings

const SEA_BOUNDARY_DIRECTORY = joinpath(
    homedir(), "FjordSim_data", "inneroslofjorden", "Sea_boundary",
)
const SEA_BOUNDARY_DEPTHS = Float64[
    0.5, 1.5, 2.5, 4.0, 6.25, 8.75, 12.5, 17.5, 25.0,
    35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0, 107.5,
]
const OXYDEP_TRACERS = (:NUT, :P, :HET, :POM, :DOM, :O₂)

struct SeaBoundaryData
    days::Vector{Int}
    values::Dict{Symbol,Matrix{Float64}}
end

function read_sea_boundary_file(name)
    table, _ = readdlm(joinpath(SEA_BOUNDARY_DIRECTORY, name * ".csv"), ';'; header = true)
    return Int.(table[:, 1]), Float64.(table[:, 2:end])
end

function load_sea_boundary_data()
    days, nut = read_sea_boundary_file("NUT")
    days_dom, dom = read_sea_boundary_file("DOM")
    days_o2, oxygen = read_sea_boundary_file("O2")
    days == days_dom == days_o2 || error("Sea-boundary CSV files use different day axes")
    return SeaBoundaryData(days, Dict(:NUT => nut, :DOM => dom, :O₂ => oxygen))
end

function sea_boundary_value(data, tracer, depth, date)
    if tracer === :P
        return 0.05
    elseif tracer === :HET
        return 0.015
    elseif tracer === :POM
        return 0.01
    end

    target_depth = clamp(-Float64(depth), first(SEA_BOUNDARY_DEPTHS), last(SEA_BOUNDARY_DEPTHS))
    row = clamp(dayofyear(Date(date)), first(data.days), last(data.days))
    day_row = findfirst(==(row), data.days)
    values = data.values[tracer][day_row, :]
    upper = clamp(searchsortedlast(SEA_BOUNDARY_DEPTHS, target_depth), 1, length(SEA_BOUNDARY_DEPTHS) - 1)
    weight = (target_depth - SEA_BOUNDARY_DEPTHS[upper]) /
             (SEA_BOUNDARY_DEPTHS[upper + 1] - SEA_BOUNDARY_DEPTHS[upper])
    return (1 - weight) * values[upper] + weight * values[upper + 1]
end

const OXYDEP_DEFLATE_LEVEL = 5

function append_forcing_variables!(filepath, data)
    NCDataset(filepath, "a") do ds
        depths = ds["Nz"][:]
        dates = ds["time"][:]
        shape = (ds.dim["Nx"], ds.dim["Ny"], ds.dim["Nz"])
        wet = map(value -> !ismissing(value) && isfinite(value), ds["S"][:, :, :, 1])
        zero_rates = zeros(Float32, shape)

        for tracer in OXYDEP_TRACERS
            name = String(tracer)
            variable = defVar(ds, name, Float32, ("Nx", "Ny", "Nz", "time");
                chunksizes = [shape[1], shape[2], 1, 1],
                deflatelevel = OXYDEP_DEFLATE_LEVEL,
                attrib = ["_FillValue" => NaN32])
            rates = defVar(ds, name * "_lambda", Float32, ("Nx", "Ny", "Nz", "time");
                chunksizes = [shape[1], shape[2], 1, 1],
                deflatelevel = OXYDEP_DEFLATE_LEVEL,
                attrib = ["_FillValue" => NaN32])

            slab = Array{Float32}(undef, shape)
            for (time_index, date) in enumerate(dates)
                column = Float32[sea_boundary_value(data, tracer, depth, date) for depth in depths]
                for k in axes(slab, 3)
                    slab[:, :, k] .= ifelse.(wet[:, :, k], column[k], NaN32)
                end
                variable[:, :, :, time_index] = slab
                rates[:, :, :, time_index] = zero_rates
            end
        end
    end
    return filepath
end

function append_boundary_variables!(filepath, edges, data)
    NCDataset(filepath, "a") do ds
        depths = ds["Nz"][:]
        dates = ds["time"][:]
        for edge in edges
            along = edge in (:south, :north) ? "Nx" : "Ny"
            n_along = ds.dim[along]
            for tracer in OXYDEP_TRACERS
                name = boundary_variable_name(edge, String(tracer))
                variable = defVar(ds, name, Float32, (along, "Nz", "time");
                    deflatelevel = OXYDEP_DEFLATE_LEVEL,
                    attrib = ["_FillValue" => NaN32])
                values = Float32[
                    sea_boundary_value(data, tracer, depth, date)
                    for depth in depths, date in dates
                ]
                variable[:, :, :] = repeat(reshape(values, 1, size(values)...), n_along, 1, 1)
            end
        end
    end
    return filepath
end

struct OxyDepModel{M,T,P} <: AbstractCoupledSimulationConfig
    base::M
    tracers::T
    parameter_file::String
    surface_PAR::P
end

OxyDepModel(base; tracers, parameter_file, surface_PAR) =
    OxyDepModel(base, Tuple(Symbol.(tracers)), String(parameter_file), surface_PAR)

FjordSim.model_tracers(model::OxyDepModel) = model.tracers

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

function FjordSim.coupled_simulation(model::OxyDepModel, grid; forcing, boundary_conditions, kwargs...)
    light_attenuation = TwoBandPhotosyntheticallyActiveRadiation(grid, model.surface_PAR)
    negative_tracer_modifier = ScaleNegativeTracers(
        (:NUT, :P, :HET, :POM, :DOM, :O₂);
        invalid_fill_value = 0,
    )
    bgc = OXYDEP(
        grid,
        model.parameter_file;
        scale_negatives = false,
        modifiers = negative_tracer_modifier,
        light_attenuation_model = light_attenuation,
    )
    configured_model = CoupledHydrostaticSimulation(
        buoyancy = model.base.buoyancy,
        closure = model.base.closure,
        tracer_advection = model.base.tracer_advection,
        momentum_advection = model.base.momentum_advection,
        tracers = model.tracers,
        coriolis = model.base.coriolis,
        sea_ice = model.base.sea_ice,
        biogeochemistry = bgc,
        free_surface = model.base.free_surface,
        extra_kwargs = model.base.extra_kwargs,
    )
    return coupled_simulation(
        configured_model,
        grid;
        forcing = oxydep_forcing(forcing, bgc),
        boundary_conditions = oxydep_boundary_conditions(boundary_conditions, bgc, grid),
        kwargs...,
    )
end

struct OxyDepForcing{F} <: AbstractForcingConfig
    base::F
    data_root::String
    output_file::String
    plot_file::String
    rivers
end

OxyDepForcing(base) = OxyDepForcing(
    base, base.data_root, base.output_file, base.plot_file, base.rivers,
)
FjordSim.download_forcing(target_grid, config::OxyDepForcing) = download_forcing(target_grid, config.base)

function FjordSim.prepare_forcing(target_grid, config::OxyDepForcing; coverage = nothing, edges = nothing)
    result = prepare_forcing(target_grid, config.base; coverage, edges)
    append_forcing_variables!(result.output_file, load_sea_boundary_data())
    return (; result..., variables = [result.variables; String.(OXYDEP_TRACERS)])
end

struct OxyDepBoundaries{B} <: AbstractBoundaryDataConfig
    base::B
    data_root::String
    output_file::String
    plot_file::String
    open_edges::Vector{Symbol}
end

OxyDepBoundaries(base) = OxyDepBoundaries(
    base, base.data_root, base.output_file, base.plot_file, base.open_edges,
)
FjordSim.boundary_variable_names(config::OxyDepBoundaries) = merge(
    boundary_variable_names(config.base),
    Dict(String(tracer) => String(tracer) for tracer in OXYDEP_TRACERS),
)
FjordSim.download_boundaries(target_grid, config::OxyDepBoundaries) =
    download_boundaries(target_grid, config.base)

function FjordSim.prepare_boundaries(target_grid, config::OxyDepBoundaries; coverage = nothing)
    result = prepare_boundaries(target_grid, config.base; coverage)
    append_boundary_variables!(result.output_file, config.open_edges, load_sea_boundary_data())
    appended = [boundary_variable_name(edge, String(tracer))
                for edge in config.open_edges for tracer in OXYDEP_TRACERS]
    return (; result..., variables = [result.variables; appended])
end

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

base = inneroslofjorden()
sea_boundary_data = load_sea_boundary_data()

rivers = deepcopy(base.forcing_config.rivers)
rivers.output_file = "forcing_rivers_oxydep.nc"
rivers.plot_file = "forcing_rivers_oxydep.png"
forcing = OxyDepForcing(NorKystConfig(
    data_root = base.forcing_config.data_root,
    output_directory = base.forcing_config.output_directory,
    output_file = "forcing_oxydep.nc",
    plot_file = "forcing_oxydep.png",
    architecture = base.forcing_config.architecture,
    parameters = base.forcing_config.parameters,
    years = base.forcing_config.years,
    rivers = rivers,
))

boundaries = OxyDepBoundaries(NorKystBoundariesConfig(
    data_root = joinpath(homedir(), "FjordSim_data", "oslofjorden"),
    output_directory = joinpath(homedir(), "FjordSim_data", "oslofjorden", "norkyst_hourly"),
    output_file = joinpath(base.boundary_config.data_root, "boundaries_oxydep.nc"),
    plot_file = joinpath(base.boundary_config.data_root, "boundaries_oxydep.png"),
    open_edges = base.boundary_config.open_edges,
    margin = base.boundary_config.margin,
    architecture = base.boundary_config.architecture,
    parameters = base.boundary_config.parameters,
    years = base.boundary_config.years,
))

simulation = base.simulation_config
FjordConfig(
    grid_config = base.grid_config,
    bathymetry_config = base.bathymetry_config,
    forcing_config = forcing,
    boundary_config = boundaries,
    atmosphere_config = base.atmosphere_config,
    simulation_config = SimulationConfig(
        results_root = joinpath(homedir(), "FjordSim_results", "inneroslofjorden_oxydep"),
        architecture = simulation.architecture,
        model = OxyDepModel(
            simulation.model,
            tracers = (:T, :S, OXYDEP_TRACERS...),
            parameter_file = joinpath(@__DIR__, "src", "oxydep_bgc_params.toml"),
            surface_PAR = surface_PAR,
        ),
        boundary_conditions = simulation.boundary_conditions,
        writers = (
            SnapshotWriter(
                name = :ocean,
                output_file = "snapshots_ocean.nc",
                variables = (:T, :S, :u, :v, :e, :NUT, :P, :HET, :POM, :DOM, :O₂),
                interval = 6hours,
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
        callbacks = simulation.callbacks,
        time_stepping = simulation.time_stepping,
        ## for uniform initial conditions from forcing
         initial_conditions = FromForcing(),
         start_date = simulation.start_date,

        ## for initial conditions from previous results
        #initial_conditions = FromResults(joinpath(homedir(), "FjordSim_results", "inneroslofjorden_oxydep",
        #     "snapshots_ocean_20261001T090439.nc"),),
       # start_date = DateTime(2020, 1, 1),

        stop_time = simulation.stop_time,
        loops = simulation.loops,
        pickup = simulation.pickup,
    ),
)