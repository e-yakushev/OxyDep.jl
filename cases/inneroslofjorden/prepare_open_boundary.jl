"""
Prepare local NorKyst and OxyDep open-boundary data with the FjordSim environment:

    julia --project=/path/to/FjordSim.jl cases/inneroslofjorden/prepare_openboundary.jl

Reads bathymetry.nc, norkyst_hourly/, and bgc_open_boundary/ under
fjord_data_root("inneroslofjorden"). Writes boundaries_oxydep.nc for the southern
edge, covering 2020 through 2021-01-01. No downloads are performed.

Temperature, salinity, velocity, elevation, and barotropic velocity use NorKyst.
NUT, DOM, and O2 CSV values retain their source units and are interpolated linearly
in depth, held constant within each day and beyond their shallowest/deepest depth.
The five Ci tracers use zero exterior concentrations.
Day 366 uses day 365 when the CSV has only 365 days. P = 0.05, HET = 0.015, and
POM = 0.01 match the existing OxyDep case. TEMP.csv and SALT.csv are not used.

To replace an existing output, include this script and call
InnerOslofjordOpenBoundary.main(overwrite=true). In a FjordConfig, use
InnerOslofjordOpenBoundary.boundary_config() as boundary_data_config so FjordSim
loads both the hydrophysical and OxyDep variables from the combined file.
"""
module InnerOslofjordOpenBoundary

using Dates: Date, DateTime, dayofyear
using DelimitedFiles: readdlm
using NCDatasets: NCDataset, defVar
using Oceananigans: CPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid
using FjordSim: NorKystBoundaryDataConfig, fjord_data_root
using FjordSim: BoundaryData, Grids, Plotting
using FjordSim.Configs: AbstractBoundaryDataConfig

const OXYDEP_TRACERS = (
    "NUT", "P", "HET", "POM", "DOM", "O\u2082",
    "Ci_free", "Ci_PHY", "Ci_HET", "Ci_POM", "Ci_DOM",
)
const CONSTANT_TRACERS = Dict(
    "P" => 0.05, "HET" => 0.015, "POM" => 0.01,
    "Ci_free" => 0.0, "Ci_PHY" => 0.0, "Ci_HET" => 0.0, "Ci_POM" => 0.0, "Ci_DOM" => 0.0,
)

struct BoundaryProfile
    days::Vector{Int}
    depths::Vector{Float64}
    values::Matrix{Float64}
end

struct OxyDepBoundaryDataConfig{C} <: AbstractBoundaryDataConfig
    hydrophysics::C
    biogeochemistry_directory::String
    data_root::String
    output_directory::String
    output_file::String
    plot_file::String
    open_edges::Vector{Symbol}
    architecture::Symbol
    parameters::Vector{String}
    years::Vector{Int}
end

BoundaryData.boundary_time_steps(config::OxyDepBoundaryDataConfig) =
    BoundaryData.boundary_time_steps(config.hydrophysics)
BoundaryData.boundary_source_grid(config::OxyDepBoundaryDataConfig, filepath) =
    BoundaryData.boundary_source_grid(config.hydrophysics, filepath)
BoundaryData.boundary_source_slab(config::OxyDepBoundaryDataConfig, reader, step, name) =
    BoundaryData.boundary_source_slab(config.hydrophysics, reader, step, name)
BoundaryData.boundary_variable_names(config::OxyDepBoundaryDataConfig) = merge(
    BoundaryData.boundary_variable_names(config.hydrophysics),
    Dict(name => name for name in OXYDEP_TRACERS),
)

function boundary_config(;
    data_root = fjord_data_root("inneroslofjorden"),
    years = [2020],
    open_edges = :south,
    architecture = :cpu,
    output_file = "boundaries_oxydep.nc",
)
    hydrophysics = NorKystBoundaryDataConfig(
        data_root = data_root,
        output_directory = "norkyst_hourly",
        output_file = output_file,
        plot_file = "boundaries_oxydep.png",
        open_edges = open_edges,
        architecture = architecture,
        parameters = [
            "temperature", "salinity", "u_eastward", "v_northward", "zeta", "ubar", "vbar",
        ],
        years = years,
    )
    return OxyDepBoundaryDataConfig(
        hydrophysics,
        joinpath(data_root, "bgc_open_boundary"),
        hydrophysics.data_root,
        hydrophysics.output_directory,
        hydrophysics.output_file,
        hydrophysics.plot_file,
        hydrophysics.open_edges,
        hydrophysics.architecture,
        hydrophysics.parameters,
        hydrophysics.years,
    )
end

function read_profile(filepath)
    isfile(filepath) || error("Missing biogeochemical boundary profile: $filepath")
    table, header = readdlm(filepath, ';', Float64; header = true)
    labels = String.(vec(header))
    first(labels) == "DayOfYear" || error("Expected DayOfYear as the first column in $filepath")
    all(label -> startswith(label, "depth_"), labels[2:end]) ||
        error("Expected depth_<metres> column labels in $filepath")
    depths = parse.(Float64, chopprefix.(labels[2:end], "depth_"))
    length(depths) >= 2 && all(isfinite, depths) && all(depths .>= 0) &&
        all(diff(depths) .> 0) || error("Depths must be finite, nonnegative, and increasing in $filepath")
    days = Int.(table[:, 1])
    (days == collect(1:365) || days == collect(1:366)) ||
        error("Expected consecutive days 1:365 or 1:366 in $filepath")
    values = table[:, 2:end]
    size(values, 2) == length(depths) || error("Depth columns do not match data in $filepath")
    all(isfinite, values) && all(values .>= 0) ||
        error("Biogeochemical concentrations must be finite and nonnegative in $filepath")
    return BoundaryProfile(days, depths, values)
end

function load_profiles(directory)
    profiles = Dict(
        "NUT" => read_profile(joinpath(directory, "NUT.csv")),
        "DOM" => read_profile(joinpath(directory, "DOM.csv")),
        "O\u2082" => read_profile(joinpath(directory, "O2.csv")),
    )
    reference_days = profiles["NUT"].days
    all(profile -> profile.days == reference_days, values(profiles)) ||
        error("Biogeochemical boundary CSVs must use the same day axis")
    return profiles
end

function profile_value(profile::BoundaryProfile, bottom_coordinate, date)
    depth = clamp(-Float64(bottom_coordinate), first(profile.depths), last(profile.depths))
    day = min(dayofyear(Date(date)), last(profile.days))
    lower = clamp(searchsortedlast(profile.depths, depth), 1, length(profile.depths) - 1)
    weight = (depth - profile.depths[lower]) / (profile.depths[lower + 1] - profile.depths[lower])
    return (1 - weight) * profile.values[day, lower] + weight * profile.values[day, lower + 1]
end

function append_biogeochemistry!(filepath, edges, profiles)
    NCDataset(filepath, "a") do dataset
        depths = dataset["Nz"][:]
        dates = DateTime.(dataset["time"][:])
        for edge in edges
            salinity_name = BoundaryData.boundary_variable_name(edge, "S")
            wet = map(value -> !ismissing(value) && isfinite(value), dataset[salinity_name][:, :, 1])
            for tracer in OXYDEP_TRACERS
                name = BoundaryData.boundary_variable_name(edge, tracer)
                haskey(dataset, name) && error("Boundary variable already exists: $name")
                dimensions = BoundaryData.boundary_dimension_names(Val(edge), tracer)
                variable = defVar(dataset, name, Float32, dimensions;
                    chunksizes = [size(wet)..., 1],
                    deflatelevel = 5,
                    attrib = ["_FillValue" => NaN32])
                if haskey(CONSTANT_TRACERS, tracer)
                    variable.attrib["source"] = "Constant concentration from the OxyDep case"
                else
                    variable.attrib["source"] = "Daily CSV profile; depth-linear interpolation; endpoint clamping"
                end
                slab = Matrix{Float32}(undef, size(wet))
                for (time_index, date) in enumerate(dates)
                    for depth_index in eachindex(depths)
                        concentration = haskey(CONSTANT_TRACERS, tracer) ? CONSTANT_TRACERS[tracer] :
                            profile_value(profiles[tracer], depths[depth_index], date)
                        slab[:, depth_index] .= ifelse.(wet[:, depth_index], Float32(concentration), NaN32)
                    end
                    variable[:, :, time_index] = slab
                end
            end
        end
        dataset.attrib["biogeochemistry_time_policy"] =
            "DayOfYear; daily values held constant; day 366 clamped to day 365 for 365-day profiles"
        dataset.attrib["biogeochemistry_depth_policy"] =
            "Linear interpolation at model depth centers; constant beyond profile endpoints"
    end
    return filepath
end

function BoundaryData.regrid_boundary_data(grid, config::OxyDepBoundaryDataConfig; coverage = nothing)
    profiles = load_profiles(config.biogeochemistry_directory)
    result = BoundaryData.regrid_boundary_data(grid, config.hydrophysics; coverage)
    append_biogeochemistry!(result.output_file, Grids.open_edges(config), profiles)
    variables = [result.variables; [
        BoundaryData.boundary_variable_name(edge, tracer)
        for edge in Grids.open_edges(config) for tracer in OXYDEP_TRACERS
    ]]
    return (; result..., variables)
end

function main(; config = boundary_config(), overwrite = false)
    bathymetry_file = joinpath(config.hydrophysics.data_root, "bathymetry.nc")
    isfile(bathymetry_file) || error("Prepare bathymetry first: $bathymetry_file")
    output_file = BoundaryData.boundary_data_path(config)
    isfile(output_file) && !overwrite &&
        error("Output already exists: $output_file. Call main(overwrite=true) to replace it.")
    load_profiles(config.biogeochemistry_directory)
    records = BoundaryData.boundary_time_steps(config)
    @info "Using local boundary inputs" bathymetry_file records = length(records)
    @warn "BGC profiles are held at their endpoint values beyond the CSV depth/day range; source units are unchanged"
    grid = ImmersedBoundaryGrid(bathymetry_file, CPU(), (7, 7, 7), Float32)
    coverage = (DateTime(minimum(config.hydrophysics.years), 1, 1),
                DateTime(maximum(config.hydrophysics.years) + 1, 1, 1))
    result = BoundaryData.regrid_boundary_data(grid, config; coverage)
    @info "Prepared hydrophysical and OxyDep boundaries" output_file variables = result.variables
    return result
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    InnerOslofjordOpenBoundary.main()
end