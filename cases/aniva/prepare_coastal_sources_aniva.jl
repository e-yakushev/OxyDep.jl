#=
Run from this directory: julia prepare_coastal_sources_aniva.jl

Reads rivers_aniva/ and bathymetry_aniva.nc under fjord_data_root("aniva"), then writes
forcing_rivers_oxydep.nc in the current FjordSim river-file format. No downloads are made.
The CSVs are daily 2024 data used as a seasonal proxy for the 2023 run. The default window
is 2023-01-01 through 2024-01-01: observations are mapped by month/day, 29 February is
omitted, and the final daily record is held for the padding day. The output metadata records
this assumption. AnivaRiversConfig(target_year=nothing) preserves the original 2024 dates.

River coordinates below are longitude, latitude. FjordSim snaps them to coastal water cells
and spreads their volume over connected columns to limit inflow speed to 1e-3 m/s.
For a 1 m surface layer and a 180 s step this gives a vertical inflow Courant number of 0.18.

CSV column 3 is concentration, not a discharge-weighted load: T is degrees Celsius,
S is freshwater salinity (zero in these files), and NUT/DOM are mmol N/m3. Leading blank
NUT/DOM concentrations are held at the first available value, with a warning and metadata.
Interior/trailing missing concentrations are errors. P, HET and POM are explicitly zero.
Oxygen is 100% air saturation using OceanBioME Garcia-Gordon (1992), river T/S and 1 atm,
in mmol O2/m3. No oxygen-profile or atmosphere file is required.

Including this file defines AnivaCoastalSources.AnivaRiversConfig, usable as river_config
in a FjordSim setup. To replace an existing output, call
AnivaCoastalSources.main(overwrite=true). Direct execution never overwrites existing output.
FJORDSIM_PROJECT overrides the sibling FjordSim.jl environment used for direct execution.
=#
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import Pkg
    project_directory = abspath(get(ENV, "FJORDSIM_PROJECT",
        joinpath(@__DIR__, "..", "..", "..", "FjordSim.jl")))
    isfile(joinpath(project_directory, "Project.toml")) || error(
        "FjordSim project not found at $project_directory; set FJORDSIM_PROJECT to its directory.",
    )
    Pkg.activate(project_directory)
end

module AnivaCoastalSources

using Dates: Dates, Date, DateTime, Day, daysinyear
using DelimitedFiles: readdlm
using FjordSim: Rivers, Grids, fjord_data_root
using FjordSim.Configs: AbstractRiverConfig
using FjordSim.Rivers: RiverLocation
using Oceananigans: CPU, RectilinearGrid
using Oceananigans.Fields: ConstantField
using OceanBioME.Models.GasExchangeModel: GarciaGordonOxygenSaturation, surface_value
using NCDatasets: NCDataset, defVar

include(joinpath(@__DIR__, "..", "inneroslofjorden", "prepare_coastal_sources_oxydep.jl"))
using .CoastalRiverSources: CSVRiversConfig, CSVTracer

const OXYGEN_NAME = "O\u2082"
const RIVER_LOCATIONS = [
    RiverLocation(1, "Lyutoga", 142.5302243, 46.6978129),
    RiverLocation(2, "Susuya", 142.723888, 46.7427152),
]

struct AnivaRiversConfig{Y} <: AbstractRiverConfig
    base::CSVRiversConfig
    output_file::String
    missing_concentrations::Symbol
    target_year::Y
end

function AnivaRiversConfig(;
    data_root = fjord_data_root("aniva"),
    directory = joinpath(data_root, "rivers_aniva"),
    output_file = "forcing_rivers_oxydep.nc",
    years = [2024],
    search_radius = 10,
    maximum_inflow_velocity = 1e-3,
    locations = RIVER_LOCATIONS,
    constants = Dict("P" => 0.0, "HET" => 0.0, "POM" => 0.0),
    missing_concentrations = :hold_first,
    target_year = 2023,
)
    missing_concentrations in (:hold_first, :error) ||
        throw(ArgumentError("missing_concentrations must be :hold_first or :error"))
    Set(keys(constants)) == Set(("P", "HET", "POM")) ||
        throw(ArgumentError("Specify constants for P, HET and POM only; oxygen is calculated"))
    all(>=(0), values(constants)) || throw(ArgumentError("Tracer constants must be non-negative"))
    isnothing(target_year) || length(years) == 1 ||
        throw(ArgumentError("A seasonal proxy requires exactly one source year"))
    base = CSVRiversConfig(;
        data_root, directory, output_file, years, search_radius, maximum_inflow_velocity,
        locations, constants,
        plot_file = "forcing_rivers_oxydep.png",
        csv_tracers = [CSVTracer(name, suffix, 1.0, 0.0)
                      for (name, suffix) in (("T", "temp"), ("S", "S"), ("NUT", "N3_n"), ("DOM", "R2_n"))],
        discharge_directions = Dict(location.id => 1 for location in locations),
    )
    return AnivaRiversConfig(base, joinpath(data_root, output_file), missing_concentrations, target_year)
end

Rivers.river_locations(config::AnivaRiversConfig) = config.base.locations
Rivers.river_forcing_path(config::AnivaRiversConfig) = config.output_file
Rivers.river_search_radius(config::AnivaRiversConfig) = config.base.search_radius
Rivers.river_maximum_inflow_velocity(config::AnivaRiversConfig) = config.base.maximum_inflow_velocity

function Rivers.fetch_rivers(target_grid, config::AnivaRiversConfig)
    return Rivers.fetch_rivers(target_grid, config.base)
end

function river_records(config::AnivaRiversConfig, id, suffix)
    dates = Date[]
    discharge = Float64[]
    concentration = Float64[]
    for data_year in config.base.years
        filepath = joinpath(config.base.directory, "river_$(suffix)_from_$(id)_year_$(data_year).csv")
        isfile(filepath) || error("Missing river CSV: $filepath")
        table, header = readdlm(filepath, ';', String; header = true)
        size(table, 1) > 0 && size(table, 2) >= 3 || error("Expected daily rows and three columns in $filepath")
        strip(header[1]) == "DayOfYear" && strip(header[2]) == "flux_m3/s" ||
            error("Expected DayOfYear;flux_m3/s headers in $filepath")
        days = parse.(Float64, strip.(table[:, 1]))
        all(day -> isinteger(day) && 1 <= day <= daysinyear(data_year), days) ||
            error("Invalid day of year in $filepath")
        issorted(days) && length(unique(days)) == length(days) || error("Duplicate or unsorted days in $filepath")
        flows = parse.(Float64, strip.(table[:, 2]))
        all(value -> isfinite(value) && value >= 0, flows) || error("Expected finite non-negative inflows in $filepath")
        values = [isempty(strip(cell)) ? NaN : parse(Float64, strip(cell)) for cell in table[:, 3]]
        missing_rows = findall(isnan, values)
        if !isempty(missing_rows)
            first_valid = findfirst(isfinite, values)
            config.missing_concentrations == :hold_first && suffix in ("N3_n", "R2_n") &&
                !isnothing(first_valid) && all(row -> row < first_valid, missing_rows) ||
                error("Missing concentrations in $filepath; only leading NUT/DOM blanks may be held")
            values[missing_rows] .= values[first_valid]
            @warn "Holding first available river concentration over leading blank days" river=id tracer=suffix days=days[missing_rows] value=values[first_valid]
        end
        all(isfinite, values) || error("Non-finite concentrations in $filepath")
        suffix == "temp" || all(>=(0), values) || error("Negative concentration in $filepath")
        append!(dates, Date(data_year, 1, 1) .+ Day.(Int.(days) .- 1))
        append!(discharge, flows)
        append!(concentration, values)
    end
    if !isnothing(config.target_year)
        selected = findall(date -> !(Dates.month(date) == 2 && Dates.day(date) == 29 &&
                                   !Dates.isleapyear(config.target_year)), dates)
        dates = [Date(config.target_year, Dates.month(dates[index]), Dates.day(dates[index])) for index in selected]
        discharge = discharge[selected]
        concentration = concentration[selected]
    end
    return (; dates, discharge, concentration)
end

function record_indices(config::AnivaRiversConfig, records, times, id, suffix)
    if isnothing(config.target_year)
        return CoastalRiverSources.record_indices(config.base, records, times, id, suffix)
    end
    indices = Dict(date => index for (index, date) in enumerate(records.dates))
    return map(times) do time
        date = Date(time)
        haskey(indices, date) && return indices[date]
        config.base.hold_final_day && date == Date(config.target_year + 1, 1, 1) &&
            last(records.dates) == date - Day(1) && return length(records.dates)
        error("River $id ($suffix) has no seasonal-proxy record for $date")
    end
end

function Rivers.river_discharge(config::AnivaRiversConfig, times)
    discharge = Matrix{Float64}(undef, length(config.base.locations), length(times))
    for (row, location) in enumerate(config.base.locations)
        records = river_records(config, location.id, "temp")
        indices = record_indices(config, records, times, location.id, "temp")
        discharge[row, :] = records.discharge[indices]
    end
    return discharge
end

function Rivers.river_series(config::AnivaRiversConfig, times)
    shape = (length(config.base.locations), length(times))
    series = Dict(name => fill(value, shape) for (name, value) in config.base.constants)
    for tracer in config.base.csv_tracers
        series[tracer.name] = Matrix{Float64}(undef, shape)
    end
    for (row, location) in enumerate(config.base.locations)
        reference = river_records(config, location.id, "temp")
        for tracer in config.base.csv_tracers
            records = river_records(config, location.id, tracer.suffix)
            records.dates == reference.dates || error("River $(location.id) has inconsistent tracer dates")
            all(isapprox.(records.discharge, reference.discharge; rtol = 1e-6, atol = 1e-8)) ||
                error("River $(location.id) has inconsistent tracer discharges")
            indices = record_indices(config, records, times, location.id, tracer.suffix)
            series[tracer.name][row, :] = records.concentration[indices]
        end
    end
    saturation = GarciaGordonOxygenSaturation(Float64)
    grid = RectilinearGrid(CPU(), Float64; size = (1, 1, 1), extent = (1, 1, 1))
    oxygen = Matrix{Float64}(undef, shape)
    for index in eachindex(oxygen)
        fields = (T = ConstantField(series["T"][index]), S = ConstantField(series["S"][index]))
        oxygen[index] = surface_value(saturation, 1, 1, grid, nothing, fields)
    end
    series[OXYGEN_NAME] = oxygen
    return series
end

function main(;
    config = AnivaRiversConfig(),
    coverage = isnothing(config.target_year) ?
        (DateTime(minimum(config.base.years), 1, 1), DateTime(maximum(config.base.years) + 1, 1, 1)) :
        (DateTime(config.target_year, 1, 1), DateTime(config.target_year + 1, 1, 1)),
    overwrite = false,
)
    coverage[1] < coverage[2] || throw(ArgumentError("coverage must have increasing dates"))
    bathymetry_file = joinpath(config.base.data_root, "bathymetry_aniva.nc")
    isfile(bathymetry_file) || error("Missing bathymetry: $bathymetry_file")
    output_file = Rivers.river_forcing_path(config)
    isfile(output_file) && !overwrite && error("Output exists: $output_file; call main(overwrite=true) to replace it")
    grid = Grids.ImmersedBoundaryGrid(bathymetry_file, CPU(), (1, 1, 1), Float32)
    Rivers.fetch_rivers(grid, config)
    mkpath(dirname(output_file))
    return mktempdir(dirname(output_file); prefix = "aniva_rivers_") do directory
        staged = AnivaRiversConfig(config.base, joinpath(directory, basename(output_file)),
            config.missing_concentrations, config.target_year)
        result = Rivers.place_rivers(grid, staged; coverage)
        Set(mouth.location.id for mouth in result.mouths) == Set(location.id for location in config.base.locations) ||
            error("Not all river mouths were placed; the existing output is unchanged")
        all(mouth -> maximum(abs, Rivers.river_discharge(config, result.times)[findfirst(location -> location.id == mouth.location.id, config.base.locations), :]) /
                     mouth.area <= config.base.maximum_inflow_velocity * (1 + 1e-6), result.mouths) ||
            error("Insufficient connected water to respect maximum_inflow_velocity")
        NCDataset(result.output_file, "a") do dataset
            dataset.attrib["source_directory"] = basename(normpath(config.base.directory))
            dataset.attrib["missing_concentration_policy"] = string(config.missing_concentrations)
            dataset.attrib["concentration_gap_assumption"] = "Leading NUT/DOM blanks held at first available value when policy is hold_first"
            dataset.attrib["constant_tracers_mmol_m3"] = join(["$name=$(config.base.constants[name])" for name in sort(collect(keys(config.base.constants)))], ", ")
            dataset.attrib["oxygen_saturation"] = "100 percent; OceanBioME Garcia-Gordon (1992); atmospheric pressure 101325 Pa"
            dataset.attrib["time_policy"] = isnothing(config.target_year) ?
                "CSV calendar years retained; final daily record held for one padding day only" :
                "Seasonal proxy: source $(only(config.base.years)) mapped by month/day to $(config.target_year); February 29 omitted for a non-leap target; final day held for one padding day"
            longitude, latitude = NCDataset(bathymetry_file) do bathymetry
                return bathymetry["lon"][:], bathymetry["lat"][:]
            end
            for (name, values) in (
                ("river_source_longitude", [mouth.location.longitude for mouth in result.mouths]),
                ("river_source_latitude", [mouth.location.latitude for mouth in result.mouths]),
                ("river_grid_longitude", [longitude[mouth.i] for mouth in result.mouths]),
                ("river_grid_latitude", [latitude[mouth.j] for mouth in result.mouths]),
            )
                defVar(dataset, name, values, ("river",))
            end
            for name in result.variables
                dataset[name].attrib["units"] = name == "T" ? "degree_Celsius" : name == "S" ? "1" : "mmol m-3"
            end
        end
        mv(result.output_file, output_file; force = overwrite)
        @info "Aniva coastal forcing saved" output_file coverage variables=result.variables
        return (; result..., output_file)
    end
end

end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    AnivaCoastalSources.main()
end