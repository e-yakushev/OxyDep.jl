"""
Local daily river forcing for OxyDep.

Run with:
    julia --project=/home/eya/src/FjordSim.jl prepare_coastal_sources_oxydep.jl

The script loads inneroslofjorden_oxydep.jl and calls FjordSim.add_rivers: the output is a
standalone (Nx, Ny, time) river file, not a copy of the interior forcing. Bathymetry must
already exist; no NVE key, download, or prepared interior forcing is needed.

CSV column 1 is day of year, column 2 discharge in m3/s, and column 3 concentration.
Discharge signs encode old grid-face directions: river 12 is positive, rivers 13-16
negative. DISCHARGE_DIRECTIONS converts them to non-negative inflow volumes and rejects
a sign reversal, rather than silently taking the absolute value of an actual outflow.
NUT, POM and DOM use the CSV's mmol N/m3 directly, salinity uses PSU, and temperature
uses degrees Celsius (despite the temperature CSV's generic c_mmol/m3 header).
Column 4 is already discharge times concentration and must not be used as concentration.

Edit CSV_TRACERS for concentration = scale * CSV_column_3 + offset, CONSTANT_TRACERS
for tracers without observations, or RIVER_CONCENTRATION_OVERRIDES for a constant
concentration for one river/tracer. These settings change river loads, not BGC kinetics.
The final record may be held for the one daily padding step after the data year.
All other missing dates are errors; data are never repeated as a climatology.
"""
module CoastalRiverSources

using Dates: Date, Day, dayofyear, daysinyear, year
using DelimitedFiles: readdlm
using FjordSim

export CSVRiversConfig

struct CSVTracer
    name::String
    suffix::String
    scale::Float64
    offset::Float64
end

const CSV_TRACERS = [
    CSVTracer("T", "temp", 1.0, 0.0),
    CSVTracer("S", "S", 1.0, 0.0),
    CSVTracer("NUT", "N3_n", 1.0, 0.0),
    CSVTracer("POM", "PON0", 1.0, 0.0),
    CSVTracer("DOM", "R2_n", 1.0, 0.0),
]
const CONSTANT_TRACERS = Dict("P" => 0.0, "HET" => 0.0, "O₂" => 300.0)

# For example, 12 => Dict("S" => 0.0, "NUT" => 100.0) overrides those two series.
const RIVER_CONCENTRATION_OVERRIDES = Dict{Int,Dict{String,Float64}}()
const DISCHARGE_DIRECTIONS = Dict(12 => 1, 13 => -1, 14 => -1, 15 => -1, 16 => -1)

# Original tracer-cell centers on the 63 x 114 grid spanning (10.46, 10.79), (59.62, 59.92).
# Store their geographic positions so changing the model resolution does not move the outlets.
const RIVER_LOCATIONS = [
    RiverLocation(id, "CSV river $id",
        10.46 + (i - 1//2) * (10.79 - 10.46) / 63,
        59.62 + (j - 1//2) * (59.92 - 59.62) / 114)
    for (id, i, j) in ((12, 52, 60), (13, 56, 107), (14, 35, 110), (15, 14, 102), (16, 12, 32))
]

struct CSVRiversConfig <: AbstractRiverConfig
    data_root::String
    output_file::String
    plot_file::String
    directory::String
    locations::Vector{RiverLocation}
    years::Vector{Int}
    search_radius::Int
    maximum_inflow_velocity::Float64
    csv_tracers::Vector{CSVTracer}
    constants::Dict{String,Float64}
    overrides::Dict{Int,Dict{String,Float64}}
    discharge_directions::Dict{Int,Int}
    hold_final_day::Bool
end

function CSVRiversConfig(
    base;
    directory = joinpath(base.data_root, "Rivers"),
    output_file = "forcing_rivers_oxydep.nc",
    plot_file = "forcing_rivers_oxydep.png",
    locations = RIVER_LOCATIONS,
    years = base.years,
    csv_tracers = CSV_TRACERS,
    constants = CONSTANT_TRACERS,
    overrides = RIVER_CONCENTRATION_OVERRIDES,
    discharge_directions = DISCHARGE_DIRECTIONS,
    hold_final_day = true,
)
    names = [tracer.name for tracer in csv_tracers]
    length(unique(names)) == length(names) || error("Duplicate CSV tracer names")
    isempty(intersect(names, keys(constants))) ||
        error("A river tracer cannot have both CSV and constant values")
    ids = [location.id for location in locations]
    !isempty(ids) && length(unique(ids)) == length(ids) ||
        error("River locations must be nonempty and have unique IDs")
    isempty(years) && error("Specify at least one river data year")
    all(id -> haskey(discharge_directions, id) && abs(discharge_directions[id]) == 1, ids) ||
        error("Specify a discharge direction of +1 or -1 for each river")
    for (id, concentrations) in overrides
        id in ids || error("Concentration overrides name unknown river $id")
        all(name -> name in names || haskey(constants, name), keys(concentrations)) ||
            error("Concentration overrides for river $id name an unknown tracer")
    end
    all(isfinite, values(constants)) || error("River constants must be finite")
    all(concentrations -> all(isfinite, values(concentrations)), values(overrides)) ||
        error("River concentration overrides must be finite")
    all(tracer -> isfinite(tracer.scale) && isfinite(tracer.offset), csv_tracers) ||
        error("CSV tracer scales and offsets must be finite")

    return CSVRiversConfig(
        base.data_root, String(output_file), String(plot_file), String(directory),
        copy(locations), sort(unique(years)), base.search_radius, base.maximum_inflow_velocity,
        copy(csv_tracers), copy(constants), deepcopy(overrides), copy(discharge_directions),
        hold_final_day,
    )
end

FjordSim.river_locations(config::CSVRiversConfig) = config.locations

function FjordSim.download_rivers(target_grid, config::CSVRiversConfig)
    isdir(config.directory) || error("River CSV directory does not exist: $(config.directory)")
    @info "Using local river CSVs from $(config.directory); no download required"
    return nothing
end

function read_river_csv(config, id, suffix, data_year)
    path = joinpath(config.directory, "river_$(suffix)_from_$(id)_year_$(data_year).csv")
    isfile(path) || error("Missing river CSV: $path")
    table, header = readdlm(path, ';', Float64; header = true)
    size(table, 1) > 0 && size(table, 2) >= 3 ||
        error("Expected daily rows with at least three columns in $path")
    strip(header[1]) == "DayOfYear" && strip(header[2]) == "flux_m3/s" ||
        error("Expected DayOfYear and flux_m3/s as the first two columns in $path")
    all(isfinite, table[:, 1:3]) || error("Non-finite day, discharge or concentration in $path")
    days = table[:, 1]
    all(day -> isinteger(day) && 1 <= day <= daysinyear(data_year), days) ||
        error("Invalid day of year in $path")
    issorted(days) && length(unique(days)) == length(days) ||
        error("Days must be strictly increasing without duplicates in $path")
    discharge = config.discharge_directions[id] .* table[:, 2]
    all(>=(0), discharge) ||
        error("Discharge reverses the configured inflow direction in $path")
    dates = Date(data_year, 1, 1) .+ Day.(Int.(days) .- 1)
    return (; dates, discharge, concentration = table[:, 3])
end

function river_records(config, id, suffix)
    dates = Date[]
    discharge = Float64[]
    concentration = Float64[]
    for data_year in config.years
        records = read_river_csv(config, id, suffix, data_year)
        append!(dates, records.dates)
        append!(discharge, records.discharge)
        append!(concentration, records.concentration)
    end
    return (; dates, discharge, concentration)
end

function record_indices(config, records, times, id, suffix)
    indices = Dict(date => index for (index, date) in enumerate(records.dates))
    last_date = last(records.dates)
    held = false
    result = map(times) do time
        date = Date(time)
        if haskey(indices, date)
            return indices[date]
        elseif config.hold_final_day && date == last_date + Day(1) &&
               dayofyear(date) == 1 && year(last_date) == last(config.years)
            held = true
            return length(records.dates)
        end
        error("River $id ($suffix) has no CSV record for $date")
    end
    held && @info "Holding river $id ($suffix) record from $last_date for the final padding day"
    return result
end

function FjordSim.river_discharge(config::CSVRiversConfig, times)
    discharge = Matrix{Float64}(undef, length(config.locations), length(times))
    for (row, location) in enumerate(config.locations)
        records = river_records(config, location.id, "N3_n")
        indices = record_indices(config, records, times, location.id, "N3_n")
        discharge[row, :] = records.discharge[indices]
    end
    return discharge
end

function FjordSim.river_series(config::CSVRiversConfig, times)
    shape = (length(config.locations), length(times))
    series = Dict(name => fill(value, shape) for (name, value) in config.constants)
    for tracer in config.csv_tracers
        series[tracer.name] = Matrix{Float64}(undef, shape)
    end
    for (row, location) in enumerate(config.locations)
        reference = river_records(config, location.id, "N3_n")
        for tracer in config.csv_tracers
            records = river_records(config, location.id, tracer.suffix)
            records.dates == reference.dates ||
                error("River $(location.id): $(tracer.suffix) and N3_n have different date axes")
            all(isapprox.(records.discharge, reference.discharge; rtol = 1e-6, atol = 1e-8)) ||
                error("River $(location.id): $(tracer.suffix) and N3_n have different discharges")
            indices = record_indices(config, records, times, location.id, tracer.suffix)
            series[tracer.name][row, :] =
                tracer.scale .* records.concentration[indices] .+ tracer.offset
        end
        for (name, value) in get(config.overrides, location.id, Dict{String,Float64}())
            series[name][row, :] .= value
        end
    end
    for (name, values) in series
        all(isfinite, values) || error("Non-finite river concentration for $name")
        name == "T" || all(>=(0), values) || error("Negative river concentration for $name")
    end
    return series
end

function main()
    config = FjordSim.Setups.fjord_config(joinpath(@__DIR__, "inneroslofjorden_oxydep.jl"))
    result = Base.invokelatest(add_rivers, config)
    @info "OxyDep CSV river forcing saved to $(result.output_file)"
    return result
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    CoastalRiverSources.main()
end
