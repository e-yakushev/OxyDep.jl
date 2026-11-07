# The JRA55-do repeat-year (RYF) atmosphere for the Aniva Bay setup, as a FjordSim atmosphere config.
#
# Included by a setup, it defines `JRA55RYFConfig`, which `download_atmosphere`, `prepare_atmosphere`
# and `run_simulation` dispatch on. Run as a script, it prepares the atmosphere for the domain of
# `bathymetry_aniva.nc` directly:
#
#   julia jra55_atmosphere.jl
#
# Direct execution activates the sibling FjordSim.jl project. Set FJORDSIM_PROJECT to use a
# different checkout; including this file from a setup keeps that setup's active environment.
#
# which subsets the global RYF files in `~/FjordSim_data/aniva/JRA55/` (a few MB in `JRA55/subset/`,
# re-cut when the existing subset does not cover the domain) and writes
# `~/FjordSim_data/aniva/atmosphere_aniva.nc` and its plot for `START_DATE` to `STOP_DATE`.

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    import Pkg
    project_directory = abspath(get(ENV, "FJORDSIM_PROJECT",
        joinpath(@__DIR__, "..", "..", "..", "FjordSim.jl")))
    isfile(joinpath(project_directory, "Project.toml")) || error(
        "FjordSim project not found at $project_directory; set FJORDSIM_PROJECT to its directory.",
    )
    Pkg.activate(project_directory)
end

import FjordSim
using FjordSim: fjord_data_root
using FjordSim.Configs: AbstractAtmosphereConfig
using FjordSim.Atmospheres:
    Atmospheres, atmosphere_directory, atmosphere_path, atmosphere_target_axes,
    atmosphere_variable_attributes
using FjordSim.Atmospheres.NORA3: MultiYearNORA3, NORA3PrescribedAtmosphere, NORA3PrescribedRadiation
using FjordSim.Plotting: plot_atmosphere
using Oceananigans: CPU
using Oceananigans.Grids: x_domain, y_domain
using NCDatasets
using Dates: DateTime, Hour, Date
import Dates

# Prepared name => JRA55-do source variables, summed. Rain and snow are separate source fluxes.
const JRA55_SOURCE_VARIABLES = Dict(
    "u_wind_10m" => ("uas",),
    "v_wind_10m" => ("vas",),
    "air_temperature_2m" => ("tas",),
    "specific_humidity_2m" => ("huss",),
    "air_pressure_at_sea_level" => ("psl",),
    "precipitation" => ("prra", "prsn"),
    "swrad" => ("rsds",),
    "lwrad" => ("rlds",),
)

# Source cells kept around the prepared grid so bilinear interpolation has a full stencil at its edge.
const JRA55_SUBSET_MARGIN_CELLS = 2
const JRA55_STEPS_PER_DAY = 8
const JRA55_DEFLATE_LEVEL = 5

"""
    JRA55RYFConfig(; data_root, kw...)

Atmosphere configuration for the JRA55-do 1.4.0 repeat-year forcing (RYF), global 3-hourly files
`RYF.<variable>.<source_period>.nc` on the TL319 grid, already on disk in `atmosphere_directory`.

RYF is one climatological year, not the weather of any real year. The files carry a placeholder time
axis, so `regrid_atmosphere` dates each record by its position in the year: the record for a given
month, day and hour of the simulated window is the one at the same calendar position in the file,
with 29 February taking 28 February. The file's seasonal cycle (insolation, temperature) starts on
1 January, so no offset is applied.

`fetch_atmosphere` subsets the global files to the prepared grid's box, which shrinks them from
about 12 GB to a few MB, and `regrid_atmosphere` regrids that subset to a regular lon/lat grid in the
layout FjordSim's NORA3 reader takes, so the simulation reads it back with that reader.

# Fields
- `data_root`, `output_directory`: `output_directory` holds the global RYF files; the subset goes
  into its `subset` folder.
- `output_file`, `plot_file`: the prepared file and its diagnostic plot.
- `source_period`: the `<years>` part of the source file names.
- `resolution`, `padding`: prepared grid spacing and margin around the ocean domain, in degrees.
"""
Base.@kwdef mutable struct JRA55RYFConfig <: AbstractAtmosphereConfig
    data_root::String
    output_directory::String = "JRA55"
    output_file::String = "atmosphere_aniva.nc"
    plot_file::String = "atmosphere_aniva.png"
    source_period::String = "1990_1991"
    resolution::Float64 = 0.25
    padding::Float64 = 1.0
end

jra55_source_names() = sort!(unique(reduce(vcat, collect.(values(JRA55_SOURCE_VARIABLES)))))

jra55_source_path(config::JRA55RYFConfig, name) =
    joinpath(atmosphere_directory(config), "RYF.$name.$(config.source_period).nc")

jra55_subset_path(config::JRA55RYFConfig, name) =
    joinpath(atmosphere_directory(config), "subset", "RYF.$name.$(config.source_period).nc")

# The index range of a regular ascending axis covering [lower, upper] plus the margin cells.
function jra55_window(axis, lower, upper)
    (lower >= first(axis) && upper <= last(axis)) ||
        error("Requested $lower to $upper lies outside the source axis $(first(axis)) to $(last(axis))")
    first_index = max(searchsortedlast(axis, lower) - JRA55_SUBSET_MARGIN_CELLS, 1)
    last_index = min(searchsortedfirst(axis, upper) + JRA55_SUBSET_MARGIN_CELLS, length(axis))
    return first_index:last_index
end

# Whether the subset at `subset_path` already spans the prepared `longitude` and `latitude`.
function jra55_subset_covers(subset_path, longitude, latitude)
    isfile(subset_path) || return false
    return NCDataset(subset_path) do ds
        subset_longitude = ds["lon"][:]
        subset_latitude = ds["lat"][:]
        first(subset_longitude) <= first(longitude) && last(longitude) <= last(subset_longitude) &&
            first(subset_latitude) <= first(latitude) && last(latitude) <= last(subset_latitude)
    end
end

"""
    fetch_atmosphere(target_grid, config::JRA55RYFConfig)

Subset each global RYF file to the prepared grid's box, into `atmosphere_directory(config)/subset`.

A subset that already spans the prepared grid is kept; one that does not, cut for another domain,
`resolution` or `padding`, is cut again. Returns the subset directory.
"""
function Atmospheres.fetch_atmosphere(target_grid, config::JRA55RYFConfig)
    longitude, latitude = atmosphere_target_axes(target_grid, config)
    mkpath(dirname(jra55_subset_path(config, first(jra55_source_names()))))

    for name in jra55_source_names()
        subset_path = jra55_subset_path(config, name)
        jra55_subset_covers(subset_path, longitude, latitude) && continue
        source_path = jra55_source_path(config, name)
        isfile(source_path) || error("JRA55 source file $source_path does not exist")

        @info "Subsetting $(basename(source_path))"
        temporary_path = subset_path * ".tmp"
        NCDataset(source_path) do source
            source_longitude = Float64.(source["lon"][:])
            source_latitude = Float64.(source["lat"][:])
            longitude_window = jra55_window(source_longitude, first(longitude), last(longitude))
            latitude_window = jra55_window(source_latitude, first(latitude), last(latitude))
            values = Float32.(coalesce.(source[name][longitude_window, latitude_window, :], NaN32))

            NCDataset(temporary_path, "c") do output
                defDim(output, "lon", length(longitude_window))
                defDim(output, "lat", length(latitude_window))
                defDim(output, "time", size(values, 3))
                defVar(output, "lon", source_longitude[longitude_window], ("lon",))
                defVar(output, "lat", source_latitude[latitude_window], ("lat",))
                variable = defVar(output, name, Float32, ("lon", "lat", "time");
                    chunksizes = [size(values, 1), size(values, 2), 1],
                    deflatelevel = JRA55_DEFLATE_LEVEL,
                    attrib = ["_FillValue" => NaN32])
                variable[:, :, :] = values
            end
        end
        mv(temporary_path, subset_path; force = true)
    end

    return dirname(jra55_subset_path(config, first(jra55_source_names())))
end

# Position of `date` in the 365-day RYF year as a 1-based record index.
function jra55_cycle_index(date)
    calendar_day = Dates.dayofyear(Date(2001, Dates.month(date), min(Dates.day(date), Dates.month(date) == 2 ? 28 : 31)))
    return (calendar_day - 1) * JRA55_STEPS_PER_DAY + Dates.hour(date) ÷ 3 + 1
end

# Lower source index and weight of each target coordinate on a sorted source axis.
function jra55_interpolation_weights(axis, targets)
    (first(targets) >= first(axis) && last(targets) <= last(axis)) || error(
        "Prepared grid reaches outside the subset ($(first(targets)) to $(last(targets)) vs " *
        "$(first(axis)) to $(last(axis))). Re-run `download_atmosphere` to cut a new subset.",
    )
    lower = [clamp(searchsortedlast(axis, target), 1, length(axis) - 1) for target in targets]
    weight = [(target - axis[index]) / (axis[index + 1] - axis[index]) for (target, index) in zip(targets, lower)]
    return lower, weight
end

function jra55_interpolate!(output, slab, longitude_weights, latitude_weights)
    (lower_i, weight_i), (lower_j, weight_j) = longitude_weights, latitude_weights
    for j in axes(output, 2), i in axes(output, 1)
        a, b = lower_i[i], lower_j[j]
        s, t = weight_i[i], weight_j[j]
        output[i, j] = (1 - s) * (1 - t) * slab[a, b] + s * (1 - t) * slab[a + 1, b] +
                       (1 - s) * t * slab[a, b + 1] + s * t * slab[a + 1, b + 1]
    end
    return output
end

"""
    regrid_atmosphere(target_grid, config::JRA55RYFConfig; coverage)

Regrid the subset onto a regular lon/lat grid and write the prepared file the simulation reads.

`coverage` is the `(first, last)` window the run needs, from `coverage_window`. The time axis is the
window at the source's 3-hour spacing, both ends included, each record taken from the matching
position in the RYF year. Returns `(; output_file, times, variables)`.
"""
function Atmospheres.regrid_atmosphere(target_grid, config::JRA55RYFConfig; coverage = nothing)
    isnothing(coverage) && error("JRA55 repeat-year data needs a simulation window to date its records")
    longitude, latitude = atmosphere_target_axes(target_grid, config)

    arrays = Dict{String,Array{Float32,3}}()
    source_longitude = source_latitude = nothing
    for name in jra55_source_names()
        subset_path = jra55_subset_path(config, name)
        isfile(subset_path) || error("$subset_path does not exist; run `download_atmosphere` for this setup first.")
        NCDataset(subset_path) do ds
            source_longitude = Float64.(ds["lon"][:])
            source_latitude = Float64.(ds["lat"][:])
            arrays[name] = Float32.(coalesce.(ds[name][:, :, :], NaN32))
        end
    end

    longitude_weights = jra55_interpolation_weights(source_longitude, longitude)
    latitude_weights = jra55_interpolation_weights(source_latitude, latitude)

    step = Hour(24 ÷ JRA55_STEPS_PER_DAY)
    times = collect(floor(coverage[1], step):step:ceil(coverage[2], step))
    variables = sort!(collect(keys(JRA55_SOURCE_VARIABLES)))
    @info "Regridding $(length(times)) steps from $(first(times)) to $(last(times)) onto $(length(longitude)) x $(length(latitude)) cells"

    output_file = atmosphere_path(config)
    mkpath(dirname(output_file))
    rm(output_file; force = true)
    Nx, Ny = length(longitude), length(latitude)
    slab = Matrix{Float32}(undef, Nx, Ny)
    source_slab = Matrix{Float32}(undef, size(first(values(arrays)))[1:2]...)

    NCDataset(output_file, "c") do ds
        defDim(ds, "lon", Nx)
        defDim(ds, "lat", Ny)
        defDim(ds, "time", length(times))
        defVar(ds, "lon", Float64, ("lon",))[:] = longitude
        defVar(ds, "lat", Float64, ("lat",))[:] = latitude
        defVar(ds, "time", times, ("time",))

        for name in variables
            defVar(ds, name, Float32, ("lon", "lat", "time");
                chunksizes = [Nx, Ny, 1],
                deflatelevel = JRA55_DEFLATE_LEVEL,
                attrib = ["_FillValue" => NaN32; atmosphere_variable_attributes(name)])
        end

        for (time_index, date) in enumerate(times)
            record = jra55_cycle_index(date)
            for name in variables
                fill!(source_slab, 0)
                for source_name in JRA55_SOURCE_VARIABLES[name]
                    source_slab .+= view(arrays[source_name], :, :, record)
                end
                jra55_interpolate!(slab, source_slab, longitude_weights, latitude_weights)
                all(isfinite, slab) || error("Non-finite $name at $date")
                ds[name][:, :, time_index] = slab
            end
        end
    end

    return (; output_file, times, variables)
end

Atmospheres.prescribed_atmosphere(config::JRA55RYFConfig, architecture; reference_date = nothing) =
    NORA3PrescribedAtmosphere(architecture; dataset = MultiYearNORA3(config), reference_date)

Atmospheres.prescribed_radiation(config::JRA55RYFConfig, architecture; reference_date = nothing) =
    NORA3PrescribedRadiation(architecture; dataset = MultiYearNORA3(config), reference_date)

function Atmospheres.atmosphere_date_range(config::JRA55RYFConfig)
    return NCDataset(atmosphere_path(config)) do ds
        times = ds["time"][:]
        return (DateTime(first(times)), DateTime(last(times)))
    end
end

# --- Run as a script ---------------------------------------------------------------------------

# The run window the prepared file must span: the Aniva setup's `start_date` and
# `start_date + stop_time`. Re-run this script after changing either there.
const START_DATE = DateTime(2023, 1, 1)
const STOP_DATE = DateTime(2024, 1, 1)
const BATHYMETRY_FILE = "bathymetry_aniva.nc"

"""
    prepare_jra55_atmosphere_aniva()

Subset and regrid JRA55 RYF for the domain of `bathymetry_aniva.nc` over `START_DATE` to
`STOP_DATE`, writing `atmosphere_aniva.nc` and its plot under `fjord_data_root("aniva")`.
"""
function prepare_jra55_atmosphere_aniva()
    data_root = fjord_data_root("aniva")
    config = JRA55RYFConfig(data_root = data_root)

    bathymetry_file = joinpath(data_root, BATHYMETRY_FILE)
    isfile(bathymetry_file) || error("$bathymetry_file does not exist; run prepare_bathymetry_aniva.jl first.")
    target_grid = FjordSim.Grids.ImmersedBoundaryGrid(bathymetry_file, CPU(), (1, 1, 1)).underlying_grid
    @info "Domain of $(BATHYMETRY_FILE): $(x_domain(target_grid))°E, $(y_domain(target_grid))°N"

    Atmospheres.fetch_atmosphere(target_grid, config)
    result = Atmospheres.regrid_atmosphere(target_grid, config; coverage = (START_DATE, STOP_DATE))
    plot_file = plot_atmosphere(config, result.output_file)

    @info "Atmosphere file saved to $(result.output_file)"
    @info "Atmosphere plot saved to $plot_file"
    return result.output_file
end

if abspath(PROGRAM_FILE) == @__FILE__
    prepare_jra55_atmosphere_aniva()
end
