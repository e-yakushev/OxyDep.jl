#=
Prepare Aniva's hydrophysical and OxyDep open-boundary forcing with the current FjordSim.

Run from this directory with `julia prepare_open_boundary_aniva.jl`. Direct execution activates
the sibling FjordSim.jl checkout; FJORDSIM_PROJECT can select another checkout.

Reads bathymetry_aniva.nc and copernicus_1day/{copernicus_boundary_2023,copernicus_bgc_2023}.nc
under fjord_data_root("aniva"), and writes boundaries_aniva.nc for the open southern edge.
Oxygen is 100% air saturation from boundary temperature and salinity and atmospheric pressure
in atmosphere_aniva.nc, using OceanBioME's Garcia-Gordon (1992) parameterization in mmol/m3.
Pressure is interpolated in space and time, converted from Pa to atm, and is atmospheric, not
hydrostatic: this describes equilibrium with air, not with compressed gas at depth.
HET and POM use the concentrations in aniva_oxydep.jl, and phytoplankton carbon is converted
to nitrogen with its Redfield factor. No downloads or source-file rewrites occur.

The hourly output interpolates daily means; it contains no resolved tidal forcing.
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

using FjordSim: fjord_data_root
using Oceananigans: CPU, Center
using Oceananigans.Fields: ConstantField
using OceanBioME.Models.GasExchangeModel: GarciaGordonOxygenSaturation, surface_value
using Dates: DateTime

if !isdefined(@__MODULE__, :OxyDepBoundaryDataConfig)
    include(joinpath(@__DIR__, "aniva_oxydep.jl"))
end

"""
    boundary_oxygen_saturation(temperature, salinity, pressure, grid, saturation)

Air-equilibrium oxygen in mmol/m3 for temperature in degrees Celsius, practical salinity,
and atmospheric pressure in Pa. Uses the same pressure scaling as OceanBioME gas exchange.
"""
function boundary_oxygen_saturation(temperature, salinity, pressure, grid, saturation)
    fields = (T = ConstantField(temperature), S = ConstantField(salinity))
    concentration = surface_value(saturation, 1, 1, grid, nothing, fields)
    return concentration * (pressure / 101325)
end

function boundary_pressure_slab(dataset, dates, date, longitude_weights, latitude_weights)
    first(dates) <= date <= last(dates) || error("Atmospheric pressure does not cover $date")
    lower = clamp(searchsortedlast(dates, date), 1, length(dates) - 1)
    upper = lower + 1
    weight = Dates.value(date - dates[lower]) / Dates.value(dates[upper] - dates[lower])
    variable = dataset["air_pressure_at_sea_level"]
    slab = (1 - weight) .* coalesce.(variable[:, :, lower], NaN32) .+
           weight .* coalesce.(variable[:, :, upper], NaN32)
    pressure = Matrix{Float32}(undef, length(first(longitude_weights)), length(first(latitude_weights)))
    jra55_interpolate!(pressure, slab, longitude_weights, latitude_weights)
    all(value -> isfinite(value) && value > 0, pressure) || error("Invalid atmospheric pressure at $date")
    return vec(pressure)
end

function merge_saturated_boundary_bgc!(output_file, bgc_file, atmosphere_file, config, grid)
    saturation = GarciaGordonOxygenSaturation(eltype(grid))
    return NCDataset(atmosphere_file) do atmosphere
        pressure_dates = DateTime.(atmosphere["time"][:])
        NCDataset(bgc_file) do bgc
            NCDataset(output_file, "a") do output
                dates = DateTime.(output["time"][:])
                dates == DateTime.(bgc["time"][:]) || error("The BGC boundary file is on another time axis")
                for edge in config.open_edges
                    longitude, latitude = BoundaryData.boundary_target_nodes(Val(edge), grid, Center, Center)
                    longitude_weights = jra55_interpolation_weights(atmosphere["lon"][:], longitude)
                    latitude_weights = jra55_interpolation_weights(atmosphere["lat"][:], latitude)
                    temperature_name = BoundaryData.boundary_variable_name(edge, "T")
                    salinity_name = BoundaryData.boundary_variable_name(edge, "S")
                    shape = size(output[temperature_name])[1:2]
                    for tracer in OXYDEP_TRACERS
                        name = String(tracer)
                        variable_name = BoundaryData.boundary_variable_name(edge, name)
                        attributes = ["_FillValue" => NaN32, "units" => "mmol m-3"]
                        if tracer == last(OXYDEP_TRACERS)
                            append!(attributes, [
                                "long_name" => "Oxygen concentration at 100 percent air saturation",
                                "saturation_method" => "OceanBioME Garcia-Gordon (1992), atmospheric pressure scaling",
                                "pressure_source" => basename(atmosphere_file),
                            ])
                        end
                        defVar(output, variable_name, Float32,
                            BoundaryData.boundary_dimension_names(Val(edge), name);
                            chunksizes = [shape..., 1], deflatelevel = 5, attrib = attributes)
                    end
                    for (index, date) in enumerate(dates)
                        temperature = coalesce.(output[temperature_name][:, :, index], NaN32)
                        salinity = coalesce.(output[salinity_name][:, :, index], NaN32)
                        pressure = boundary_pressure_slab(atmosphere, pressure_dates, date,
                            longitude_weights, latitude_weights)
                        wet = isfinite.(temperature) .& isfinite.(salinity)
                        for tracer in OXYDEP_TRACERS
                            name = String(tracer)
                            variable_name = BoundaryData.boundary_variable_name(edge, name)
                            values = fill(NaN32, shape)
                            if haskey(COPERNICUS_BGC, name)
                                values .= COPERNICUS_BGC[name].scale .* coalesce.(bgc[variable_name][:, :, index], NaN32)
                            elseif haskey(CONSTANT_TRACERS, name)
                                values .= CONSTANT_TRACERS[name]
                            else
                                for level in axes(values, 2), column in axes(values, 1)
                                    wet[column, level] || continue
                                    values[column, level] = boundary_oxygen_saturation(
                                        temperature[column, level], salinity[column, level],
                                        pressure[column], grid, saturation)
                                end
                            end
                            all(isfinite, values[wet]) || error("Non-finite $variable_name at $date")
                            values[.!wet] .= NaN32
                            output[variable_name][:, :, index] = values
                        end
                    end
                end
            end
        end
    end
end

"""
    prepare_open_boundary_aniva(; data_root, coverage, open_edges, architecture, atmosphere_file)

Regrid the local physical and BGC files separately on their native source grids, then merge
them with oxygen at 100% air saturation. The default window is the calendar year 2023.
Only replace boundaries_aniva.nc after both regridding passes and the BGC merge succeed.
"""
function prepare_open_boundary_aniva(;
    data_root = fjord_data_root("aniva"),
    coverage = (DateTime(2023, 1, 1), DateTime(2024, 1, 1)),
    open_edges = :south,
    architecture = :auto,
    atmosphere_file = joinpath(data_root, "atmosphere_aniva.nc"),
)
    coverage[1] < coverage[2] || throw(ArgumentError("coverage must have increasing dates"))
    bathymetry_file = joinpath(data_root, "bathymetry_aniva.nc")
    physics = CopernicusBoundaryDataConfig(;
        data_root,
        output_directory = "copernicus_1day",
        open_edges,
        architecture,
    )
    source_files = (
        bathymetry_file,
        copernicus_boundary_file(physics),
        joinpath(data_root, "copernicus_1day", COPERNICUS_BGC_COMBINED_FILE),
        atmosphere_file,
    )
    for filepath in source_files
        isfile(filepath) || error("Required boundary input does not exist: $filepath")
    end
    NCDataset(atmosphere_file) do dataset
        variable = dataset["air_pressure_at_sea_level"]
        get(variable.attrib, "units", "") == "Pa" || error("Atmospheric pressure must be in Pa")
        dates = DateTime.(dataset["time"][:])
        length(dates) >= 2 && all(diff(dates) .> Dates.Millisecond(0)) ||
            error("Atmospheric pressure needs at least two strictly increasing dates")
        first(dates) <= coverage[1] && last(dates) >= coverage[2] ||
            error("Atmospheric pressure does not cover the requested boundary window")
    end
    grid = Grids.ImmersedBoundaryGrid(bathymetry_file, CPU(), (1, 1, 1), Float32)
    output_file = joinpath(data_root, "boundaries_aniva.nc")
    @warn "Daily Copernicus means are interpolated hourly; tidal variability is not restored."

    return mktempdir(data_root; prefix = "aniva_boundaries_") do directory
        config = OxyDepBoundaryDataConfig(
            physics;
            output_file = joinpath(directory, "boundaries_aniva.nc"),
            plot_file = joinpath(directory, "boundaries_aniva.png"),
        )
        result = BoundaryData.regrid_boundary_data(grid, config.hydrophysics; coverage)
        bgc = BoundaryData.regrid_boundary_data(grid, config.biogeochemistry; coverage)
        merge_saturated_boundary_bgc!(result.output_file, bgc.output_file, atmosphere_file, config, grid)
        expected = [BoundaryData.boundary_variable_name(edge, name)
                    for edge in config.open_edges
                    for name in ("T", "S", "u", "v", "eta", "ubar", "vbar", String.(OXYDEP_TRACERS)...)]
        NCDataset(result.output_file) do dataset
            all(name -> haskey(dataset, name), expected) || error("Prepared boundary variables are incomplete")
            dates = DateTime.(dataset["time"][:])
            first(dates) <= coverage[1] && last(dates) >= coverage[2] ||
                error("Prepared boundaries do not cover the requested dates")
        end
        mv(result.output_file, output_file; force = true)
        @info "Boundary forcing saved" output_file edges=config.open_edges variables=expected
        return (; result..., output_file, variables = expected)
    end
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    prepare_open_boundary_aniva()
end