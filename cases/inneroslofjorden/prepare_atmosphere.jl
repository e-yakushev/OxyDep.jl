"""
Prepare the local NORA3 atmosphere with the current FjordSim environment:

    julia --project=/path/to/FjordSim.jl cases/inneroslofjorden/prepare_atmosphere.jl

Reads nora3/NORA3_2020MM.nc under fjord_data_root("inneroslofjorden") and writes
atmosphere.nc plus atmosphere.png. Uses bathymetry.nc in the same directory for
the ocean domain extent, with 0.02-degree atmosphere spacing and 0.1-degree padding.
No downloads are performed. The source files must already contain the normalized
NORA3 variables that FjordSim's download step writes, including rotated winds,
Kelvin air temperature, and de-accumulated downwelling radiation/precipitation.

The default window is 2020-01-01 through 2021-01-01; the final hour is held for
the endpoint. Snowfall is disabled, matching the existing local files.
To replace an existing output, include this script and call
InnerOslofjordAtmosphere.main(overwrite=true). A simulation setup can use
InnerOslofjordAtmosphere.atmosphere_config() as its atmosphere_config.
"""
module InnerOslofjordAtmosphere

using Dates: DateTime
using FjordSim: Atmospheres, NORA3Config, Plotting, fjord_data_root
using Oceananigans: CPU
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid

function atmosphere_config(;
    data_root = fjord_data_root("inneroslofjorden"),
    years = [2020],
    output_file = "atmosphere.nc",
    resolution = 0.02,
    padding = 0.1,
)
    isempty(years) && error("Specify at least one atmosphere data year")
    return NORA3Config(
        data_root = data_root,
        output_directory = "nora3",
        output_file = output_file,
        plot_file = "atmosphere.png",
        resolution = resolution,
        padding = padding,
        years = sort(unique(years)),
        snowfall = false,
    )
end

function main(;
    config = atmosphere_config(),
    coverage = (DateTime(minimum(config.years), 1, 1), DateTime(maximum(config.years) + 1, 1, 1)),
    overwrite = false,
)
    source_directory = Atmospheres.atmosphere_directory(config)
    isdir(source_directory) || error("Local NORA3 directory does not exist: $source_directory")
    bathymetry_file = joinpath(config.data_root, "bathymetry.nc")
    isfile(bathymetry_file) || error("Prepare bathymetry first: $bathymetry_file")
    output_file = Atmospheres.atmosphere_path(config)
    isfile(output_file) && !overwrite &&
        error("Output already exists: $output_file. Call main(overwrite=true) to replace it.")
    grid = ImmersedBoundaryGrid(bathymetry_file, CPU(), (7, 7, 7), Float32)
    result = Atmospheres.regrid_atmosphere(grid, config; coverage)
    plot_file = Plotting.plot_atmosphere(config, result.output_file)
    @info "Prepared local NORA3 atmosphere" output_file variables = result.variables plot_file
    return (; result..., plot_file)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    InnerOslofjordAtmosphere.main()
end