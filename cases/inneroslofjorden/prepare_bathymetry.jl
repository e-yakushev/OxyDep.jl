"""
Prepare Inner Oslofjord bathymetry with the FjordSim project environment:

    julia --project=/path/to/FjordSim.jl cases/inneroslofjorden/prepare_bathymetry.jl

Writes bathymetry.nc and bathymetry.png under fjord_data_root("inneroslofjorden").
Longitude and latitude bounds are cell faces, not the endpoints of cell-center arrays.
Uses the existing Basisdata_0000_Norge_25833_Dybdedata_FGDB.gdb directory under
fjord_data_root("oslofjorden"); no database download is performed if it is missing.
"""
module InnerOslofjordBathymetry

using FjordSim: DybdedataConfig, EvenGrid, fjord_data_root
using FjordSim.Bathymetry:
    GEONORGE_DYBDEDATA_GDB, bathymetry_path, process_bathymetry, write_bathymetry_file
using FjordSim.Grids: domain_grid
using FjordSim.Plotting: plot_bathymetry
using Oceananigans: CPU, interior

function grid_config()
    longitude_cells = 63
    latitude_cells = 114
  #  z_faces = [
  #      -200.0,
  #      -180.0,
  #      -160.0,
  #      -140.0,
  #      -120.0,
  #      -100.0,
  #      -80.0,
  #      -60.0,
  #      -40.0,
  #      -24.0,
z_faces = [
        -185.0,
        -175.0,
        -165.0,
        -155.0,
        -145.0,
        -135.0,
        -125.0,
        -115.0,
        -105.0,
        -95.0,
        -85.0,
        -75.0,
        -65.0,
        -55.0,
        -45.0,
        -35.0,
        -25.0,        
        -15.0,
        -10.0,
        -7.5,
        -5.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
    ]

    return EvenGrid(
        size = (longitude_cells, latitude_cells, length(z_faces) - 1),
        halo = (7, 7, 7),
        longitude = (10.46, 10.79),
        latitude = (59.62, 59.92),
        z_faces = z_faces,
        float_type = Float32,
    )
end

function bathymetry_config()
    geodatabase_directory = joinpath(fjord_data_root("inneroslofjorden"), GEONORGE_DYBDEDATA_GDB)
    isdir(geodatabase_directory) || error("Existing Geonorge database not found: $geodatabase_directory")

    return DybdedataConfig(
        data_root = fjord_data_root("inneroslofjorden"),
        output_file = "bathymetry.nc",
        plot_file = "bathymetry.png",
        geodatabase_file = geodatabase_directory,
        raw_resolution_factor = 2,
        padding_cells = 2,
        include_contours = false,
        contour_stride = 10,
        interpolation_passes = 1,
        major_basins = 1,
        minimum_depth = 2.0,
        open_boundary_land_cells = 1,
        max_island_cells = 3,
        close_narrow_passages = true,
        spike_ratio = 0.25,
        minimum_cell_fraction = 0.2,
        max_slope_factor = 0.5,
        geonorge_cache = true,
        regrid_cache = false,
    )
end

"""
    fill_isolated_deeps!(h)

Raise every sea cell (`h < 0`) that is deeper than all of its orthogonally adjacent sea cells to
the depth of its deepest sea neighbor, and repeat until no cell changes. Such a deep is confirmed
by a single grid point — one sounding artifact or regrid overshoot — and `smooth_bathymetry_gaps!`
has no stage for it: `fill_shallow_spikes` targets the shallow case and `limit_bottom_slope` only
bounds the contrast, so a one-cell hole survives both. Capping to a neighbor's depth keeps the
value on a depth the snap stage already aligned to the vertical faces, so no new partial-cell
slivers appear. Returns the number of cells raised.
"""
function fill_isolated_deeps!(h)
    Nx, Ny = size(h)
    raised = 0
    changed = true

    while changed
        changed = false
        for i = 1:Nx, j = 1:Ny
            h[i, j] < 0 || continue

            deepest_neighbour = zero(eltype(h))
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ii, jj = i + di, j + dj
                (1 <= ii <= Nx && 1 <= jj <= Ny) || continue
                h[ii, jj] < 0 && (deepest_neighbour = min(deepest_neighbour, h[ii, jj]))
            end
            deepest_neighbour < 0 || continue

            if h[i, j] < deepest_neighbour
                h[i, j] = deepest_neighbour
                raised += 1
                changed = true
            end
        end
    end

    return raised
end

function main()
    config = bathymetry_config()
    grid = domain_grid(grid_config(), CPU())
    mkpath(dirname(bathymetry_path(config)))
    result = process_bathymetry(grid, config; edges = [:south])

    raised = fill_isolated_deeps!(interior(result.bottom_height, :, :, 1))
    @info "Raised $raised one-cell deeps to their deepest neighbour depth"
    write_bathymetry_file(result.output_file, grid, result.bottom_height)

    plot_file = plot_bathymetry(grid, result.bottom_height, config)
    @info "Prepared Inner Oslofjord bathymetry" output_file = result.output_file plot_file
    return (; result..., plot_file)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    InnerOslofjordBathymetry.main()
end