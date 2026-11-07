# Regrids the Aniva Bay - La Perouse bathymetry of `misc/aniva_bathymetry_Arkhipkin.nc` onto a
# ~500 m model grid over 142.08-143.45°E, 46.03-46.75°N, and writes it in the layout FjordSim's
# `simulation_grid` reads: `lon` and `lat` cell centres, `z_faces`, and `h(lon, lat)` as bottom
# height (negative below sea level, 0 on land).
#
#   julia --project=/home/eya/src/FjordSim.jl prepare_bathymetry_aniva.jl
#
# 0.72° of latitude is 80.1 km, so 160 rows give Δy = 500 m; 1.37° of longitude is 105.1 km at the
# box's mid-latitude 46.39°N, so 210 columns give Δx = 504 m on the southern edge and 497 m on the
# northern. The model reads the grid size and box from this file, so a setup using it states the
# same `size`, `longitude`, `latitude` and `z_faces` in its `EvenGrid`.

using FjordSim: fjord_data_root
using NCDatasets
using CairoMakie: Figure, Axis, heatmap!, Colorbar, save

const SOURCE_FILE = joinpath("misc", "aniva_bathymetry_Arkhipkin.nc")
const OUTPUT_FILE = "bathymetry_aniva.nc"
const PLOT_FILE = "bathymetry_aniva.png"

const LONGITUDE = (142.08, 143.45)
const LATITUDE = (46.03, 46.75)
const SIZE = (210, 160)

const Z_FACES = [
    -200.0, -180.0, -160.0, -140.0, -120.0, -100.0, -80.0, -60.0,
    -40.0, -24.0, -15.0, -10.0, -7.5, -5.0, -3.0, -2.0, -1.0, 0.0,
]

# Cells shallower than this become land: the 1 m surface layer cannot hold a wet cell of a few
# centimetres, and wind stress on one accelerates it past stability within days.
const MINIMUM_DEPTH = 2.0

cell_centers(range, n) = [range[1] + (i - 1 // 2) * (range[2] - range[1]) / n for i in 1:n]

"""
    bracket(axis, x)

The index `i` with `axis[i] <= x <= axis[i + 1]` on the increasing `axis`, and the weight of
`axis[i + 1]` in the linear interpolation at `x`; `nothing` when `x` lies outside `axis`.
"""
function bracket(axis, x)
    first(axis) <= x <= last(axis) || return nothing
    i = clamp(searchsortedlast(axis, x), 1, length(axis) - 1)
    return i, (x - axis[i]) / (axis[i + 1] - axis[i])
end

"""
    interpolated_bottom_height(source_longitude, source_latitude, source_height, λ, φ)

Bottom height at (`λ`, `φ`) when no source node falls inside the model cell: wet where the nearest
source node is wet, with the bilinear interpolation of the wet nodes around it, so the coast does
not average land's zero into the first sea cells. `nothing` outside the source.
"""
function interpolated_bottom_height(source_longitude, source_latitude, source_height, λ, φ)
    x = bracket(source_longitude, λ)
    y = bracket(source_latitude, φ)
    (isnothing(x) || isnothing(y)) && return nothing
    (si, wx), (sj, wy) = x, y
    nearest_i = wx < 1 // 2 ? si : si + 1
    nearest_j = wy < 1 // 2 ? sj : sj + 1
    source_height[nearest_i, nearest_j] < 0 || return 0.0

    weights = ((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy)
    corners = ((si, sj), (si + 1, sj), (si, sj + 1), (si + 1, sj + 1))
    total = 0.0
    weighted = 0.0
    for (weight, (ci, cj)) in zip(weights, corners)
        source_height[ci, cj] < 0 || continue
        total += weight
        weighted += weight * source_height[ci, cj]
    end
    return weighted / total
end

"""
    regrid_bottom_height(source_longitude, source_latitude, source_height, longitude, latitude)

Bottom height at every (`longitude`, `latitude`) model cell centre, and the number of cells the
source does not cover, which are left as land.

A source finer than the model has several nodes in each cell: the cell is wet when at least half of
them are, at the mean of its wet nodes. A cell holding no node falls back to
`interpolated_bottom_height`.
"""
function regrid_bottom_height(source_longitude, source_latitude, source_height, longitude, latitude)
    Δλ = (LONGITUDE[2] - LONGITUDE[1]) / SIZE[1]
    Δφ = (LATITUDE[2] - LATITUDE[1]) / SIZE[2]
    height = zeros(length(longitude), length(latitude))
    uncovered = 0

    for (j, φ) in enumerate(latitude), (i, λ) in enumerate(longitude)
        columns = searchsortedfirst(source_longitude, λ - Δλ / 2):searchsortedlast(source_longitude, λ + Δλ / 2)
        rows = searchsortedfirst(source_latitude, φ - Δφ / 2):searchsortedlast(source_latitude, φ + Δφ / 2)

        if isempty(columns) || isempty(rows)
            value = interpolated_bottom_height(source_longitude, source_latitude, source_height, λ, φ)
            if isnothing(value)
                uncovered += 1
            else
                height[i, j] = value
            end
            continue
        end

        nodes = view(source_height, columns, rows)
        wet = count(<(0), nodes)
        2wet >= length(nodes) || continue
        height[i, j] = sum(value for value in nodes if value < 0) / wet
    end

    return height, uncovered
end

"""
    largest_water_body(wet)

The mask of the largest 4-connected component of `wet`. A box cut from a larger domain can leave bays
or single cells connected to the rest only outside it, and those have no exchange with the model sea.
"""
function largest_water_body(wet)
    labels = zeros(Int, size(wet))
    sizes = Int[]
    stack = CartesianIndex{2}[]
    steps = (CartesianIndex(1, 0), CartesianIndex(-1, 0), CartesianIndex(0, 1), CartesianIndex(0, -1))

    for start in CartesianIndices(wet)
        (wet[start] && labels[start] == 0) || continue
        push!(sizes, 0)
        label = length(sizes)
        labels[start] = label
        push!(stack, start)
        while !isempty(stack)
            cell = pop!(stack)
            sizes[label] += 1
            for step in steps
                neighbour = cell + step
                checkbounds(Bool, wet, neighbour) || continue
                (wet[neighbour] && labels[neighbour] == 0) || continue
                labels[neighbour] = label
                push!(stack, neighbour)
            end
        end
    end

    isempty(sizes) && error("The regridded bathymetry has no water")
    return labels .== argmax(sizes)
end

function prepare_bathymetry_aniva()
    directory = fjord_data_root("aniva")
    longitude = cell_centers(LONGITUDE, SIZE[1])
    latitude = cell_centers(LATITUDE, SIZE[2])

    source_path = joinpath(directory, SOURCE_FILE)
    isfile(source_path) || error("Source bathymetry $source_path does not exist")
    source_longitude, source_latitude, source_height = NCDataset(source_path) do source
        height = Float64.(coalesce.(source["h"][:, :], NaN))
        wet_values = filter(value -> isfinite(value) && value != 0, height)
        # A source storing positive depth rather than bottom height.
        if !isempty(wet_values) && all(>(0), wet_values)
            height .= .-height
        end
        # NaN is land or no data, and the source distinguishes neither: both become land, and so
        # does anything at or above sea level.
        height[.!isfinite.(height) .| (height .>= 0)] .= 0
        nodes_longitude = Float64.(source["lon"][:])
        nodes_latitude = Float64.(source["lat"][:])
        if !issorted(nodes_longitude)
            order = sortperm(nodes_longitude)
            nodes_longitude, height = nodes_longitude[order], height[order, :]
        end
        if !issorted(nodes_latitude)
            order = sortperm(nodes_latitude)
            nodes_latitude, height = nodes_latitude[order], height[:, order]
        end
        return nodes_longitude, nodes_latitude, height
    end
    @info "Source $(basename(source_path)): $(length(source_longitude)) x $(length(source_latitude)) " *
          "nodes over $(first(source_longitude))-$(last(source_longitude))°E, " *
          "$(first(source_latitude))-$(last(source_latitude))°N"

    height, uncovered = regrid_bottom_height(source_longitude, source_latitude, source_height, longitude, latitude)
    uncovered > 0 && @warn "$uncovered model cells lie outside the source and are left as land"
    height = max.(height, first(Z_FACES))

    shallow = (height .< 0) .& (height .> -MINIMUM_DEPTH)
    height[shallow] .= 0
    connected = largest_water_body(height .< 0)
    disconnected = (height .< 0) .& .!connected
    height[disconnected] .= 0

    wet = height .< 0
    @info "Aniva bathymetry: $(SIZE[1]) x $(SIZE[2]) cells, $(count(wet)) wet; " *
          "$(count(shallow)) cells shallower than $MINIMUM_DEPTH m and " *
          "$(count(disconnected)) cells outside the largest water body set to land; " *
          "deepest $(-minimum(height)) m"
    @info "Wet cells on the edges (open-edge candidates): " *
          "west $(count(wet[1, :])), east $(count(wet[end, :])), " *
          "south $(count(wet[:, 1])), north $(count(wet[:, end]))"

    output_path = joinpath(directory, OUTPUT_FILE)
    rm(output_path; force = true)
    NCDataset(output_path, "c") do output
        output.attrib["title"] = "Aniva Bay - La Perouse model bathymetry"
        output.attrib["source"] = SOURCE_FILE
        output.attrib["history"] =
            "Mean of the wet source nodes in each of $(SIZE[1]) x $(SIZE[2]) cells, wet where at " *
            "least half the nodes are; " *
            "capped at $(first(Z_FACES)) m; cells shallower than $MINIMUM_DEPTH m and outside the " *
            "largest connected water body set to land (h = 0)"
        defVar(output, "lon", longitude, ("lon",);
            attrib = ["units" => "degrees_east", "long_name" => "cell centre longitude"])
        defVar(output, "lat", latitude, ("lat",);
            attrib = ["units" => "degrees_north", "long_name" => "cell centre latitude"])
        defVar(output, "z_faces", Z_FACES, ("z_faces",);
            attrib = ["units" => "m", "positive" => "up", "long_name" => "vertical model grid faces"])
        defVar(output, "h", height, ("lon", "lat");
            attrib = ["units" => "m", "positive" => "up",
                      "long_name" => "bottom height, negative below sea level, 0 on land"])
    end

    figure = Figure(size = (1100, 650))
    axis = Axis(figure[1, 1]; xlabel = "Longitude (°E)", ylabel = "Latitude (°N)",
                title = "Aniva bathymetry, $(SIZE[1]) x $(SIZE[2]) cells",
                aspect = (LONGITUDE[2] - LONGITUDE[1]) * cosd(sum(LATITUDE) / 2) / (LATITUDE[2] - LATITUDE[1]))
    depth = ifelse.(wet, -height, NaN)
    plot = heatmap!(axis, longitude, latitude, depth; colormap = :deep, nan_color = :tan)
    Colorbar(figure[1, 2], plot; label = "Depth (m)")
    save(joinpath(directory, PLOT_FILE), figure)

    return output_path
end

prepare_bathymetry_aniva()
