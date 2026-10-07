using NCDatasets, CairoMakie, Statistics

# --- Configuration ---
#input_file = joinpath(@__DIR__, "..", "data", "input", "bathymetry_63to114_v2_edited.nc")
#output_file = joinpath(@__DIR__, "..", "data", "input", "bathymetry_63to114_fine.nc")
input_file =  joinpath("/home", "eya", "FjordSim_data", "inneroslofjorden", "bathymetry.nc") #"bathymetry_63to114_v2.nc")
output_file = joinpath("/home", "eya", "FjordSim_data", "inneroslofjorden", "bathymetry_mod.nc") #"bathymetry_63to114_v2_edited.nc")

# --- Read data ---
ds = Dataset(input_file)
lat = ds["lat"][:]
lon = ds["lon"][:]
h = Array(ds["h"])  # (63, 114) matrix: h[lon_idx, lat_idx]
z_faces = ds["z_faces"][:]
close(ds)

println("Loaded bathymetry: size=$(size(h)), depth range: $(minimum(skipmissing(h))) to $(maximum(skipmissing(h)))")

# --- Plot original bathymetry ---
function plot_bathymetry(h, lon, lat; title="Bathymetry", filename=nothing)
    delta_lat = (lat[end] - lat[1]) * 60 * 1852
    delta_lon = (lon[end] - lon[1]) * 60 * 1852 * cosd(mean(lat))
    aspect_ratio = delta_lon / delta_lat
    fig_height = 2400
    fig_width = round(Int, fig_height * aspect_ratio) + 400  # extra for colorbar
    fig = Figure(size=(fig_width, fig_height))
    xticks = 5:5:size(h,1)
    yticks = 5:5:size(h,2)
    ax = Axis(fig[1, 1], title=title, xlabel="Lon index", ylabel="Lat index", aspect=DataAspect(),
              xticks=collect(xticks), yticks=collect(yticks))
    hm = heatmap!(ax, 1:size(h,1), 1:size(h,2), coalesce.(h, NaN), colormap=:CMRmap, colorrange=(-300.0, 10.0))
    Colorbar(fig[1, 2], hm, label="Depth [m]")

    # --- Gridlines every cell ---
    for x in 0:size(h,1)
        vlines!(ax, x + 0.5; color=:black, linewidth=0.3)
    end
    for y in 0:size(h,2)
        hlines!(ax, y + 0.5; color=:black, linewidth=0.3)
    end

    # --- Red gridlines every 5 gridpoints ---
    for x in 0:5:size(h,1)
        vlines!(ax, x + 0.5; color=:red, linewidth=1.0)
    end
    for y in 0:5:size(h,2)
        hlines!(ax, y + 0.5; color=:red, linewidth=1.0)
    end

    # --- Depth values in each cell ---
    fontsize = min(fig_height, fig_width) / max(size(h,1), size(h,2)) * 0.55
    for i in 1:size(h,1), j in 1:size(h,2)
        val = h[i, j]
        (ismissing(val) || val >= 0) && continue
        text!(ax, i, j; text=string(round(Int, val)), align=(:center, :center),
              fontsize=fontsize, color=:black)
    end

    if filename !== nothing
        save(filename, fig)
        println("Plot saved: $filename")
    end
    return fig
end

plot_bathymetry(h, lon, lat;
    title="Original bathymetry (63×114)",
    filename=joinpath(@__DIR__, "bathymetry_original.png"))

# --- Define edits here ---
# Format: (lon_index, lat_index) => new_depth_value
# Negative = below sea level, positive = land/above sea level, 10.0 = land mask
edits = Dict(
    # (lon_idx, lat_idx) => new_depth,
    # Example: (10, 50) => -5.0,
    # Example: (11, 50) => -8.0,
    #(50, 44) => -175.0,
    #(32, 31) => 10.0,
    #(32, 30) => 10.0,
    #(32, 29) => 10.0,
    #(32, 28) => 10.0,
    #(11, 40) => -25.0,
    #(10, 39) => -25.0,
    #(10, 40) => -25.0,
    #(11, 39) => -25.0,
"""
    (23, 100) => -25.0,
    (24, 100) => -25.0,
    (25, 100) => -20.0,
    (19, 101) => -10.0,
    (20, 101) => -15.0,
    (20, 101) => -15.0,
    (21, 101) => -20.0,
    (22, 101) => -20.0,
    (23, 101) => -20.0,
    (24, 101) => -27.0,
    (25, 101) => -25.0,
    (24, 102) => -20.0,
    (25, 102) => -25.0,
    (9, 61) => -9.0,
"""    
)

# --- Apply edits ---
if !isempty(edits)
    println("\nApplying $(length(edits)) edits:")
    for ((i, j), val) in sort(collect(edits))
        old = h[i, j]
        h[i, j] = val
        println("  h[$i, $j]: $old → $val")
    end

    # Plot edited bathymetry
    plot_bathymetry(h, lon, lat;
        title="Oslofjord bathymetry",
        filename=joinpath(@__DIR__, "bathymetry_edited.png"))

    # --- Save to new file ---
    cp(input_file, output_file; force=true)
    ds_out = Dataset(output_file, "a")
    ds_out["h"][:, :] = h
    close(ds_out)
    println("\nSaved edited bathymetry to: $output_file")
else
    println("\nNo edits defined. To edit, add entries to the `edits` Dict above.")
    println("Use the plot to identify (lon_index, lat_index) of points to change.")
    println("\nTo inspect specific values, uncomment the block below.")
end

# --- Inspect a region (uncomment and adjust as needed) ---
#println("\nh[:, 1:15]:")
#display(h[:, 1:15])
println("\nfinished:)")
