using NCDatasets, FjordSim, CairoMakie

path = joinpath(fjord_data_root("aniva"), "bathymetry_aniva.nc")
NCDataset(path) do ds
    h   = ds["h"][:, :]
    lon = ds["lon"][:]
    lat = ds["lat"][:]
    fig = Figure(size=(1000, 700))
    ax  = Axis(fig[1, 1], xlabel="lon", ylabel="lat", aspect=DataAspect(),
               title="Ztime positions — все точки должны быть на синем")
    hm  = heatmap!(ax, lon, lat, ifelse.(h .< 0, -h, NaN);
                   colormap=:deep, nan_color=:tan)
    positions = [
    ( 75, 110, "southwest"), # Near Busse Lagoon / west coast
    (150,  50, "southeast"), # Southeast part of the bay
    ]
    for (i, j, name) in positions
        scatter!(ax, lon[i], lat[j]; markersize=14, color=:red, strokewidth=1, strokecolor=:white)
        text!(ax, lon[i] + 0.015, lat[j] + 0.008; text=name, fontsize=11, color=:white)
    end
    Colorbar(fig[1, 2], hm; label="depth (m)")
    save(joinpath(homedir(), "ztime_positions_check.png"), fig)
    println("Saved: ~/ztime_positions_check.png")
end