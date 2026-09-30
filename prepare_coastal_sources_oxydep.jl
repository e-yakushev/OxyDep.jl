using NCDatasets
using DelimitedFiles

const SOURCE_FORCING = "/home/eya/FjordSim_data/inneroslofjorden/forcing_oxydep.nc"
const OUTPUT_FORCING = "/home/eya/FjordSim_data/inneroslofjorden/forcing_rivers_oxydep.nc"
const RIVERS_DIRECTORY = "/home/eya/FjordSim_data/inneroslofjorden/Rivers"
const EARTH_RADIUS = 6.371e6
const LONGITUDE_RANGE = (10.46, 10.79)
const LATITUDE_RANGE = (59.62, 59.92)

const TRACER_VARIABLES = (
    ("NUT", "N3_n"),
    ("POM", "PON0"),
    ("DOM", "R2_n"),
    ("S", "S"),
    ("T", "temp"),
)

struct RiverOutlet
    id::Int
    tracer_j::Int
    tracer_i::Int
    component::Symbol
    face_j::Int
    face_i::Int
    velocity_lambda::Float64
    layers::Int
end

const RIVER_OUTLETS = (
    RiverOutlet(12, 60, 52, :u, 60, 53, 2.0, 3),
    RiverOutlet(13, 107, 56, :u, 107, 57, 2.0, 3),
    RiverOutlet(14, 110, 35, :v, 111, 35, -2.0, 3),
    RiverOutlet(15, 102, 14, :u, 103, 14, 2.0, 3),
    RiverOutlet(16, 32, 12, :u, 33, 12, 2.0, 3),
)

river_csv_path(id, suffix) = joinpath(
    RIVERS_DIRECTORY, "river_$(suffix)_from_$(id)_year_2020.csv",
)

function read_river_column(id, suffix, column)
    table, _ = readdlm(river_csv_path(id, suffix), ';'; header = true)
    return Float64.(table[:, column])
end

read_discharge(id) = read_river_column(id, "N3_n", 2)
read_concentration(id, suffix) = read_river_column(id, suffix, 3)
pad_to_forcing_time(daily) = vcat(daily[1], daily, daily[end])

function cell_area(ds, j)
    Δλ = deg2rad(LONGITUDE_RANGE[2] - LONGITUDE_RANGE[1]) / ds.dim["Nx"]
    Δφ = (LATITUDE_RANGE[2] - LATITUDE_RANGE[1]) / ds.dim["Ny"]
    south = deg2rad(LATITUDE_RANGE[1] + (j - 1) * Δφ)
    north = deg2rad(LATITUDE_RANGE[1] + j * Δφ)
    return EARTH_RADIUS^2 * Δλ * (sin(north) - sin(south))
end

# The plume layers are 1 m thick, so the layer volume equals the cell area numerically.
tracer_lambda_per_discharge(ds, outlet) = 1 / cell_area(ds, outlet.tracer_j)

function write_outlet!(ds, outlet, flux)
    n_time = size(ds["NUT"], 4)
    per_layer_flux = pad_to_forcing_time(flux ./ outlet.layers)
    tracer_lambda_series = tracer_lambda_per_discharge(ds, outlet) .* abs.(per_layer_flux)
    @assert length(per_layer_flux) == n_time

    n_surface = size(ds["NUT"], 3)
    top_levels = (n_surface - outlet.layers + 1):n_surface
    velocity_variable = outlet.component === :u ? "u" : "v"
    velocity_lambda_variable = velocity_variable * "_lambda"

    for (variable, suffix) in TRACER_VARIABLES
        series = pad_to_forcing_time(read_concentration(outlet.id, suffix))
        for k in top_levels
            ds[variable][outlet.tracer_i, outlet.tracer_j, k, :] = series
            ds[variable * "_lambda"][outlet.tracer_i, outlet.tracer_j, k, :] = tracer_lambda_series
        end
    end

    for k in top_levels
        ds[velocity_variable][outlet.face_i, outlet.face_j, k, :] = per_layer_flux
        ds[velocity_lambda_variable][outlet.face_i, outlet.face_j, k, :] .= outlet.velocity_lambda
    end
end

function main()
    isfile(SOURCE_FORCING) || error(
        "$SOURCE_FORCING does not exist. Run `julia --project=/home/eya/src/FjordSim.jl " *
        "-m FjordSim prepare_forcing --config /home/eya/src/OxyDep.jl/inneroslofjorden_oxydep.jl` first.",
    )

    cp(SOURCE_FORCING, OUTPUT_FORCING; force = true)
    NCDataset(OUTPUT_FORCING, "a") do ds
        for outlet in RIVER_OUTLETS
            write_outlet!(ds, outlet, read_discharge(outlet.id))
            @info "Wrote river $(outlet.id) at cell (j=$(outlet.tracer_j), i=$(outlet.tracer_i))"
        end
    end
    @info "OxyDep forcing with rivers 12-16 saved to $OUTPUT_FORCING"
end

main()