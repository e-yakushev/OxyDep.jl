using NCDatasets
using DelimitedFiles

const SOURCE_FORCING = "/home/eya/FjordSim_data/inneroslofjorden/forcing_oxydep.nc"
const OUTPUT_FORCING = "/home/eya/FjordSim_data/inneroslofjorden/forcing_rivers_oxydep.nc"
const RIVERS_DIRECTORY = "/home/eya/FjordSim_data/inneroslofjorden/Rivers"
const TRACER_LAMBDA_PER_DISCHARGE = 1 / (250 * 250) / 3
const RIVER_DISCHARGE_MULTIPLIER = 5.0

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
    lambda::Float64
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

function write_outlet!(ds, outlet, flux)
    n_time = size(ds["NUT"], 4)
    scaled_flux = RIVER_DISCHARGE_MULTIPLIER .* flux
    per_layer_flux = pad_to_forcing_time(scaled_flux ./ outlet.layers)
    lambda_series = TRACER_LAMBDA_PER_DISCHARGE .* per_layer_flux
    @assert length(per_layer_flux) == n_time

    n_surface = size(ds["NUT"], 3)
    top_levels = (n_surface - outlet.layers + 1):n_surface
    velocity_variable = outlet.component === :u ? "u" : "v"
    velocity_lambda_variable = velocity_variable * "_lambda"

    for (variable, suffix) in TRACER_VARIABLES
        series = pad_to_forcing_time(read_concentration(outlet.id, suffix))
        for k in top_levels
            ds[variable][outlet.tracer_i, outlet.tracer_j, k, :] = series
            ds[variable * "_lambda"][outlet.tracer_i, outlet.tracer_j, k, :] = lambda_series
        end
    end

    for k in top_levels
        ds[velocity_variable][outlet.face_i, outlet.face_j, k, :] = per_layer_flux
        ds[velocity_lambda_variable][outlet.face_i, outlet.face_j, k, :] .= outlet.lambda
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