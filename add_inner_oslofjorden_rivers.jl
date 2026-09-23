# Patches a copy of inneroslofjorden's forcing_npzd.nc with river discharge and nitrate for
# rivers 12-16, using their daily CSV series under
# /home/eya/FjordSim_data/inneroslofjorden/Rivers/.
#
# Ported from the "INNER OSLOFJORD RIVERS" block of prepare_coastal_sources.ipynb. That notebook
# writes two things per outlet: the water discharge into a velocity face with a `u_lambda`/
# `v_lambda` of +-2 -- the flux-through-a-face convention `FjordSim.Forcing.forcing_term_x_flux`/
# `forcing_term_y_flux` read (+2 an x-face flux, -2 a y-face flux, as opposed to the |lambda| < 1
# relaxation regime `add_rivers` itself writes) -- and each tracer's concentration into its cell
# as a relaxation, at a lambda proportional to that same discharge.
#
# Run with the FjordSim project active, e.g.:
#   julia --project=/home/eya/src/FjordSim.jl /home/eya/src/OxyDep.jl/add_inner_oslofjorden_rivers.jl

using NCDatasets
using DelimitedFiles

const SOURCE_FORCING = "/home/eya/FjordSim_data/inneroslofjorden/forcing_npzd.nc"
const OUTPUT_FORCING = "/home/eya/FjordSim_data/inneroslofjorden/forcing_rivers_npzd.nc"
const RIVERS_DIRECTORY = "/home/eya/FjordSim_data/inneroslofjorden/Rivers"

# Q/V per unit discharge for a 250m x 250m surface cell split over 3 layers, matching the
# `river_lambda` constant prepare_coastal_sources.ipynb uses for these same rivers.
const TRACER_LAMBDA_PER_DISCHARGE = 1 / (250 * 250) / 3

# Forcing variable => CSV file suffix (`river_<suffix>_from_<id>_year_2020.csv`), each carrying
# `DayOfYear;flux_m3/s;<concentration>;<discharge>`. Discharge itself is read once, from the N3_n
# file, and shared across every tracer and the velocity face -- the flux column is the same river
# in every file.
const TRACER_VARIABLES = (
    ("N", "N3_n"),    # nitrate -> NPZD nutrient
    ("D", "PON0"),    # particulate organic nitrogen -> NPZD detritus
    ("S", "S"),       # salinity
    ("T", "temp"),    # temperature
)

"""
    RiverOutlet(id, tracer_j, tracer_i, component, face_j, face_i, lambda, layers)

One river outlet: `id` names its CSV, `(tracer_j, tracer_i)` is the `(Ny, Nx)` cell its nitrate
relaxes into, `component`/`(face_j, face_i)` the velocity face its discharge is written to, and
`lambda` the `u_lambda`/`v_lambda` FjordSim reads as that face's flux direction. `layers` is how
many surface levels (`Nz`, `Nz-1`, ...) the outlet fills.
"""
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

# The notebook's 0-based (y, x) converted to this file's 1-based (j, i): tracer_j/tracer_i is
# (y + 1, x + 1); the velocity face is (y + 1, x + 2) for an eastward outlet (x + 1 there) or
# (y + 2, x + 1) for a northward one (y + 1 there). Outlets 15 and 16 write `u` at the row above
# the tracer cell rather than the face east of it, unlike 12/13 -- kept exactly as the notebook
# has it.
const RIVER_OUTLETS = (
    RiverOutlet(12, 60, 52, :u, 60, 53, 2.0, 3),
    RiverOutlet(13, 107, 56, :u, 107, 57, 2.0, 3),
    RiverOutlet(14, 110, 35, :v, 111, 35, -2.0, 3),
    RiverOutlet(15, 102, 14, :u, 103, 14, 2.0, 3),
    RiverOutlet(16, 32, 12, :u, 33, 12, 2.0, 3),
)

"""
    river_csv_path(id, suffix)

Path to river `id`'s daily CSV for `suffix` (`DayOfYear;flux_m3/s;<concentration>;<discharge>`).
"""
river_csv_path(id, suffix) = joinpath(RIVERS_DIRECTORY, "river_$(suffix)_from_$(id)_year_2020.csv")

"""
    read_river_column(id, suffix, column)

One column of river `id`'s `suffix` CSV, as a 366-entry `Vector{Float64}` for 2020-01-01 through
2020-12-31.
"""
function read_river_column(id, suffix, column)
    table, _ = readdlm(river_csv_path(id, suffix), ';'; header = true)
    return Float64.(table[:, column])
end

"""
    read_discharge(id)

River `id`'s water discharge (m3/s), from its N3_n CSV.
"""
read_discharge(id) = read_river_column(id, "N3_n", 2)

"""
    read_concentration(id, suffix)

River `id`'s concentration for the tracer whose CSV suffix is `suffix`.
"""
read_concentration(id, suffix) = read_river_column(id, suffix, 3)

"""
    pad_to_forcing_time(daily)

`daily` (366 entries, one per day of 2020) stretched to the 368-step forcing time axis by
replicating the first and last day once, matching the padding `prepare_forcing` itself applies
around the same year.
"""
pad_to_forcing_time(daily) = vcat(daily[1], daily, daily[end])

"""
    write_outlet!(ds, outlet, flux)

Write one river's discharge into its velocity face, with lambda marking the flux direction, and
each of `TRACER_VARIABLES`' concentration into its own tracer, relaxed at a lambda proportional
to that discharge, across the outlet's surface levels.
"""
function write_outlet!(ds, outlet, flux)
    n_time = size(ds["N"], 4)
    per_layer_flux = pad_to_forcing_time(flux ./ outlet.layers)
    lambda_series = TRACER_LAMBDA_PER_DISCHARGE .* per_layer_flux
    @assert length(per_layer_flux) == n_time

    n_surface = size(ds["N"], 3)
    top_levels = (n_surface-outlet.layers+1):n_surface

    velocity_variable = outlet.component === :u ? "u" : "v"
    velocity_lambda_variable = velocity_variable * "_lambda"

    for (variable, suffix) in TRACER_VARIABLES
        series = pad_to_forcing_time(read_concentration(outlet.id, suffix))
        for k in top_levels
            ds[variable][outlet.tracer_i, outlet.tracer_j, k, :] = series
            ds[variable*"_lambda"][outlet.tracer_i, outlet.tracer_j, k, :] = lambda_series
        end
    end

    for k in top_levels
        ds[velocity_variable][outlet.face_i, outlet.face_j, k, :] = per_layer_flux
        ds[velocity_lambda_variable][outlet.face_i, outlet.face_j, k, :] .= outlet.lambda
    end

    return nothing
end

function main()
    isfile(SOURCE_FORCING) || error(
        "$SOURCE_FORCING does not exist. Run `julia --project -m FjordSim prepare_forcing " *
        "--config examples/inneroslofjorden_npzd.jl` first.",
    )

    cp(SOURCE_FORCING, OUTPUT_FORCING; force = true)

    NCDataset(OUTPUT_FORCING, "a") do ds
        for outlet in RIVER_OUTLETS
            flux = read_discharge(outlet.id)
            write_outlet!(ds, outlet, flux)
            @info "Wrote river $(outlet.id) at cell (j=$(outlet.tracer_j), i=$(outlet.tracer_i))"
        end
    end

    @info "Forcing with rivers 12-16 saved to $OUTPUT_FORCING"
end

main()
