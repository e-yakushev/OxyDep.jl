# Land-masks the cells of aniva_bathymetry.nc shallower than MINIMUM_DEPTH, and any NaN, into bathymetry.nc.
#
#   julia --project=/home/eya/src/FjordSim.jl prepare_bathymetry_aniva.jl
#
# The raw file has wet cells down to a few millimetres, which the 1 m surface layer cannot hold: wind
# stress on such a cell accelerates it past stability within days. Re-run prepare_forcing_aniva_oxydep.jl
# after changing MINIMUM_DEPTH, since the forcing is regridded with this land mask.

using FjordSim: fjord_data_root
using NCDatasets

const MINIMUM_DEPTH = 2.0
const SOURCE_FILE = "aniva_bathymetry.nc"
const OUTPUT_FILE = "bathymetry.nc"

function prepare_bathymetry_aniva()
    directory = fjord_data_root("aniva")
    output_path = joinpath(directory, OUTPUT_FILE)
    rm(output_path; force = true)

    NCDataset(joinpath(directory, SOURCE_FILE)) do source
        NCDataset(output_path, "c") do output
            for (name, length) in source.dim
                defDim(output, name, length)
            end
            for (name, value) in source.attrib
                output.attrib[name] = value
            end
            output.attrib["history"] = "Cells shallower than $MINIMUM_DEPTH m and NaN set to land (h = 0)"

            for name in ("lon", "lat", "z_faces")
                defVar(output, name, source[name][:], dimnames(source[name]); attrib = source[name].attrib)
            end

            bottom_height = Float64.(coalesce.(source["h"][:, :], NaN))
            land = .!isfinite.(bottom_height) .| (bottom_height .> -MINIMUM_DEPTH)
            @info "Masking $(count(land .& (bottom_height .< 0))) wet cells shallower than $MINIMUM_DEPTH m"
            bottom_height[land] .= 0
            defVar(output, "h", bottom_height, dimnames(source["h"]); attrib = source["h"].attrib)
        end
    end
    return output_path
end

prepare_bathymetry_aniva()
