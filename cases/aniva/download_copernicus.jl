# Download the Aniva forcing subset from Copernicus Marine into <aniva data root>/copernicus.
#
#   julia --project=/home/eya/src/FjordSim.jl download_copernicus.jl
#
# Needs COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD, and the
# `copernicusmarine` client on PATH or in COPERNICUSMARINE_BIN. Existing files are skipped.
#######
##Before running this code in terminal do the following
#cd /home/eya/src/OxyDep.jl  (change eya to your username)
#export COPERNICUSMARINE_SERVICE_USERNAME='your_username'
#export COPERNICUSMARINE_SERVICE_PASSWORD='your_password'#
#export COPERNICUSMARINE_BIN=/tmp/cmt/bin/copernicusmarine
####


using FjordSim: fjord_data_root

const PRODUCT = "GLOBAL_ANALYSISFORECAST_PHY_001_024"

const LONGITUDE_RANGE = (141.0, 144.45)
const LATITUDE_RANGE = (44.0, 47.8)

# The bathymetry reaches 116 m; the next source level below it is 130.7 m, so 160 m brackets it.
const DEPTH_RANGE = (0.0, 160.0)

const START_DATETIME = "2023-01-01T00:00:00"
const END_DATETIME = "2023-12-31T23:59:59"

const OUTPUT_DIRECTORY = joinpath(fjord_data_root("aniva"), "copernicus")

# (output file, dataset id, variables)
const DOWNLOADS = (
    ("copernicus_thetao.nc", "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", ("thetao",)),
    ("copernicus_so.nc", "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", ("so",)),
    ("copernicus_cur.nc", "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", ("uo", "vo")),
    ("copernicus_zos.nc", "cmems_mod_glo_phy_anfc_0.083deg_P1D-m", ("zos",)),
)

function subset_command(client, dataset_id, variables, filename)
    command = `$client subset
        --dataset-id $dataset_id
        --minimum-longitude $(LONGITUDE_RANGE[1]) --maximum-longitude $(LONGITUDE_RANGE[2])
        --minimum-latitude $(LATITUDE_RANGE[1]) --maximum-latitude $(LATITUDE_RANGE[2])
        --minimum-depth $(DEPTH_RANGE[1]) --maximum-depth $(DEPTH_RANGE[2])
        --start-datetime $START_DATETIME --end-datetime $END_DATETIME
        --output-directory $OUTPUT_DIRECTORY --output-filename $filename
        --skip-existing --disable-progress-bar`
    for variable in variables
        command = `$command --variable $variable`
    end
    return command
end

function download_copernicus()
    haskey(ENV, "COPERNICUSMARINE_SERVICE_USERNAME") && haskey(ENV, "COPERNICUSMARINE_SERVICE_PASSWORD") ||
        error("Set COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD")
    client = get(ENV, "COPERNICUSMARINE_BIN", "copernicusmarine")
    mkpath(OUTPUT_DIRECTORY)
    for (filename, dataset_id, variables) in DOWNLOADS
        @info "Downloading $filename" PRODUCT dataset_id variables
        run(subset_command(client, dataset_id, variables, filename))
    end
    return OUTPUT_DIRECTORY
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    download_copernicus()
end

