using Oceananigans
using NCDatasets
using Printf
using CairoMakie
using Statistics

include("plot_nc_stuff_iof_fine.jl")
using .plot_nc_stuff
import .plot_nc_stuff: plot_ztime, plot_bottom_depth_map!,
       compute_pressure_from_depth, compute_oxygen_saturation_fields,
       compute_bottom_index_from_O2, plot_tracer_subplot_map!, bottom_slices_at_day,
       record_horizontal_tracer, record_bottom_tracer

# ===================== GLOBAL THEME =====================
# Ensures identical fonts across all plots and animations
CairoMakie.set_theme!(
    font = "DejaVu Sans",
    fontsize = 16,
    Axis = (xlabelsize = 14, ylabelsize = 14, titlesize = 16,
            xticklabelsize = 12, yticklabelsize = 12),
    Colorbar = (labelsize = 14, ticklabelsize = 12),
)

# ===================== SETUP =====================
folder   = joinpath(homedir(), "FjordSim_results", "aniva_oxydep")
filename = joinpath(folder, "snapshots_ocean_20261118T171527.nc")

# Scales based on p99 from the given snapshot file.
# (max is often a single cell at the river mouth or OBC, so we do not use it for the color scale)
const PLOT = Dict(
    :T     => (label="T [°C]",     colorrange=(-2, 22),     colormap=Reverse(:RdYlBu),            whiteline=0.0),
    :S     => (label="S [psu]",    colorrange=(0, 35),      colormap=:viridis,                    whiteline=32.0),
    :e     => (label="e [m²/s²]",  colorrange=(0, 1.5e-3),  colormap=:viridis,                    whiteline=5e-6),
    :O2    => (label="O₂ [μM]",    colorrange=(0, 400),     colormap=:turbo,                      whiteline=67.0),
    :O2sat => (label="O₂ [%]",     colorrange=(0, 150),     colormap=:gist_stern,                 whiteline=100.0),
    :P     => (label="PHY [μM N]", colorrange=(0, 2.5),     colormap=Reverse(:cubehelix),         whiteline=0.0),
    :HET   => (label="HET [μM N]", colorrange=(0, 0.8),     colormap=Reverse(:afmhot),            whiteline=0.0),
    :NUT   => (label="NUT [μM N]", colorrange=(0, 20),      colormap=Reverse(:cherry),            whiteline=0.0),
    :DOM   => (label="DOM [μM N]", colorrange=(0, 20),      colormap=Reverse(:CMRmap),            whiteline=0.0),
    :POM   => (label="POM [μM N]", colorrange=(0, 0.6),     colormap=Reverse(:greenbrownterrain), whiteline=0.0),
    :u     => (label="u [m/s]",    colorrange=(-0.3, 0.3),  colormap=Reverse(:seismic),           whiteline=0.0),
    :v     => (label="v [m/s]",    colorrange=(-0.3, 0.3),  colormap=Reverse(:seismic),           whiteline=0.0),
)

plot_dates      = [25, 70, 115, 160, 205, 250, 295, 350]
animation_speed = 10.0
stride          = 4

# Helper function: safely extract a 2D slice and replace zeros/missing with NaN
function safe_slice(A, k, t)
    slice = coalesce.(A[:, :, k, t], NaN)
    slice[slice .== 0] .= NaN
    return slice
end

# ===================== LOAD DATA =====================
ds = NCDataset(filename, "r")

times = ds["time"][1:stride:end]
depth = ds["z_aac"][:]

T   = ds["T"][:,:,:,1:stride:end]
S   = ds["S"][:,:,:,1:stride:end]
e   = ds["e"][:,:,:,1:stride:end]
u   = ds["u"][:,:,:,1:stride:end]
v   = ds["v"][:,:,:,1:stride:end]
P   = ds["P"][:,:,:,1:stride:end]
HET = ds["HET"][:,:,:,1:stride:end]
NUT = ds["NUT"][:,:,:,1:stride:end]
POM = ds["POM"][:,:,:,1:stride:end]
DOM = ds["DOM"][:,:,:,1:stride:end]
O₂  = ds["O₂"][:,:,:,1:stride:end]

Nz = size(ds["z_aac"], 1)
data_Nx, data_Ny = size(T, 1), size(T, 2)

for (nm, A) in (("T",T), ("S",S), ("e",e), ("u",u), ("v",v),
                ("P",P), ("HET",HET), ("NUT",NUT), ("POM",POM),
                ("DOM",DOM), ("O₂",O₂))
    println("$nm stats — min: ", minimum(A), ", max: ", maximum(A))
end

# Coordinates of the box
real_lon = range(142.08, 143.45, length=data_Nx)
real_lat = range(46.03,  46.75, length=data_Ny)

# ===================== DERIVED FIELDS =====================
# Needed beforehand: used by maps, Hovmöller, and bottom animation
Pressure = compute_pressure_from_depth(T, depth, Nz)
O₂_sat_val, O₂_sat = compute_oxygen_saturation_fields(T, S, Pressure, O₂)
println("O₂_sat [%] stats — min: ", minimum(O₂_sat), ", max: ", maximum(O₂_sat))

bottom_z = compute_bottom_index_from_O2(O₂, Nz)

water_mask = falses(data_Nx, data_Ny, Nz)
for i in 1:data_Nx, j in 1:data_Ny
    bz = bottom_z[i, j]
    O₂[i, j, bz, 1] != 0 && (water_mask[i, j, bz:Nz] .= true)
end
@info "Water mask: $(sum(water_mask)) water cells out of $(data_Nx*data_Ny*Nz)"

# ===================== SURFACE ANIMATIONS =====================
@info "Surface movies…"
record_horizontal_tracer(S,   times, folder, "Ssurf",   PLOT[:S].label,
                        real_lon, real_lat; colorrange=PLOT[:S].colorrange,
                        colormap=PLOT[:S].colormap, iz=Nz, speed=animation_speed)
record_horizontal_tracer(NUT, times, folder, "NUTsurf", PLOT[:NUT].label,
                        real_lon, real_lat; colorrange=PLOT[:NUT].colorrange,
                        colormap=PLOT[:NUT].colormap, iz=Nz, speed=animation_speed)
record_horizontal_tracer(DOM, times, folder, "DOMsurf", PLOT[:DOM].label,
                        real_lon, real_lat; colorrange=PLOT[:DOM].colorrange,
                        colormap=PLOT[:DOM].colormap, iz=Nz, speed=animation_speed)
record_horizontal_tracer(O₂,  times, folder, "O2surf",  PLOT[:O2].label,
                        real_lon, real_lat; colorrange=PLOT[:O2].colorrange,
                        colormap=PLOT[:O2].colormap, iz=Nz, speed=animation_speed)
record_horizontal_tracer(P,   times, folder, "PHYsurf", PLOT[:P].label,
                        real_lon, real_lat; colorrange=PLOT[:P].colorrange,
                        colormap=PLOT[:P].colormap, iz=Nz, speed=animation_speed)

# ===================== BOTTOM OXYGEN ANIMATION =====================
@info "Bottom O₂ movie…"
record_bottom_tracer(O₂, "O2_bottom", Nz, times, folder, real_lon, real_lat;
                    colorrange=PLOT[:O2].colorrange, colormap=PLOT[:O2].colormap,
                    figsize=(400, 550), speed=animation_speed)

# ===================== HOVMÖLLER (VERTICAL–TIME) =====================
@info "Hovmöller plots…"
# (i, j) = (75, 110) Lososey Bay 
plot_ztime(NUT, O₂, O₂_sat, P, HET, T, DOM, POM, S, 100,  80, times, depth, folder)
# (i, j) = (150, 50) southeast
plot_ztime(NUT, O₂, O₂_sat, P, HET, T, DOM, POM, S, 100, 130, times, depth, folder)

# ===================== SURFACE MAPS + BOTTOM MAPS =====================
fig_width  = 1000
fig_height = 1150
depth_indexes = [Nz]     # surface

for plot_day in plot_dates
    day_index = plot_day * round(Int, length(times) / 365)

    # --- surface maps ---
    for depth_index in depth_indexes
        @info "Surface map: day $plot_day, iz=$depth_index"
        T_slice   = safe_slice(T,   depth_index, day_index)
        S_slice   = safe_slice(S,   depth_index, day_index)
        O₂_slice  = safe_slice(O₂,  depth_index, day_index)
        NUT_slice = safe_slice(NUT, depth_index, day_index)
        P_slice   = safe_slice(P,   depth_index, day_index)
        HET_slice = safe_slice(HET, depth_index, day_index)
        O₂_sat_slice = safe_slice(O₂_sat, depth_index, day_index)
        DOM_slice = safe_slice(DOM, depth_index, day_index)
        POM_slice = safe_slice(POM, depth_index, day_index)

        fig = Figure(size=(fig_width, fig_height))
        plot_tracer_subplot_map!(fig, (1, 1), T_slice,   PLOT[:T].label,   real_lon, real_lat;
            colorrange=PLOT[:T].colorrange,   colormap=PLOT[:T].colormap,   whiteline=PLOT[:T].whiteline)
        plot_tracer_subplot_map!(fig, (1, 3), S_slice,   PLOT[:S].label,   real_lon, real_lat;
            colorrange=PLOT[:S].colorrange,   colormap=PLOT[:S].colormap,   whiteline=PLOT[:S].whiteline)
        plot_tracer_subplot_map!(fig, (1, 5), O₂_slice,  PLOT[:O2].label,  real_lon, real_lat;
            colorrange=PLOT[:O2].colorrange,  colormap=PLOT[:O2].colormap,  whiteline=PLOT[:O2].whiteline)
        plot_tracer_subplot_map!(fig, (2, 1), P_slice,   PLOT[:P].label,   real_lon, real_lat;
            colorrange=PLOT[:P].colorrange,   colormap=PLOT[:P].colormap,   whiteline=PLOT[:P].whiteline)
        plot_tracer_subplot_map!(fig, (2, 3), HET_slice, PLOT[:HET].label, real_lon, real_lat;
            colorrange=PLOT[:HET].colorrange, colormap=PLOT[:HET].colormap, whiteline=PLOT[:HET].whiteline)
        plot_tracer_subplot_map!(fig, (2, 5), O₂_sat_slice, PLOT[:O2sat].label, real_lon, real_lat;
            colorrange=PLOT[:O2sat].colorrange, colormap=PLOT[:O2sat].colormap, whiteline=PLOT[:O2sat].whiteline)
        plot_tracer_subplot_map!(fig, (3, 1), DOM_slice, PLOT[:DOM].label, real_lon, real_lat;
            colorrange=PLOT[:DOM].colorrange, colormap=PLOT[:DOM].colormap, whiteline=PLOT[:DOM].whiteline)
        plot_tracer_subplot_map!(fig, (3, 3), POM_slice, PLOT[:POM].label, real_lon, real_lat;
            colorrange=PLOT[:POM].colorrange, colormap=PLOT[:POM].colormap, whiteline=PLOT[:POM].whiteline)
        plot_tracer_subplot_map!(fig, (3, 5), NUT_slice, PLOT[:NUT].label, real_lon, real_lat;
            colorrange=PLOT[:NUT].colorrange, colormap=PLOT[:NUT].colormap, whiteline=PLOT[:NUT].whiteline)

        save(joinpath(folder, "map_iz_$(depth_index)_day_$(plot_day).png"), fig)
    end

    # --- bottom maps (including O₂ and O₂ %) ---
    @info "Bottom map: day $plot_day"
    T_bot, S_bot, O₂_bot, P_bot, HET_bot, O₂_sat_bot, DOM_bot, POM_bot, NUT_bot =
        bottom_slices_at_day(T, S, O₂, P, HET, O₂_sat, DOM, POM, NUT, bottom_z, day_index)

    fig_b = Figure(size=(fig_width, fig_height))
    plot_tracer_subplot_map!(fig_b, (1, 1), T_bot,       PLOT[:T].label,     real_lon, real_lat;
        colorrange=PLOT[:T].colorrange,     colormap=PLOT[:T].colormap,     whiteline=PLOT[:T].whiteline)
    plot_tracer_subplot_map!(fig_b, (1, 3), S_bot,       PLOT[:S].label,     real_lon, real_lat;
        colorrange=PLOT[:S].colorrange,     colormap=PLOT[:S].colormap,     whiteline=PLOT[:S].whiteline)
    plot_tracer_subplot_map!(fig_b, (1, 5), O₂_bot,      PLOT[:O2].label,    real_lon, real_lat;
        colorrange=PLOT[:O2].colorrange,    colormap=PLOT[:O2].colormap,    whiteline=PLOT[:O2].whiteline)
    plot_tracer_subplot_map!(fig_b, (2, 1), P_bot,       PLOT[:P].label,     real_lon, real_lat;
        colorrange=PLOT[:P].colorrange,     colormap=PLOT[:P].colormap,     whiteline=PLOT[:P].whiteline)
    plot_tracer_subplot_map!(fig_b, (2, 3), HET_bot,     PLOT[:HET].label,   real_lon, real_lat;
        colorrange=PLOT[:HET].colorrange,   colormap=PLOT[:HET].colormap,   whiteline=PLOT[:HET].whiteline)
    plot_tracer_subplot_map!(fig_b, (2, 5), O₂_sat_bot,  PLOT[:O2sat].label, real_lon, real_lat;
        colorrange=PLOT[:O2sat].colorrange, colormap=PLOT[:O2sat].colormap, whiteline=PLOT[:O2sat].whiteline)
    plot_tracer_subplot_map!(fig_b, (3, 1), DOM_bot,     PLOT[:DOM].label,   real_lon, real_lat;
        colorrange=PLOT[:DOM].colorrange,   colormap=PLOT[:DOM].colormap,   whiteline=PLOT[:DOM].whiteline)
    plot_tracer_subplot_map!(fig_b, (3, 3), POM_bot,     PLOT[:POM].label,   real_lon, real_lat;
        colorrange=PLOT[:POM].colorrange,   colormap=PLOT[:POM].colormap,   whiteline=PLOT[:POM].whiteline)
    plot_tracer_subplot_map!(fig_b, (3, 5), NUT_bot,     PLOT[:NUT].label,   real_lon, real_lat;
        colorrange=PLOT[:NUT].colorrange,   colormap=PLOT[:NUT].colormap,   whiteline=PLOT[:NUT].whiteline)

    save(joinpath(folder, "map_bottom_day_$(plot_day).png"), fig_b)
end

# ===================== BOTTOM DEPTH MAP =====================
@info "Bottom depth map…"
fig_depth = Figure(size=(1200, 1000))
plot_bottom_depth_map!(fig_depth, (1, 1), bottom_z, depth;
                       title_str="Bottom depth (m)", use_abs=true,
                       colormap=Reverse(:oslo25), whiteline=0.0)
save(joinpath(folder, "Bottom_depth_map.png"), fig_depth)

# ===================== VOLUME-AVERAGED TIMESERIES =====================
@info "Volume averages…"
strided_range = 1:stride:length(ds["time"])
nt            = length(times)
dt_days       = 6 / 24 * stride
times_days    = (0:nt-1) .* dt_days

Int_S   = zeros(nt); Int_T   = zeros(nt); Int_e   = zeros(nt)
Int_NUT = zeros(nt); Int_P   = zeros(nt); Int_HET = zeros(nt)
Int_POM = zeros(nt); Int_DOM = zeros(nt); Int_O2  = zeros(nt)

for (i, raw_t) in enumerate(strided_range)
    Int_S[i]   = mean(ds["S"][:,:,:,raw_t][water_mask])
    Int_T[i]   = mean(ds["T"][:,:,:,raw_t][water_mask])
    Int_e[i]   = mean(ds["e"][:,:,:,raw_t][water_mask])
    Int_NUT[i] = mean(ds["NUT"][:,:,:,raw_t][water_mask])
    Int_P[i]   = mean(ds["P"][:,:,:,raw_t][water_mask])
    Int_HET[i] = mean(ds["HET"][:,:,:,raw_t][water_mask])
    Int_POM[i] = mean(ds["POM"][:,:,:,raw_t][water_mask])
    Int_DOM[i] = mean(ds["DOM"][:,:,:,raw_t][water_mask])
    Int_O2[i]  = mean(ds["O₂"][:,:,:,raw_t][water_mask])
end

fig = Figure(size=(800, 800))
ax1 = Axis(fig[1, 1], xlabel="Time (days)", ylabel="T")
ax2 = Axis(fig[1, 2], xlabel="Time (days)", ylabel="S")
ax3 = Axis(fig[1, 3], xlabel="Time (days)", ylabel="e")
ax4 = Axis(fig[2, 1], xlabel="Time (days)", ylabel="P")
ax5 = Axis(fig[2, 2], xlabel="Time (days)", ylabel="HET")
ax6 = Axis(fig[2, 3], xlabel="Time (days)", ylabel="POM")
ax7 = Axis(fig[3, 1], xlabel="Time (days)", ylabel="DOM")
ax8 = Axis(fig[3, 2], xlabel="Time (days)", ylabel="NUT")
ax9 = Axis(fig[3, 3], xlabel="Time (days)", ylabel="N-total")

lines!(ax1, times_days, Int_T, linewidth=2, color=:cyan)
lines!(ax2, times_days, Int_S, linewidth=2, color=:magenta)
lines!(ax3, times_days, Int_e, linewidth=2, color=:blue)
lines!(ax4, times_days, Int_P,   linewidth=2, color=:green)
lines!(ax5, times_days, Int_HET, linewidth=2, color=:orange)
lines!(ax6, times_days, Int_POM, linewidth=2, color=:blue)
lines!(ax7, times_days, Int_DOM, linewidth=2, color=:purple)
lines!(ax8, times_days, Int_NUT, linewidth=2, color=:red)
lines!(ax9, times_days, Int_NUT .+ Int_P .+ Int_HET .+ Int_POM .+ Int_DOM, linewidth=2, color=:black)
lines!(ax9, times_days, Int_NUT,                                              linewidth=2, color=:red)
lines!(ax9, times_days, Int_NUT .+ Int_P,                                     linewidth=2, color=:green)
lines!(ax9, times_days, Int_NUT .+ Int_P .+ Int_HET,                          linewidth=2, color=:orange)
lines!(ax9, times_days, Int_NUT .+ Int_P .+ Int_HET .+ Int_POM,               linewidth=2, color=:blue)

CairoMakie.ylims!(ax3, 0, nothing)
CairoMakie.ylims!(ax4, 0, nothing)
CairoMakie.ylims!(ax5, 0, nothing)
CairoMakie.ylims!(ax6, 0, nothing)
CairoMakie.ylims!(ax7, 0, nothing)
CairoMakie.ylims!(ax8, 0, nothing)
CairoMakie.ylims!(ax9, 0, nothing)

save(joinpath(folder, "vol_averaged_timeseries.png"), fig)

# ===================== MONTHLY / ANNUAL TABLE =====================
@info "Monthly/annual table…"
month_days    = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
month_cumdays = cumsum(month_days)

function day_to_month(d)
    d_mod = mod(d, 365)
    for m in 1:12
        d_mod < month_cumdays[m] && return m
    end
    return 12
end

timestep_month = [day_to_month(times_days[t]) for t in 1:nt]

avg_vars = Pair{String, Vector{Float64}}[
    "T"     => Int_T,
    "S"     => Int_S,
    "e*1e6" => Int_e .* 1e6,
    "NUT"   => Int_NUT,
    "P"     => Int_P,
    "HET"   => Int_HET,
    "POM"   => Int_POM,
    "DOM"   => Int_DOM,
    "O2"    => Int_O2,
]

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
header = @sprintf("%-10s", "Variable") *
         join([@sprintf("%10s", mn) for mn in month_names]) *
         @sprintf("%10s", "Annual")

table_file = joinpath(folder, "monthly_annual_averages.txt")
open(table_file, "w") do io
    println(io, "Volume-averaged monthly and annual means")
    println(io, "="^length(header))
    println(io, header)
    println(io, "-"^length(header))
    for (varname, vardata) in avg_vars
        row = @sprintf("%-10s", varname)
        for m in 1:12
            idx = findall(timestep_month .== m)
            row *= isempty(idx) ? @sprintf("%10s", "N/A") : @sprintf("%10.5f", mean(vardata[idx]))
        end
        row *= @sprintf("%10.5f", mean(vardata))
        println(io, row)
    end
    println(io, "="^length(header))
end
@info "Table saved to $table_file"

close(ds)
@info "Script completed!"