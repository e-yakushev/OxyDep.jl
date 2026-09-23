"""
OXYgen DEPletion model, OXYDEP targests on the silmplest possible way of parameterization of the oxygen  (DO) fate in changeable redox conditions.
It has a simplified ecosystem, and simulates production of DO due to photosynthesis and consumation of DO for biota respiration,
OM mineralization, nitrification, and oxidation of reduced species of S, Mn, Fe, present in suboxic conditions.
For the details of  OxyDEP  implemented here see (Berezina et al, 2022)
Tracers
=======
OXYDEP consists of 6 state variables ( in N-units):
    P - all the phototrophic organisms (phytoplankton and bacteria).
    P grows due to photosynthesis, loses inorganic matter
    due to respiraion, and loses organic matter in dissolved (DOM) and particulate (POM)
    forms due to metabolism and mortality. P growth is limited by irradiance, temperature and NUT availability.
    Het - heterotrophs, can consume P and POM,  produce DOM and POM and respirate NUT.
    NUT - represents oxydized forms of nutrients (i.e. NO3 and NO2 for N),
    that doesn't need additional  oxygen for nitrification.
    DOM - is dissolved organic matter. DOM  includes all kinds of labile dissolved organic matter
    and reduced forms of inorganic nutrients (i.e. NH4 and Urea for N).
    POM - is particular organic matter (less labile than DOM). Temperature affects DOM and POM mineralization.
    Oxy - is dissolved oxygen.

When Ci_ == true, 5 additional Ci tracers (Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM) are added.

Required submodels
==================
* Photosynthetically available radiation: PAR (W/m²)
"""
module OXYDEPModel

export OXYDEP
export bgh_oxydep_boundary_conditions
export oxydep_sediment_forcings

using TOML
using Oceananigans: fields
using Oceananigans.Units
using Oceananigans.Fields: Field, TracerFields, CenterField, ZeroField
using Oceananigans.Forcings: DiscreteForcing
using Oceananigans.Operators: Δzᶜᶜᶜ
using Oceananigans.Grids: bottommost_active_node, Center
using Oceananigans.BoundaryConditions:
    fill_halo_regions!,
    ValueBoundaryCondition,
    FieldBoundaryConditions,
    regularize_field_boundary_conditions
using Oceananigans.Biogeochemistry: AbstractContinuousFormBiogeochemistry
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using OceanBioME:
    setup_velocity_fields, show_sinking_velocities, Biogeochemistry, ScaleNegativeTracers
using OceanBioME.Light:
    update_TwoBandPhotosyntheticallyActiveRadiation!,
    default_surface_PAR,
    TwoBandPhotosyntheticallyActiveRadiation
using OceanBioME.Sediments: sinking_flux
using Oceananigans.BoundaryConditions: FluxBoundaryCondition, ValueBoundaryCondition, FieldBoundaryConditions

import Adapt: adapt_structure, adapt
import Base: show, summary
import Oceananigans.Biogeochemistry:
    required_biogeochemical_tracers,
    required_biogeochemical_auxiliary_fields,
    biogeochemical_drift_velocity,
    update_biogeochemical_state!
import OceanBioME: redfield, conserved_tracers
import OceanBioME: maximum_sinking_velocity

# =====================================================================
# Switch: set to true to include Ci tracers and calculations, false for pure BGC
# =====================================================================
const Ci_ = false

# =====================================================================
# TOML parameter loader
# =====================================================================

"""
    load_oxydep_params(bgc_file; ci_file=nothing) -> Dict{Symbol,Any}

Read BGC parameters from `bgc_file`.  When `ci_file` is given (and Ci_ == true),
also read Ci-specific parameters from that file and merge them in.
Rate parameters are converted from 1/day to 1/s.  Sinking speeds likewise.
Sediment parameters are applied to the module-level globals via `apply_sediment_config!`.
"""
function load_oxydep_params(bgc_file::AbstractString; ci_file::Union{AbstractString,Nothing}=nothing)
    cfg = TOML.parsefile(bgc_file)
    params = Dict{Symbol,Any}()

    # BGC rate keys (stored as 1/day in TOML, need / day conversion)
    bgc_rate_keys = Set([
        :initial_photosynthetic_slope, :Max_uptake,
        :r_phy_nut, :r_phy_pom, :r_phy_dom,
        :r_phy_het, :r_pom_het, :r_het_nut, :r_het_pom,
        :r_pom_nut_oxy, :r_pom_dom, :r_dom_nut_oxy,
        :r_pom_nut_nut, :r_dom_nut_nut,
    ])

    # Pure BGC sections
    for section in ("PHY", "HET", "POM", "DOM", "O2", "stoichiometry")
        haskey(cfg, section) || continue
        for (k, v) in cfg[section]
            sym = Symbol(k)
            params[sym] = sym in bgc_rate_keys ? v / day : v
        end
    end

    # BGC sinking speeds
    if haskey(cfg, "sinking")
        ss = cfg["sinking"]
        sinking = (P = ss["P"] / day, HET = ss["HET"] / day, POM = ss["POM"] / day)
        params[:sinking_speeds] = sinking
    end

    # Sediment parameters → update module-level globals
    if haskey(cfg, "sediment")
        apply_sediment_config!(cfg["sediment"])
    end

    # Ci_ additions: load from separate ci_file when Ci_ is enabled
    if Ci_ && ci_file !== nothing
        ci_cfg = TOML.parsefile(ci_file)
        ci_rate_keys = Set([:r_ci_degrad, :r_ci_free_phy, :r_ci_food_het])
        if haskey(ci_cfg, "Ci")
            for (k, v) in ci_cfg["Ci"]
                sym = Symbol(k)
                params[sym] = sym in ci_rate_keys ? v / day : v
            end
        end
        if haskey(ci_cfg, "sinking")
            ci_ss = ci_cfg["sinking"]
            if haskey(ci_ss, "Ci_PHY")
                params[:sinking_speeds] = (params[:sinking_speeds]...,
                    Ci_PHY = ci_ss["Ci_PHY"] / day, Ci_HET = ci_ss["Ci_HET"] / day, Ci_POM = ci_ss["Ci_POM"] / day)
            end
        end
    end

    return params
end

# =====================================================================
# Surface PAR
# =====================================================================

""" Surface PAR and turbulent vertical diffusivity based on idealised mixed layer depth """
@inline PAR⁰(x, y, t) =
    60 * (1 - cos((t + 15days) * 2π / 365days)) * (1 / (1 + 0.2 * exp(-((mod(t, 365days) - 200days) / 50days)^2))) + 2

# =====================================================================
# struct, constructor, adapt_structure — split for BGC-only vs BGC+Ci_
# =====================================================================

if !Ci_

# ----- BGC-only struct -----
struct OXYDEP{FT,B,W} <: AbstractContinuousFormBiogeochemistry
    # PHY
    initial_photosynthetic_slope::FT
    Iopt::FT
    alphaI::FT
    betaI::FT
    gammaD::FT
    Max_uptake::FT
    Knut::FT
    r_phy_nut::FT
    r_phy_pom::FT
    r_phy_dom::FT
    # HET
    r_phy_het::FT
    Kphy::FT
    r_pom_het::FT
    Kpom::FT
    Uz::FT
    Hz::FT
    r_het_nut::FT
    r_het_pom::FT
    # POM
    r_pom_nut_oxy::FT
    r_pom_dom::FT
    # DOM
    r_dom_nut_oxy::FT
    # O₂
    O2_suboxic::FT
    r_pom_nut_nut::FT
    r_dom_nut_nut::FT
    OtoN::FT
    CtoN::FT
    NtoN::FT
    NtoB::FT
    # Internal
    optionals::B
    sinking_velocities::W
end

# ----- BGC-only constructor -----
function OXYDEP(grid;
    initial_photosynthetic_slope::FT = 0.1953 / day,
    Iopt::FT = 80.0,
    alphaI::FT = 1.8,
    betaI::FT = 5.2e-4,
    gammaD::FT = 0.71,
    Max_uptake::FT = 1.85 / day,
    Knut::FT = 0.8,
    r_phy_nut::FT = 0.10 / day,
    r_phy_pom::FT = 0.15 / day,
    r_phy_dom::FT = 0.17 / day,
    r_phy_het::FT = 0.8 / day,
    Kphy::FT = 0.1,
    r_pom_het::FT = 0.7 / day,
    Kpom::FT = 2.0,
    Uz::FT = 0.6,
    Hz::FT = 0.5,
    r_het_nut::FT = 0.15 / day,
    r_het_pom::FT = 0.10 / day,
    r_pom_nut_oxy::FT = 0.02 / day,
    r_pom_dom::FT = 0.10 / day,
    r_dom_nut_oxy::FT = 0.10 / day,
    O2_suboxic::FT = 20.0,
    r_pom_nut_nut::FT = 0.005 / day,
    r_dom_nut_nut::FT = 0.003 / day,
    OtoN::FT = 8.625,
    CtoN::FT = 6.625,
    NtoN::FT = 5.3,
    NtoB::FT = 0.016,
    surface_photosynthetically_active_radiation = PAR⁰,
    light_attenuation_model::LA = TwoBandPhotosyntheticallyActiveRadiation(;
        grid, surface_PAR = surface_photosynthetically_active_radiation),
    sediment_model::S = nothing,
    TS_forced::Bool = false,
    Chemicals::Bool = false,
    sinking_speeds = (P = 1.0 / day, HET = 4.0 / day, POM = 9.0 / day),
    open_bottom::Bool = true,
    scale_negatives = true,
    particles::P = nothing,
    modifiers::M = nothing,
) where {FT,LA,S,P,M}

    sinking_velocities = setup_velocity_fields(sinking_speeds, grid, open_bottom)
    optionals = Val((TS_forced, Chemicals))

    underlying_biogeochemistry = OXYDEP(
        initial_photosynthetic_slope,
        Iopt, alphaI, betaI, gammaD,
        Max_uptake, Knut,
        r_phy_nut, r_phy_pom, r_phy_dom,
        r_phy_het, Kphy, r_pom_het, Kpom,
        Uz, Hz, r_het_nut, r_het_pom,
        r_pom_nut_oxy, r_pom_dom,
        r_dom_nut_oxy,
        O2_suboxic, r_pom_nut_nut, r_dom_nut_nut,
        OtoN, CtoN, NtoN, NtoB,
        optionals, sinking_velocities,
    )

    if scale_negatives
        scaler = ScaleNegativeTracers(underlying_biogeochemistry, grid)
        modifiers = isnothing(modifiers) ? scaler : (modifiers..., scaler)
    end

    return Biogeochemistry(underlying_biogeochemistry;
        light_attenuation = light_attenuation_model,
        sediment = sediment_model, particles, modifiers)
end

# ----- BGC-only required tracers -----
required_biogeochemical_tracers(::OXYDEP{<:Any,<:Val{(false, false)},<:Any}) =
    (:NUT, :P, :HET, :POM, :DOM, :O₂, :T)

# ----- BGC-only adapt_structure -----
adapt_structure(to, o::OXYDEP) = OXYDEP(
    adapt(to, o.initial_photosynthetic_slope),
    adapt(to, o.Iopt), adapt(to, o.alphaI), adapt(to, o.betaI), adapt(to, o.gammaD),
    adapt(to, o.Max_uptake), adapt(to, o.Knut),
    adapt(to, o.r_phy_nut), adapt(to, o.r_phy_pom), adapt(to, o.r_phy_dom),
    adapt(to, o.r_phy_het), adapt(to, o.Kphy), adapt(to, o.r_pom_het), adapt(to, o.Kpom),
    adapt(to, o.Uz), adapt(to, o.Hz), adapt(to, o.r_het_nut), adapt(to, o.r_het_pom),
    adapt(to, o.r_pom_nut_oxy), adapt(to, o.r_pom_dom),
    adapt(to, o.r_dom_nut_oxy),
    adapt(to, o.O2_suboxic), adapt(to, o.r_pom_nut_nut), adapt(to, o.r_dom_nut_nut),
    adapt(to, o.OtoN), adapt(to, o.CtoN), adapt(to, o.NtoN), adapt(to, o.NtoB),
    adapt(to, o.optionals), adapt(to, o.sinking_velocities),
)

else # Ci_ == true

# ----- BGC + Ci_ struct -----
struct OXYDEP{FT,B,W} <: AbstractContinuousFormBiogeochemistry
    # PHY
    initial_photosynthetic_slope::FT
    Iopt::FT
    alphaI::FT
    betaI::FT
    gammaD::FT
    Max_uptake::FT
    Knut::FT
    r_phy_nut::FT
    r_phy_pom::FT
    r_phy_dom::FT
    # HET
    r_phy_het::FT
    Kphy::FT
    r_pom_het::FT
    Kpom::FT
    Uz::FT
    Hz::FT
    r_het_nut::FT
    r_het_pom::FT
    # POM
    r_pom_nut_oxy::FT
    r_pom_dom::FT
    # DOM
    r_dom_nut_oxy::FT
    # O₂
    O2_suboxic::FT
    r_pom_nut_nut::FT
    r_dom_nut_nut::FT
    OtoN::FT
    CtoN::FT
    NtoN::FT
    NtoB::FT
    # Ci_
    r_ci_degrad::FT
    r_ci_free_phy::FT
    r_ci_food_het::FT
    thr_ci_food_het::FT
    # Internal
    optionals::B
    sinking_velocities::W
end

# ----- BGC + Ci_ constructor -----
function OXYDEP(grid;
    # BGC defaults
    initial_photosynthetic_slope::FT = 0.1953 / day,
    Iopt::FT = 80.0,
    alphaI::FT = 1.8,
    betaI::FT = 5.2e-4,
    gammaD::FT = 0.71,
    Max_uptake::FT = 1.85 / day,
    Knut::FT = 0.8,
    r_phy_nut::FT = 0.10 / day,
    r_phy_pom::FT = 0.15 / day,
    r_phy_dom::FT = 0.17 / day,
    r_phy_het::FT = 0.8 / day,
    Kphy::FT = 0.1,
    r_pom_het::FT = 0.7 / day,
    Kpom::FT = 2.0,
    Uz::FT = 0.6,
    Hz::FT = 0.5,
    r_het_nut::FT = 0.15 / day,
    r_het_pom::FT = 0.10 / day,
    r_pom_nut_oxy::FT = 0.02 / day,
    r_pom_dom::FT = 0.10 / day,
    r_dom_nut_oxy::FT = 0.10 / day,
    O2_suboxic::FT = 20.0,
    r_pom_nut_nut::FT = 0.005 / day,
    r_dom_nut_nut::FT = 0.003 / day,
    OtoN::FT = 8.625,
    CtoN::FT = 6.625,
    NtoN::FT = 5.3,
    NtoB::FT = 0.016,
    # Ci_ defaults
    r_ci_degrad::FT = 0.003 / day,
    r_ci_free_phy::FT = 100.0 / day,
    r_ci_food_het::FT = 1.1 / day,
    thr_ci_food_het::FT = 0.001,
    # Optional
    surface_photosynthetically_active_radiation = PAR⁰,
    light_attenuation_model::LA = TwoBandPhotosyntheticallyActiveRadiation(;
        grid, surface_PAR = surface_photosynthetically_active_radiation),
    sediment_model::S = nothing,
    TS_forced::Bool = false,
    Chemicals::Bool = false,
    sinking_speeds = (P = 1.0 / day, HET = 4.0 / day, POM = 9.0 / day,
                      Ci_PHY = 1.0 / day, Ci_HET = 4.0 / day, Ci_POM = 9.0 / day),
    open_bottom::Bool = true,
    scale_negatives = true,
    particles::P = nothing,
    modifiers::M = nothing,
) where {FT,LA,S,P,M}

    sinking_velocities = setup_velocity_fields(sinking_speeds, grid, open_bottom)
    optionals = Val((TS_forced, Chemicals))

    underlying_biogeochemistry = OXYDEP(
        initial_photosynthetic_slope,
        Iopt, alphaI, betaI, gammaD,
        Max_uptake, Knut,
        r_phy_nut, r_phy_pom, r_phy_dom,
        r_phy_het, Kphy, r_pom_het, Kpom,
        Uz, Hz, r_het_nut, r_het_pom,
        r_pom_nut_oxy, r_pom_dom,
        r_dom_nut_oxy,
        O2_suboxic, r_pom_nut_nut, r_dom_nut_nut,
        OtoN, CtoN, NtoN, NtoB,
        r_ci_degrad, r_ci_free_phy, r_ci_food_het, thr_ci_food_het,
        optionals, sinking_velocities,
    )

    if scale_negatives
        scaler = ScaleNegativeTracers(underlying_biogeochemistry, grid)
        modifiers = isnothing(modifiers) ? scaler : (modifiers..., scaler)
    end

    return Biogeochemistry(underlying_biogeochemistry;
        light_attenuation = light_attenuation_model,
        sediment = sediment_model, particles, modifiers)
end

# ----- BGC + Ci_ required tracers -----
required_biogeochemical_tracers(::OXYDEP{<:Any,<:Val{(false, false)},<:Any}) =
    (:NUT, :P, :HET, :POM, :DOM, :O₂, :T, :Ci_free, :Ci_PHY, :Ci_HET, :Ci_POM, :Ci_DOM)

# ----- BGC + Ci_ adapt_structure -----
adapt_structure(to, o::OXYDEP) = OXYDEP(
    adapt(to, o.initial_photosynthetic_slope),
    adapt(to, o.Iopt), adapt(to, o.alphaI), adapt(to, o.betaI), adapt(to, o.gammaD),
    adapt(to, o.Max_uptake), adapt(to, o.Knut),
    adapt(to, o.r_phy_nut), adapt(to, o.r_phy_pom), adapt(to, o.r_phy_dom),
    adapt(to, o.r_phy_het), adapt(to, o.Kphy), adapt(to, o.r_pom_het), adapt(to, o.Kpom),
    adapt(to, o.Uz), adapt(to, o.Hz), adapt(to, o.r_het_nut), adapt(to, o.r_het_pom),
    adapt(to, o.r_pom_nut_oxy), adapt(to, o.r_pom_dom),
    adapt(to, o.r_dom_nut_oxy),
    adapt(to, o.O2_suboxic), adapt(to, o.r_pom_nut_nut), adapt(to, o.r_dom_nut_nut),
    adapt(to, o.OtoN), adapt(to, o.CtoN), adapt(to, o.NtoN), adapt(to, o.NtoB),
    adapt(to, o.r_ci_degrad), adapt(to, o.r_ci_free_phy),
    adapt(to, o.r_ci_food_het), adapt(to, o.thr_ci_food_het),
    adapt(to, o.optionals), adapt(to, o.sinking_velocities),
)

end # if !Ci_ / else

# =====================================================================
# TOML-based constructor (shared by both modes)
# =====================================================================

"""
    OXYDEP(grid, bgc_file; ci_file=nothing, overrides...)

Construct OXYDEP by loading BGC parameters from `bgc_file`.
When Ci_ == true, pass `ci_file` to load additional Ci parameters.
"""
function OXYDEP(grid, bgc_file::AbstractString; ci_file::Union{AbstractString,Nothing}=nothing, overrides...)
    file_params = load_oxydep_params(bgc_file; ci_file=ci_file)
    merged = merge(file_params, Dict{Symbol,Any}(overrides))
    return OXYDEP(grid; merged...)
end

# =====================================================================
# Shared: auxiliary fields, drift velocities
# =====================================================================

required_biogeochemical_auxiliary_fields(::OXYDEP{<:Any,<:Val{(false, false)},<:Any}) = (:PAR,)

# =====================================================================
# Pure BGC: drift velocities
# =====================================================================

@inline function biogeochemical_drift_velocity(bgc::OXYDEP, ::Val{tracer_name}) where {tracer_name}
    if tracer_name in keys(bgc.sinking_velocities)
        return (u = ZeroField(), v = ZeroField(), w = bgc.sinking_velocities[tracer_name])
    else
        return (u = ZeroField(), v = ZeroField(), w = ZeroField())
    end
end

@inline maximum_sinking_velocity(bgc::OXYDEP) = maximum(abs, bgc.sinking_velocities.POM.w)

# =====================================================================
# BGC: helper functions, tendency functions, oxygen, sediment, BCs
# =====================================================================
include("Oxydep_bgc.jl")

# =====================================================================
# Ci_ addition: Ci transformations (only included when Ci_ == true)
# =====================================================================
if Ci_
    include("Oxydep_Ci.jl")
end

end  # module