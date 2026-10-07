"""
OxyDep basic biogeochemical transformations between NUT, P, HET, DOM, POM, O2.
This file is organized: pure BGC code first, then Ci_ additions (when enabled).
"""

# =====================================================================
# BGC helper functions: limiters, growth, grazing, decay (used by both modes)
# =====================================================================

# Limiting equations and switches
@inline yy(consta, value) = value^2 / (value^2 + consta^2)   #This is a squared Michaelis-Menten type of limiter
@inline F_ox(conc, threshold) = (0.5 + 0.5 * tanh(conc - threshold))
@inline F_subox(conc, threshold) = (0.5 - 0.5 * tanh(conc - threshold))

# P
@inline LimLight(PAR, Iopt) = PAR / Iopt * exp(1.0 - PAR / Iopt)  #!Dependence of P growth on Light (Steel)
@inline LimN(Knut, NUT, P) = yy(Knut, NUT / max(0.0001, P)) #!Dependence of P growth on NUT
@inline Q₁₀(T) = 1.88^(T / 10) # T in °C  # inital for NPZD
#@inline LimT(T) = max(0., 2^((T-10.0)/10.) - 2^((T-32.)/3.)) # ERSEM
# = q10^((T-t_upt_min)/10)-q10^((T-t_upt_max)/3):  q10=2. !Coefficient for uptake rate dependence on t
# t_upt_min=10. !Low  t limit for uptake rate dependence on t; t_upt_max=32 !High t limit for uptake rate dependence on t
@inline LimT(T) = exp(0.0663 * (T - 0.0)) #for Arctic (Moore et al.,2002; Jin et al.,2008) 
# = exp(temp_aug_rate*(T-t_0)):  t_0= 0. !reference temperature temp_aug_rate = 0.0663 !temperature augmentation rate
#@inline light_limitation(PAR, α, Max_uptake) = α * PAR / sqrt(Max_uptake ^ 2 + α ^ 2 * PAR ^ 2)

#@inline GrowthPhy(Max_uptake,PAR,α,T,Knut,NUT,P,Iopt) = Max_uptake*LimT(T)*LimN(Knut,NUT,P)*light_limitation(PAR,α,Max_uptake)*P*Iopt/Iopt
@inline GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) =
    Max_uptake * LimT(T) * LimN(Knut, NUT, P) * LimLight(PAR, Iopt) * α / α
@inline RespPhy(r_phy_nut, P) = r_phy_nut * P
@inline MortPhy(r_phy_pom, P) = r_phy_pom * P
@inline ExcrPhy(r_phy_dom, P) = r_phy_dom * P

# HET
@inline GrazPhy(r_phy_het, Kphy, P, HET) =
    r_phy_het * yy(Kphy, max(0.0, P - 0.01) / max(0.0001, HET)) * HET
@inline GrazPOM(r_pom_het, Kpom, POM, HET) =
    r_pom_het * yy(Kpom, max(0.0, POM - 0.01) / max(0.0001, HET)) * HET
@inline RespHet(r_het_nut, HET) = r_het_nut * HET
@inline MortHet(r_het_pom, HET, O₂, O2_suboxic) =
    (r_het_pom + F_subox(O₂, O2_suboxic) * 0.01 * r_het_pom) * HET

# POM
@inline POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) = 
    O₂ < 0.1 ? zero(O₂) : r_pom_nut_oxy * POM * F_ox(O₂, O2_suboxic)
@inline POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT) =
    NUT < 0.05 ? zero(NUT) : r_pom_nut_nut * POM * F_subox(O₂, O2_suboxic)
#! depends on NUT (NO3+NO2) and DOM (NH4+Urea+"real"DON) ! depends on T ! stops at NUT<0.01 
@inline Autolys(r_pom_dom, POM) = r_pom_dom * POM

# DOM
@inline DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) = 
    O₂ < 0.1 ? zero(O₂) : r_dom_nut_oxy * DOM * F_ox(O₂, O2_suboxic)
@inline DOM_decay_denitr(r_dom_nut_nut, DOM, O₂, O2_suboxic, NUT) =
    NUT < 0.05 ? zero(NUT) : r_dom_nut_nut * DOM * F_subox(O₂, O2_suboxic)
#! depends on NUT (NO3+NO2) and DOM (NH4+Urea+"real"DON) ! depends on T ! stops at NUT<0.01 

# =====================================================================
# Pure BGC tendency functions (active when Ci_ == false)
# =====================================================================

if !Ci_

@inline function (bgc::OXYDEP)(::Val{:NUT},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_nut = bgc.r_phy_nut
    r_het_nut = bgc.r_het_nut
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    NtoN = bgc.NtoN
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic
    r_dom_nut_nut = bgc.r_dom_nut_nut
    Iopt = bgc.Iopt

    return (
        RespPhy(r_phy_nut, P) +
        RespHet(r_het_nut, HET) +
        DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
        POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) - 
        GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) -
        NtoN * (
            POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT) +
            DOM_decay_denitr(r_dom_nut_nut, DOM, O₂, O2_suboxic, NUT)
        )
    )
    # Denitrification of POM and DOM leads to decrease of NUT (i.e. NOx)
end

@inline function (bgc::OXYDEP)(::Val{:P},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_phy_nut = bgc.r_phy_nut
    r_phy_pom = bgc.r_phy_pom
    r_phy_dom = bgc.r_phy_dom
    Iopt = bgc.Iopt

    return (
        GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) -
        GrazPhy(r_phy_het, Kphy, P, HET) - RespPhy(r_phy_nut, P) - MortPhy(r_phy_pom, P) -
        ExcrPhy(r_phy_dom, P)
    )
end

@inline function (bgc::OXYDEP)(::Val{:HET},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_het_nut = bgc.r_het_nut
    r_het_pom = bgc.r_het_pom
    Uz = bgc.Uz
    O2_suboxic = bgc.O2_suboxic

    return (
        Uz * (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) -
        MortHet(r_het_pom, HET, O₂, O2_suboxic) - RespHet(r_het_nut, HET)
    )
end

@inline function (bgc::OXYDEP)(::Val{:POM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_phy_pom = bgc.r_phy_pom
    r_het_pom = bgc.r_het_pom
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic

    return (
        (1.0 - Uz) *
        (1.0 - Hz) *
        (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) +
        MortPhy(r_phy_pom, P) +
        MortHet(r_het_pom, HET, O₂, O2_suboxic) - 
        POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) -
        Autolys(r_pom_dom, POM) - GrazPOM(r_pom_het, Kpom, POM, HET) -
        POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT)
    )
end

@inline function (bgc::OXYDEP)(::Val{:DOM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_phy_dom = bgc.r_phy_dom
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    O2_suboxic = bgc.O2_suboxic

    return (
        (1.0 - Uz) *
        Hz *
        (GrazPhy(r_phy_het, Kphy, P, HET) + GrazPOM(r_pom_het, Kpom, POM, HET)) +
        ExcrPhy(r_phy_dom, P) - 
        DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
        Autolys(r_pom_dom, POM) +
        POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT)
    )
    # Denitrification of "real DOM" into NH4 (DOM_decay_denitr) will not change state variable DOM
end

@inline function (bgc::OXYDEP)(::Val{:O₂},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        PAR)

    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    r_phy_nut = bgc.r_phy_nut
    r_het_nut = bgc.r_het_nut
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_dom_nut_oxy = bgc.r_dom_nut_oxy
    OtoN = bgc.OtoN
    O2_suboxic = bgc.O2_suboxic
    Iopt = bgc.Iopt

    return (
        -OtoN * (
            RespPhy(r_phy_nut, P) +
            RespHet(r_het_nut, HET) +
            DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) +
            POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) -
            GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) # due to OM production and decay in normoxia
            +
            DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) * F_subox(O₂, 0.5 * O2_suboxic)
            )
        )
    # (POM_decay_denitr + DOM_decay_denitr) & !denitrification doesn't change oxygen
    # (DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) *(F_subox) !additional 
    # consumption of O₂ due to oxidation of reduced froms of S,Mn,Fe etc.
    # In suboxic conditions (F_subox) equals consumption for NH4 oxidation (Yakushev et al, 2008)

end

end # if !Ci_

# =====================================================================
# Pure BGC: oxygen saturation and gas exchange
# =====================================================================

# Coefficients from Garcia and Gordon (1992)
const A1 = -173.4292
const A2 = 249.6339
const A3 = 143.3483
const A4 = -21.8492
const A5 = -0.033096
const A6 = 0.014259
const B1 = -0.035274
const B2 = 0.001429
const B3 = -0.00007292
const C1 = 0.0000826

""" Function to calculate oxygen saturation in seawater """
function oxygen_saturation(T::Float64, S::Float64, P::Float64)::Float64

    T_kelvin = T + 273.15  # Convert temperature to Kelvin

    # Calculate the natural logarithm of oxygen saturation concentration
    ln_O2_sat =
        A1 +
        A2 * (100 / T_kelvin) +
        A3 * log(T_kelvin / 100) +
        A4 * T_kelvin / 100 +
        A5 * (T_kelvin / 100)^2 +
        A6 * (T_kelvin / 100)^3 +
        S * (B1 + B2 * (T_kelvin / 100) + B3 * (T_kelvin / 100)^2) +
        C1 * S^2

    # Oxygen saturation concentration in µmol/kg
    O2_sat = exp(ln_O2_sat) * 44.66

    # Pressure correction factor (Weiss, 1970) for pressure in atm
    P_corr = 1.0 + P * (5.6e-6 + 2.0e-11 * P)

    # Adjusted oxygen saturation with pressure correction
    return (O2_sat * P_corr)
end

""" Sc, Schmidt number for O2  following Wanninkhof 2014 """
@inline function OxygenSchmidtNumber(T::Float64)::Float64
    return ((1920.4 - 135.6 * T + 5.2122 * T^2 - 0.10939 * T^3 + 0.00093777 * T^4))
    # can be replaced by PolynomialParameterisation{4}((a, b, c, d, e)) i.e.:
    #    a = 1953.4, b = - 128.0, c = 3.9918, d = -0.050091, e = 0.00093777  
    # Sc = PolynomialParameterisation{4}((a, b, c, d, e))
end

""" WindDependence, [mmol m-2s-1], Oxygen Sea Water Flux """
function WindDependence(windspeed::Float64)::Float64
    return (0.251 * windspeed^2.0) #ko2o=0.251*windspeed^2*(Sc/660)^(-0.5)  Wanninkhof 2014
end

""" OxygenSeaWaterFlux, [mmol m-2s-1], Oxygen Sea Water Flux """
function OxygenSeaWaterFlux(T::Float64, S::Float64, P::Float64, O₂::Float64, windspeed::Float64)::Float64
    return (
        WindDependence(windspeed) * (OxygenSchmidtNumber(T) / 660.0)^(-0.5) * (O₂ - oxygen_saturation(T, S, P)) * 0.24 /
        86400.0        # 0.24 is to convert from [cm/h] to [m/day]  * 0.24  / 86400.0
    )
end

# =====================================================================
# Pure BGC: conserved tracers and sinking
# =====================================================================

@inline nitrogen_flux(i, j, k, grid, advection, bgc::OXYDEP, tracers) =
    sinking_flux(i, j, k, grid, advection, Val(:POM), bgc, tracers) +
    sinking_flux(i, j, k, grid, advection, Val(:P), bgc, tracers)
@inline conserved_tracers(::OXYDEP) = (:NUT, :P, :HET, :POM, :DOM, :O₂)
@inline sinking_tracers(bgc::OXYDEP) = keys(bgc.sinking_velocities)

# =====================================================================
# Pure BGC: sediment/boundary parameters and forcing functions
# =====================================================================

O2_suboxic = 20.0   # OXY threshold for oxic/suboxic switch (mmol/m3)
Trel = 86400.        # Relaxation time for sediment exchange (s)
# positive for flux from water to the sediments:
b_O2_ox =       5.0  # flux of OXY at SWI, (mmol/m2/d) 
b_O2_subox =   10.0  # flux of OXY at SWI in subox, (mmol/m2/d) 
b_NUT_ox =     -2.0  # flux of NUT at SWI, (mmol/m2/d)
b_NUT_subox =   7.0  # flux of NUT at SWI in subox, (mmol/m2/d) 
b_DOM_ox =     -2.0  # flux of DOM at SWI, (mmol/m2/d) 
b_DOM_subox =  -8.0  # flux of DOM at SWI in subox, (mmol/m2/d)   
bu = 0.1            # Burial coefficient (0<bu<1) (nd) 0.001
windspeed = 5.0       # wind speed 10 m, (m/s)

function apply_sediment_config!(sed::Dict)
    global O2_suboxic, Trel, b_O2_ox, b_O2_subox
    global b_NUT_ox, b_NUT_subox, b_DOM_ox, b_DOM_subox, bu, windspeed
    haskey(sed, "O2_suboxic") && (O2_suboxic = sed["O2_suboxic"])
    haskey(sed, "Trel")       && (Trel       = sed["Trel"])
    haskey(sed, "b_O2_ox")    && (b_O2_ox    = sed["b_O2_ox"])
    haskey(sed, "b_O2_subox") && (b_O2_subox = sed["b_O2_subox"])
    haskey(sed, "b_NUT_ox")   && (b_NUT_ox   = sed["b_NUT_ox"])
    haskey(sed, "b_NUT_subox")&& (b_NUT_subox= sed["b_NUT_subox"])
    haskey(sed, "b_DOM_ox")   && (b_DOM_ox   = sed["b_DOM_ox"])
    haskey(sed, "b_DOM_subox")&& (b_DOM_subox= sed["b_DOM_subox"])
    haskey(sed, "bu")         && (bu         = sed["bu"])
    haskey(sed, "windspeed")  && (windspeed  = sed["windspeed"])
    nothing
end

@inline function _is_seafloor(i, j, k, grid)
    return bottommost_active_node(i, j, k, grid, Center(), Center(), Center())
end

# Pure BGC sediment forcing kernels

@inline function _oxy_sediment(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    O₂ = @inbounds fields.O₂[i, j, k]
    flux = O₂ ≤ 0 ? zero(O₂) : -(F_ox(O₂, p.O2_suboxic) * p.b_O2_ox +
             F_subox(O₂, p.O2_suboxic) * p.b_O2_subox) / p.Trel
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(O₂))
end

@inline function _nut_sediment(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    O₂ = @inbounds fields.O₂[i, j, k]
    NUT = @inbounds fields.NUT[i, j, k]
    flux = NUT ≤ 0 ? zero(NUT) : -(F_ox(O₂, p.O2_suboxic) * p.b_NUT_ox +
             F_subox(O₂, p.O2_suboxic) * p.b_NUT_subox) / p.Trel
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(O₂))
end

@inline function _dom_sediment(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    O₂ = @inbounds fields.O₂[i, j, k]
    flux = -(F_ox(O₂, p.O2_suboxic) * p.b_DOM_ox +
             F_subox(O₂, p.O2_suboxic) * p.b_DOM_subox) / p.Trel
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(O₂))
end

# Pure BGC burial kernels

@inline function _P_burial(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    P = @inbounds fields.P[i, j, k]
    w = @inbounds p.w[i, j, k]
    flux = -p.bu * w * P
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(P))
end

@inline function _HET_burial(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    HET = @inbounds fields.HET[i, j, k]
    w = @inbounds p.w[i, j, k]
    flux = -p.bu * w * HET
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(HET))
end

@inline function _POM_burial(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    POM = @inbounds fields.POM[i, j, k]
    w = @inbounds p.w[i, j, k]
    flux = -p.bu * w * POM
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(POM))
end

# =====================================================================
# Pure BGC: sediment forcings assembly
# =====================================================================

function oxydep_sediment_forcings(biogeochemistry)
    bgc = biogeochemistry.underlying_biogeochemistry
    w_P   = biogeochemical_drift_velocity(bgc, Val(:P)).w
    w_HET = biogeochemical_drift_velocity(bgc, Val(:HET)).w
    w_POM = biogeochemical_drift_velocity(bgc, Val(:POM)).w

    sed = (O2_suboxic=Float64(O2_suboxic), Trel=Float64(Trel),
           b_O2_ox=Float64(b_O2_ox), b_O2_subox=Float64(b_O2_subox),
           b_NUT_ox=Float64(b_NUT_ox), b_NUT_subox=Float64(b_NUT_subox),
           b_DOM_ox=Float64(b_DOM_ox), b_DOM_subox=Float64(b_DOM_subox))
    bu_val = Float64(bu)

    # Pure BGC forcings
    base = (
        O₂  = DiscreteForcing(_oxy_sediment; parameters=sed),
        NUT = DiscreteForcing(_nut_sediment; parameters=sed),
        DOM = DiscreteForcing(_dom_sediment; parameters=sed),
        P   = DiscreteForcing(_P_burial;   parameters=(w=w_P,   bu=bu_val)),
        HET = DiscreteForcing(_HET_burial; parameters=(w=w_HET, bu=bu_val)),
        POM = DiscreteForcing(_POM_burial; parameters=(w=w_POM, bu=bu_val)),
    )

    # Ci_ addition: append Ci burial forcings when Ci_ is enabled
    if Ci_
        w_Ci_PHY = biogeochemical_drift_velocity(bgc, Val(:Ci_PHY)).w
        w_Ci_HET = biogeochemical_drift_velocity(bgc, Val(:Ci_HET)).w
        w_Ci_POM = biogeochemical_drift_velocity(bgc, Val(:Ci_POM)).w
        return merge(base, (
            Ci_PHY = DiscreteForcing(_Ci_PHY_burial; parameters=(w=w_Ci_PHY, bu=bu_val)),
            Ci_HET = DiscreteForcing(_Ci_HET_burial; parameters=(w=w_Ci_HET, bu=bu_val)),
            Ci_POM = DiscreteForcing(_Ci_POM_burial; parameters=(w=w_Ci_POM, bu=bu_val)),
        ))
    else
        return base
    end
end

# =====================================================================
# Pure BGC: boundary conditions (O₂ surface gas exchange)
# =====================================================================

function bgh_oxydep_boundary_conditions(biogeochemistry, Nz)

    Oxy_top_cond(i, j, grid, clock, fields, p) = @inbounds (OxygenSeaWaterFlux(
        fields.T[i, j, Nz],
        fields.S[i, j, Nz],
        0.0,                # sea surface pressure
        fields.O₂[i, j, Nz],
        p.windspeed,
    ))

    OXY_top = FluxBoundaryCondition(Oxy_top_cond; discrete_form = true,
                                    parameters = (windspeed = Float64(windspeed),))
    oxy_bcs = FieldBoundaryConditions(top = OXY_top)

    return (O₂ = oxy_bcs,)
end
