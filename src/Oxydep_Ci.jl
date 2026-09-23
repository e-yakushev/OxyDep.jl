####################################################################
# Ci_ addition: helper functions for contaminant transformations
####################################################################

@inline Ci_phy_degrad(r_ci_degrad, Ci_PHY) = r_ci_degrad * Ci_PHY
@inline Ci_het_degrad(r_ci_degrad, Ci_HET) = r_ci_degrad * Ci_HET
@inline Ci_pom_degrad(r_ci_degrad, Ci_POM) = r_ci_degrad * Ci_POM
@inline Ci_dom_degrad(r_ci_degrad, Ci_DOM) = r_ci_degrad * Ci_DOM
@inline Ci_free_phy(r_ci_free_phy, Max_uptake, PAR, α, T, Knut, NUT, P, Iopt, Ci_free) = 
    Ci_free < 1e-6 ? zero(Ci_free) :
    r_ci_free_phy * GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt)
@inline Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het) = 
    r_ci_food_het * Uz * GrazPhy(r_phy_het, Kphy, P, HET) * yy(thr_ci_food_het, Ci_free)
@inline Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het) = 
    r_ci_food_het * Uz * GrazPhy(r_phy_het, Kphy, P, HET) * yy(thr_ci_food_het, Ci_PHY)
@inline Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het) = 
    r_ci_food_het * Uz * GrazPOM(r_pom_het, Kpom, POM, HET) * yy(thr_ci_food_het, Ci_POM)    
@inline Ci_het_pom(r_het_pom, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
     Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, O₂, O2_suboxic, thr_ci_food_het) = 
    HET < 1e-6 ? zero(HET) :
    (Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het) +
     Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het) +
     Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)) /
     Uz * (1 - Uz) * (1 - Hz) +
     Ci_HET * MortHet(r_het_pom, HET, O₂, O2_suboxic) / HET   

@inline Ci_het_dom(r_het_nut, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
     Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, thr_ci_food_het) = 
    HET < 1e-6 ? zero(HET) :
    (Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het) +
     Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het) +
     Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)) /
     Uz * (1 - Uz) * Hz +
     Ci_HET * RespHet(r_het_nut, HET) / HET

@inline Ci_pom_dom( r_pom_nut_oxy, r_pom_nut_nut, r_pom_dom, O2_suboxic, NUT, POM, O₂, Ci_POM) = 
     POM < 1e-6 ? zero(POM) :
     Ci_POM * (POM_decay_ox(r_pom_nut_oxy, POM, O₂, O2_suboxic) +
     POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT) + 
     Autolys(r_pom_dom, POM)) /POM
@inline Ci_phy_dom(r_phy_dom, P, Ci_PHY) = 
     P < 1e-6 ? zero(P) :
     Ci_PHY * ExcrPhy(r_phy_dom, P) / P
@inline Ci_phy_pom(r_phy_pom, P, Ci_PHY) = 
     P < 1e-6 ? zero(P) :
     Ci_PHY * MortPhy(r_phy_pom, P) / P

####################################################################
# Ci_ addition: Ci-aware BGC tendency functions (NUT, P, HET, POM, DOM, O₂)
# These replace the pure BGC tendency functions when Ci_ == true
####################################################################

@inline function (bgc::OXYDEP)(::Val{:NUT},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

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
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

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
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

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
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

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
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

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
        DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic)  +
        Autolys(r_pom_dom, POM) +
        POM_decay_denitr(r_pom_nut_nut, POM, O₂, O2_suboxic, NUT)
    )
    # Denitrification of "real DOM" into NH4 (DOM_decay_denitr) will not change state variable DOM
end

@inline function (bgc::OXYDEP)(::Val{:O₂},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)

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
            GrowthPhy(Max_uptake, PAR, α, T, Knut, NUT, P, Iopt) + 
            DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) * 
            (F_subox(O₂, O2_suboxic))
        )
    )
    # (POM_decay_denitr + DOM_decay_denitr) & !denitrification doesn't change oxygen
    # (DOM_decay_ox(r_dom_nut_oxy, DOM, O₂, O2_suboxic) *(F_subox) !additional consumption of O₂ due to oxidation of reduced froms of S,Mn,Fe etc.
    # in suboxic conditions (F_subox) equals consumption for NH4 oxidation (Yakushev et al, 2008)

end

####################################################################
# Ci_ addition: Ci tracer tendency functions (Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM)
####################################################################

@inline function (bgc::OXYDEP)(::Val{:Ci_free},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    Iopt = bgc.Iopt
    r_ci_free_phy = bgc.r_ci_free_phy
    r_ci_food_het = bgc.r_ci_food_het
    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    Uz = bgc.Uz
    thr_ci_food_het = bgc.thr_ci_food_het
    return (
        - Ci_free_phy(r_ci_free_phy, Max_uptake, PAR, α, T, Knut, NUT, P, Iopt, Ci_free)
        - Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het)
    )
end

@inline function (bgc::OXYDEP)(::Val{:Ci_PHY},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    Max_uptake = bgc.Max_uptake
    Knut = bgc.Knut
    α = bgc.initial_photosynthetic_slope
    Iopt = bgc.Iopt
    r_ci_free_phy = bgc.r_ci_free_phy
    r_phy_het = bgc.r_phy_het
    Kphy = bgc.Kphy
    Uz = bgc.Uz
    r_ci_food_het = bgc.r_ci_food_het
    r_phy_dom = bgc.r_phy_dom
    r_phy_pom = bgc.r_phy_pom
    thr_ci_food_het = bgc.thr_ci_food_het
    r_ci_degrad = bgc.r_ci_degrad
    return (
          Ci_free_phy(r_ci_free_phy, Max_uptake, PAR, α, T, Knut, NUT, P, Iopt, Ci_free)
        - Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het)
        - Ci_phy_pom(r_phy_pom, P, Ci_PHY) 
        - Ci_phy_dom(r_phy_dom, P, Ci_PHY)     
        - Ci_phy_degrad(r_ci_degrad, Ci_PHY)
        )
end

@inline function (bgc::OXYDEP)(::Val{:Ci_HET},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    r_phy_het = bgc.r_phy_het
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_ci_food_het = bgc.r_ci_food_het
    thr_ci_food_het = bgc.thr_ci_food_het
    r_het_pom = bgc.r_het_pom
    r_phy_het = bgc.r_phy_het
    O2_suboxic = bgc.O2_suboxic
    Kphy = bgc.Kphy
    Kpom = bgc.Kpom
    r_het_nut = bgc.r_het_nut
    r_ci_degrad = bgc.r_ci_degrad
    return (
          Ci_phy_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_PHY, thr_ci_food_het)
        + Ci_free_het(r_ci_food_het, Uz, r_phy_het, Kphy, P, HET, Ci_free, thr_ci_food_het)
        + Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)
        - Ci_het_pom(r_het_pom, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
            Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, O₂, O2_suboxic, thr_ci_food_het) 
        - Ci_het_dom(r_het_nut, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
             Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, thr_ci_food_het)            
        - Ci_het_degrad(r_ci_degrad, Ci_HET)          
    )
end
@inline function (bgc::OXYDEP)(::Val{:Ci_POM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    r_ci_degrad = bgc.r_ci_degrad
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_ci_food_het = bgc.r_ci_food_het
    thr_ci_food_het = bgc.thr_ci_food_het
    r_het_pom = bgc.r_het_pom
    r_phy_het = bgc.r_phy_het
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    r_phy_pom = bgc.r_phy_pom
    O2_suboxic = bgc.O2_suboxic
    Kphy = bgc.Kphy
    Kpom = bgc.Kpom
    return (
          Ci_het_pom(r_het_pom, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
             Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, O₂, O2_suboxic, thr_ci_food_het) 
        - Ci_pom_het(r_ci_food_het, Uz, r_pom_het, Kpom, POM, HET, Ci_POM, thr_ci_food_het)
        - Ci_pom_dom( r_pom_nut_oxy, r_pom_nut_nut, r_pom_dom, O2_suboxic, 
             NUT, POM, O₂, Ci_POM)      
        + Ci_phy_pom(r_phy_pom, P, Ci_PHY)             
        - Ci_pom_degrad(r_ci_degrad, Ci_POM)    
    )
end
@inline function (bgc::OXYDEP)(::Val{:Ci_DOM},
        x, y, z, t,
        NUT, P, HET, POM, DOM, O₂, T,
        Ci_free, Ci_PHY, Ci_HET, Ci_POM, Ci_DOM, PAR)
    r_ci_degrad = bgc.r_ci_degrad
    Uz = bgc.Uz
    Hz = bgc.Hz
    r_pom_het = bgc.r_pom_het
    Kpom = bgc.Kpom
    r_ci_food_het = bgc.r_ci_food_het
    thr_ci_food_het = bgc.thr_ci_food_het
    r_het_nut = bgc.r_het_nut
    r_phy_het = bgc.r_phy_het
    r_pom_nut_oxy = bgc.r_pom_nut_oxy
    r_pom_dom = bgc.r_pom_dom
    r_pom_nut_nut = bgc.r_pom_nut_nut
    r_phy_dom = bgc.r_phy_dom
    O2_suboxic = bgc.O2_suboxic
    Kphy = bgc.Kphy
    Kpom = bgc.Kpom
    return (
         Ci_het_dom(r_het_nut, r_phy_het, r_pom_het, r_ci_food_het, Uz, Hz, Kphy, Kpom, 
             Ci_free, Ci_PHY, Ci_HET, Ci_POM, P, HET, POM, thr_ci_food_het)   
        + Ci_pom_dom( r_pom_nut_oxy, r_pom_nut_nut, r_pom_dom, O2_suboxic, 
             NUT, POM, O₂, Ci_POM)     
        + Ci_phy_dom(r_phy_dom, P, Ci_PHY)                                    
        - Ci_dom_degrad(r_ci_degrad, Ci_DOM)          
    )
end

####################################################################
# Ci_ addition: Ci burial functions
####################################################################

@inline function _Ci_PHY_burial(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    val = @inbounds fields.Ci_PHY[i, j, k]
    w = @inbounds p.w[i, j, k]
    flux = -p.bu * w * val
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(val))
end

@inline function _Ci_HET_burial(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    val = @inbounds fields.Ci_HET[i, j, k]
    w = @inbounds p.w[i, j, k]
    flux = -p.bu * w * val
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(val))
end

@inline function _Ci_POM_burial(i, j, k, grid, clock, fields, p)
    bottom = _is_seafloor(i, j, k, grid)
    val = @inbounds fields.Ci_POM[i, j, k]
    w = @inbounds p.w[i, j, k]
    flux = -p.bu * w * val
    return ifelse(bottom, flux / Δzᶜᶜᶜ(i, j, k, grid), zero(val))
end
