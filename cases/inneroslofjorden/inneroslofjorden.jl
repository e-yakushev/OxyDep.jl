using FjordSim
using Oceananigans:
    CATKEVerticalDiffusivity,
    HydrostaticSphericalCoriolis,
    SeawaterBuoyancy,
    WENO,
    WENOVectorInvariant
using Oceananigans.TurbulenceClosures: HorizontalScalarBiharmonicDiffusivity
using Oceananigans.Units: days, hour, hours, minutes, second
using FjordSim.Bathymetry: GEONORGE_DYBDEDATA_GDB
using Dates: DateTime
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using NumericalEarth: FreezingLimitedOceanTemperature

# Uniform T and S are the January 1 boundary profiles at 0.5 m; zero velocity is a rest initial
# state. The OxyDep setup adds its own uniform biogeochemical values.
const INNEROSLOFJORDEN_UNIFORM_INITIAL_CONDITIONS = (
    T = 3.6225724f0,
    S = 28.742033f0,
    u = 0f0,
    v = 0f0,
)

"""
    initial_conditions_for_run(source, uniform; results_directory)

`uniform` for `source == :uniform`. For `source == :results`, a `FromResults` of the newest
run-tagged `snapshots_ocean_*.nc` in `results_directory`.
"""
function initial_conditions_for_run(source, uniform; results_directory)
    source == :uniform && return uniform
    source == :results || error(
        "initial_conditions_source must be :uniform or :results, got $(repr(source))",
    )

    isdir(results_directory) || error(
        "No results directory $results_directory; run the first year before selecting :results.",
    )
    snapshot_pattern = r"^snapshots_ocean_\d{8}T\d{6}(?:_loop\d+)?\.nc$"
    snapshots = sort(filter(name -> occursin(snapshot_pattern, name), readdir(results_directory)))
    isempty(snapshots) && error(
        "No run-tagged snapshots_ocean_*.nc found in $results_directory. " *
        "Run the first year before selecting :results.",
    )
    return FromResults(joinpath(results_directory, last(snapshots)))
end

"""
    inneroslofjorden(;
        architecture = :auto,
        initial_conditions_source = :uniform,
        initial_results_directory = fjord_results_root("inneroslofjorden_init"),
    )

Inner Oslofjord hydrophysics for calendar year 2020 on a 63 x 114 x 17 grid covering
10.46-10.79°E, 59.62-59.92°N — 293 m cells — open to the south. There is no interior forcing
config: the initial state is uniform, or with `initial_conditions_source = :results` the last
record of a previous run in `initial_results_directory` (`initial_conditions_for_run`), and the
rivers, southern open-boundary data and NORA3 atmosphere are
the time-dependent inputs. `inneroslofjorden_oxydep.jl` replaces the river, boundary-data and
biogeochemistry components and keeps this physics and run window.
"""
function inneroslofjorden(;
    architecture = :auto,
    initial_conditions_source = :uniform,
    initial_results_directory = fjord_results_root("inneroslofjorden_init"),
)
    data_root = fjord_data_root("inneroslofjorden")
    FT = Float32

    return FjordConfig(
        grid_config = EvenGrid(
            size      = (63, 114, 17),
            halo      = (7, 7, 7),
            longitude = (10.46, 10.79),
            latitude  = (59.62, 59.92),
            # The deepest cell `prepare_bathymetry` writes is 203 m.
            z_faces   = [
                -200.0, -180.0, -160.0, -140.0, -120.0, -100.0, -80.0, -60.0,
                -40.0, -24.0, -15.0, -10.0, -7.5, -5.0, -3.0, -2.0, -1.0, 0.0,
            ],
            float_type = FT,
        ),
        bathymetry_config = DybdedataConfig(
            data_root             = data_root,
            output_file           = "bathymetry.nc",
            plot_file             = "bathymetry.png",
            geodatabase_file      = GEONORGE_DYBDEDATA_GDB,
            raw_resolution_factor = 2,
            padding_cells         = 2,
            include_contours      = false,
            contour_stride        = 10,
            interpolation_passes  = 1,
            # One basin drops the lakes in Geonorge's soundings.
            major_basins          = 1,
            minimum_depth         = 2.0,
            open_boundary_land_cells = 1,
            max_island_cells      = 3,
            close_narrow_passages = true,
            spike_ratio           = 0.25,
            # Must equal the `minimum_fractional_cell_height` `Grids.jl` gives `PartialCellBottom`.
            minimum_cell_fraction = 0.2,
            max_slope_factor      = 0.5,
            geonorge_cache        = true,
            regrid_cache          = false,
        ),
        river_config = NVERiversConfig(
            data_root         = data_root,
            output_file       = "rivers_nve.nc",
            plot_file         = "rivers_nve.png",
            years             = [2020],
            # Every mouth NVE has in this domain carrying 0.5 m³/s or more, sized by its own
            # REGINE catchment.
            minimum_discharge = 0.5,
            # A vertical Courant bound on the 1 m top cell: 1 mm/s is w·Δt/Δz = 0.18 at the
            # 3-minute `max_time_step` below.
            maximum_inflow_velocity = 1.0e-3,
            outlets           = NVERiver[],
        ),
        # The hourly NorKyst exterior state along the open southern edge, and the one place that
        # edge is stated.
        boundary_data_config = NorKystBoundaryDataConfig(
            data_root        = data_root,
            output_directory = "norkyst_hourly",
            output_file      = "boundaries.nc",
            plot_file        = "boundaries.png",
            open_edges       = :south,
            margin           = 0.05,
            architecture     = :auto,
            parameters       = [
                "temperature", "salinity", "u_eastward", "v_northward", "zeta", "ubar", "vbar",
            ],
            years            = [2020],
        ),
        atmosphere_config = NORA3Config(
            data_root        = data_root,
            output_directory = "nora3",
            output_file      = "atmosphere.nc",
            plot_file        = "atmosphere.png",
            resolution       = 0.02,
            padding          = 0.1,
            years            = [2020],
        ),
        # The mixing and advection of `examples/oslofjorden.jl`, with its grid-spacing-dependent
        # coefficients scaled from that setup's 193 m cell to this one's 293 m (`docs/setups.md`).
        model_config = CoupledHydrostaticModel(
            buoyancy           = SeawaterBuoyancy(FT, equation_of_state = TEOS10EquationOfState(FT)),
            # A harmonic viscosity and diffusivity ramping to zero over 16 cells (~4.7 km) inward
            # from the open southern edge, damping what the open boundary radiates. Oslofjord's
            # 30 and 15 m² s⁻¹ scaled by (293/193)² = 2.3. Explicit stability Δt ≤ Δx²/4ν is 307 s
            # at 70, clear of the 180 s `max_time_step`.
            closure            = BoundarySponge(
                base = (
                    # CATKE takes `w★ = sqrt(max(minimum_tke, e))`, so this floor, not the
                    # prognostic TKE, sets κ = 0.098·e_min/N over most of the stratified column;
                    # 7e-7 is Oslofjord's value, fitted against the NIVA casts. The 1e-3 m² s⁻¹ cap
                    # on the tracer diffusivity is this setup's own and limits convective mixing.
                    CATKEVerticalDiffusivity(FT; minimum_tke = 7e-7, maximum_tracer_diffusivity = 1e-3),
                    # Oslofjord's 2e4 m⁴ s⁻¹ scaled by (293/193)⁴ = 5.3: a 77 min e-folding of the
                    # 2Δx mode (Δx⁴/16ν₄), and an explicit limit Δt ≤ Δx⁴/32ν₄ of 2300 s that
                    # `AdaptiveTimeStep` never measures. κ = 0: on Oslofjord biharmonic tracer
                    # diffusion overshot at fronts and left cells below 0 g/kg.
                    HorizontalScalarBiharmonicDiffusivity(FT; ν = 1e5, κ = 0),
                ),
                width_cells = 16,
                viscosity   = 70.0, #100.0,
                diffusivity = 35.0, #50.0,
            ),
            # T and S upwinded horizontally within three cells (WENO5's stencil) of every river
            # column, so no face reconstructs across a mouth's fresh front, and salinity's WENO
            # bounded to (0, 40). A tracer this NamedTuple omits falls back to Oceananigans'
            # unbounded `Centered()`, so it names CATKE's `e`, and a setup adding tracers must name
            # those too.
            tracer_advection   = (
                T = RiverMouthAdvection(interior = WENO(FT), width_cells = 3),
                S = RiverMouthAdvection(interior = WENO(FT; bounds = (0, 40)), width_cells = 3),
                e = WENO(FT),
            ),
            momentum_advection = WENOVectorInvariant(FT),
            tracers            = (:T, :S),
            coriolis           = HydrostaticSphericalCoriolis(FT),
            sea_ice            = FreezingLimitedOceanTemperature(FT),
            free_surface       = SplitExplicitFreeSurfaceConfig(cfl = 0.7),
            extra_kwargs       = (;),
        ),
        # Tuple order is merge precedence. The two open-boundary timescales are Marchesiello et al.
        # (2001): nudge hard towards the data on inflow, let radiation do the work on outflow.
        boundary_condition_configs = (
            AirSeaFluxes(),
            QuadraticBottomDrag(coefficient = 0.003),
            OpenLateralBoundaryFromData(
                inflow_timescale  = 3hours,
                outflow_timescale = 360days,
            ),
        ),
        run_config = RunConfig(
            results_root = fjord_results_root("inneroslofjorden"),
            architecture = architecture,
            writers = (
                SnapshotWriter(
                    name               = :ocean,
                    output_file        = "snapshots_ocean.nc",
                    variables          = (:T, :S, :u, :v, :e),
                    interval           = 3hours,
                    split_times        = Float64[],
                    overwrite_existing = true,
                ),
                FieldSnapshotWriter(
                    name               = :surface,
                    output_file        = "snapshots_surface.jld2",
                    variables          = (:η,),
                    interval           = 3hours,
                    overwrite_existing = true,
                ),
                CheckpointWriter(interval = 12hours, cleanup = true),
            ),
            callbacks = (ProgressCallback(name = :progress, interval = 24hours, report = progress),),
            time_stepping = AdaptiveTimeStep(
                initial_time_step    = 1second,
                cfl                  = 0.3,
                max_time_step        = 3minutes,
                max_time_step_change = 1.01,
            ),
            initial_conditions = initial_conditions_for_run(
                initial_conditions_source,
                INNEROSLOFJORDEN_UNIFORM_INITIAL_CONDITIONS;
                results_directory = initial_results_directory,
            ),
            # The whole of 2020, a leap year, so 366 days from midnight lands on midnight. The
            # prepare steps pad their time axes to reach both ends of this window.
            start_date         = DateTime(2020, 1, 1),
            stop_time          = 366days,
            loops              = 1,
            pickup             = false,
        ),
    )
end

inneroslofjorden()
