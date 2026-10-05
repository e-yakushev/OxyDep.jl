using FjordSim
using Oceananigans
using Oceananigans.Units: day, days, hour, hours, minutes, second
using Oceananigans.TimeSteppers: QuasiAdamsBashforth2TimeStepper
using FjordSim.Bathymetry: GEONORGE_DYBDEDATA_GDB
using Dates: DateTime
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using NumericalEarth: FreezingLimitedOceanTemperature

"""
    inneroslofjorden()

The Inneroslofjord setup: Geonorge Sjøkart Dybdedata bathymetry, NorKyst-800m forcing with NVE
rivers, and NORA3 atmosphere, on a 170 x 256 x 18 grid covering 10.20-10.50°E, 59.53-59.76°N — 99 m
cells, and the whole fjord from the Drammenselva and Lierelva mouths down past the Svelvik sill to
the last narrows before Breiangen. It runs the calendar year 2020, as `oslofjorden()` does.

The southern edge is at 59.53°N and the domain ends at 10.50°E because that is the southernmost cut
whose *eastern* wall is still land: below 59.52°N Breiangen reaches past 10.50°E, so a closed
eastern wall would sit in open water across the mouth.

The FileGDB, the daily NorKyst months and the NORA3 months are read from
`~/FjordSim_data/oslofjorden/` by absolute path, since Oslofjord's domain contains this one and both
downloaders skip a month already present. The boundary band and the NVE caches are not shared — see
`docs/setups.md`.
"""
function inneroslofjorden()
    data_root = joinpath(homedir(), "FjordSim_data", "inneroslofjorden")
    oslofjorden_data_root = joinpath(homedir(), "FjordSim_data", "oslofjorden")
    FT = Oceananigans.defaults.FloatType

    return FjordConfig(
        # Overloads LatitudeLongitudeGrid(architecture, config) — the one grid hook.

        grid_config = EvenGrid(
            # Inner Oslofjord grid
            size      = (63, 114, 17),
            halo      = (7, 7, 7), #(7, 7, 7),
            longitude = (10.46, 10.79),
            latitude  = (59.62, 59.92),
            z_faces   = [                
             -200.0, -180.0, -160.0, -140.0, -120.0, -100.0, -80.0, -60.0, 
             -40.0, -24.0, -15.0, -10.0, -7.5, -5.0, -3.0, -2.0, -1.0, 0.0,
            ],
        ),
        # Overloads bathymetry_dataset (required), regrid_options and smoothing_options (both
        # optional) — the hooks prepare_bathymetry dispatches on.
        bathymetry_config = DybdedataConfig(
            data_root             = data_root,
            output_file           = "bathymetry.nc",
            plot_file             = "bathymetry.png",
            # Oslofjord's extracted copy of the national FileGDB, by absolute path.
            geodatabase_file      = joinpath(oslofjorden_data_root, GEONORGE_DYBDEDATA_GDB),
            raw_resolution_factor = 2,
            padding_cells         = 2,
            include_contours      = false,
            contour_stride        = 10,
            interpolation_passes  = 1,
            # One basin, which is what drops the lakes: Geonorge's soundings include freshwater
            # bodies, and this domain holds several. It also drops Sandebukta, which enters the
            # fjord from the south and is therefore disconnected at this cut.
            major_basins          = 1,
            # The Oslofjord smoothing set, which this setup carried none of; `DybdedataConfig`
            # documents the failure each stage prevents. `open_boundary_land_cells` is 2 rather
            # than Oslofjord's 5 because half of this southern row is land, so the stage has far
            # more to flood here, and every cell it floods lacks an exterior profile of its own.
            minimum_depth         = 2.0, #2.0,
            open_boundary_land_cells =  1, #2,
            max_island_cells      = 3, #6,
            close_narrow_passages = true,
            spike_ratio           = 0.25, #0.5,
            # Must equal the `minimum_fractional_cell_height` `Grids.jl` gives `PartialCellBottom`.
            minimum_cell_fraction = 0.2,
            max_slope_factor      = 0.5, #0.25,
            geonorge_cache        = true,
            regrid_cache          = false,
        ),
        # Overloads forcing_time_steps, forcing_source_grid, forcing_variable_names and
        # download_forcing (required) — the hooks prepare_forcing and download_forcing dispatch on.
        forcing_config = NorKystConfig(
            data_root        = data_root,
            # Absolute: Oslofjord's twelve daily NorKyst months are read rather than fetched again.
            output_directory = joinpath(oslofjorden_data_root, "norkyst"),
            output_file      = "forcing.nc",
            plot_file        = "forcing.png",
            architecture     = :auto,
            parameters       = ["temperature", "salinity", "u_eastward", "v_northward"],
            years            = [2020],
            # Overloads river_locations, river_series and download_rivers (all required), plus
            # river_minimum_levels, river_plume_depth and river_lambdas (all optional). Its own
            # `data_root`: the ELVIS and REGINE caches record the domain they were fetched for
            # under a fixed name, so sharing Oslofjord's `nve` directory would overwrite them.
            rivers           = NVERiversConfig(
                data_root         = data_root,
                output_file       = "forcing_rivers_nve.nc",
                plot_file         = "forcing_rivers_nve.png",
                years             = [2020],
                # Discovery at Oslofjord's threshold: every mouth NVE has in this domain carrying
                # 0.5 m³/s or more, sized by its own REGINE catchment. It replaced an
                # `OF800RiversConfig` that landed one river in the domain, 602 m from the nearest
                # water. No river coordinate appears in this file.
                minimum_discharge = 0.5,
                # 5 m of surface plume, which is 4 cells on the vertical grid above.
                default_plume_depth = 5.0,
                minimum_levels    = 0,
                # A stability bound: at 99 m cells Drammenselva's peak discharge reaches λ·Δt > 1,
                # which `ForcingFromFile` reads as an x-flux.
                minimum_relaxation_timescale = 600.0,
                # This smaller domain does not contain Drammenselva or Lierelva, so there are no
                # out-of-domain gauge overrides. The discovered mouths are used as-is.
                outlets          = NVERiver[],
            ),
        ),
        # The hourly exterior state along the open southern edge, and the one place that edge is
        # stated. Its own download directory: the band is a thin strip along *this* grid's southern
        # edge, 60 km north of Oslofjord's, so it is different data rather than a subset.
        boundary_config = NorKystBoundariesConfig(
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
        # Overloads atmosphere_time_steps, atmosphere_source_grid, atmosphere_variable_names,
        # download_atmosphere, prescribed_atmosphere and prescribed_radiation.
        atmosphere_config = NORA3Config(
            data_root        = data_root,
            # Absolute, like the NorKyst directory above: NORA3 is ~10000 OPeNDAP reads per year,
            # and Oslofjord's subset already covers this domain.
            output_directory = joinpath(oslofjorden_data_root, "nora3"),
            output_file      = "atmosphere.nc",
            plot_file        = "atmosphere.png",
            resolution       = 0.02,
            padding          = 0.1,
            years            = [2020],
        ),
        # SimulationConfig itself has no hooks — everything below dispatches through one of its
        # four nested configs instead.
        simulation_config = SimulationConfig(
            results_root       = joinpath(homedir(), "FjordSim_results", "drammensfjorden"),
            architecture       = :auto,
            # Overloads coupled_simulation and model_tracers.
            model              = CoupledHydrostaticSimulation(
                buoyancy           = SeawaterBuoyancy(FT, equation_of_state = TEOS10EquationOfState(FT)),
                # A `BoundarySponge` for the same reason `oslofjorden()` has one: nothing else damps
                # what the open boundary radiates. 13 m² s⁻¹ rather than Oslofjord's 30, because
                # explicit `Δt ≤ Δx²/4ν` is 82 s at 30 on a 99 m cell — below `max_time_step`, so
                # the sponge and not the CFL would silently set the time step. At 13 it is 189 s.
                # `width_cells` stays 16: what the sponge damps is grid-scale noise, so its width
                # is a count of wavelengths rather than a distance.
                closure            = BoundarySponge(
                    base = (
                        # CATKE takes `w★ = sqrt(max(minimum_tke, e))`, so this floor, not the
                        # prognostic TKE, sets κ = 0.098·e_min/N over most of the column. See
                        # `oslofjorden()` for the measurements behind the value.
                        #CATKEVerticalDiffusivity(minimum_tke = 7e-7, maximum_tracer_diffusivity = 1e-2), #% (minimum_tke = 7e-6)
                        CATKEVerticalDiffusivity(minimum_tke = 9e-7, maximum_tracer_diffusivity = 1e-2), #% (minimum_tke = 7e-6)
                        # Scaled with Δx⁴ for the ~300 m cell: 1e5 gives an 81 min e-folding
                        # (`Δx⁴/16ν₄`) and a 2500 s explicit limit `Δt ≤ Δx⁴/32ν₄` that
                        # `AdaptiveTimeStep` never measures.
                  #      HorizontalScalarDiffusivity(ν = 5.0, κ = 2.0),
                        HorizontalScalarBiharmonicDiffusivity(ν = 1e3, κ = 1e2),
                    ),
                    width_cells = 16,
                    viscosity   = 13.0,
                    diffusivity = 6.5,
                ),
                # One scheme for every tracer: Oceananigans gives any tracer a NamedTuple omits the
                # `Centered()` default, and CATKE contributes an `e` this setup never names, so
                # `(T = WENO(), S = WENO())` left the TKE on an unbounded centered scheme.
                tracer_advection   = WENO(),
                momentum_advection = WENOVectorInvariant(FT),
                tracers            = (:T, :S),
                coriolis           = HydrostaticSphericalCoriolis(FT),
                sea_ice            = FreezingLimitedOceanTemperature(),
                biogeochemistry    = nothing,
                # Overloads free_surface(config, grid), called from inside coupled_simulation once
                # the grid exists.
                free_surface       = SplitExplicitFreeSurfaceConfig(cfl = 0.7),
                # Anything else the four constructors coupled_simulation calls accept, one slot
                # each: :ocean_model, :ocean_simulation, :coupled_model, :coupled_simulation.
                # QuasiAdamsBashforth2 with χ = -1/2 is forward Euler.
                # extra_kwargs       = (ocean_model = (timestepper = QuasiAdamsBashforth2TimeStepper(χ = -0.5),),),
                extra_kwargs       = (;),
            ),
            # MergedBoundaryConditions overloads field_boundary_conditions; each piece inside it
            # overloads boundary_condition_sides, so any one can be swapped alone. Dropping the last
            # would close the domain; argument order is merge precedence. The two timescales are
            # Marchesiello et al. (2001), and 3 hours rather than a day because at the ~10 s steps
            # these runs take a one-day timescale is ~200x weaker than advection into the boundary
            # cell.
            boundary_conditions = MergedBoundaryConditions(
                AirSeaFluxes(),
                QuadraticBottomDrag(coefficient = 0.003),
                OpenLateralBoundaryFromData(
                    inflow_timescale  = 3hours,
                    outflow_timescale = 360days,
                ),
            ),
            # Both overload attach_writer!. `variables` may name anything `Oceananigans.fields`
            # exposes on the ocean model; dropping the `CheckpointWriter` turns checkpointing off.
            writers = (
                SnapshotWriter(
                    name               = :ocean,
                    output_file        = "snapshots_ocean.nc",
                    variables          = (:T, :S, :u, :v),
                    interval           = 3hours,
                    overwrite_existing = true,
                ),
                CheckpointWriter(interval = 12hours, cleanup = true),
            ),
            # Overloads attach_callback! — what the run reports while it runs. `report` is the
            # function itself, so a model whose tracers omit :T (which `progress` reads) names its
            # own here instead. An empty tuple runs silently.
            callbacks = (ProgressCallback(name = :progress, interval = 24hours, report = progress),),
            # Overloads attach_time_stepping! and initial_time_step.
            time_stepping = AdaptiveTimeStep(
                initial_time_step    = 1second,
                cfl                  = 0.3,
                max_time_step        = 3minutes,
                max_time_step_change = 1.01,
            ),
            # The NorKyst state at `start_date`: every tracer the model names plus u and v,
            # whichever of them the prepared forcing file carries.
            initial_conditions = FromForcing(),
            # The whole of 2020, a leap year, so 366 days from midnight lands on midnight. Both
            # prepare steps pad their time axes to reach the ends of this window.
            start_date         = DateTime(2020, 1, 1),
            stop_time          = 366days,
            # One pass. Raise it to spin the basin up on the same forcing year, carrying the state
            # over; each repetition writes its own `_loopNN` file.
            loops              = 1,
            pickup             = false,
        ),
    )
end

inneroslofjorden()