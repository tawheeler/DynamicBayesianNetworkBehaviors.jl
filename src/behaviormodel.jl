type DBNSimParams
    sampling_scheme::AbstractSampleMethod
    smoothing::Symbol # :none, :SMA, :WMA
    smoothing_counts::Int    # number of previous counts to use

    function DBNSimParams(
        sampling_scheme::AbstractSampleMethod=SAMPLE_UNIFORM,
        smoothing::Symbol=:none,
        smoothing_counts::Int=1
        )

        @assert(smoothing_counts > 0)
        new(sampling_scheme, smoothing, smoothing_counts)
    end
end
type DynamicBayesianNetworkBehavior <: AbstractVehicleBehavior

    model         :: DBNModel

    ind_lat       :: Int
    ind_lon       :: Int
    symbol_lat    :: Symbol
    symbol_lon    :: Symbol
    simparams_lat :: DBNSimParams
    simparams_lon :: DBNSimParams

    indicators    :: Vector{AbstractFeature}
    ordering      :: Vector{Int}

    # preallocated memory
    observations  :: Dict{Symbol,Float64}
    assignment    :: Dict{Symbol,Int}
    logPs         :: Dict{Symbol,Float64}
    ind_lat_in_discretizers :: Int
    ind_lon_in_discretizers :: Int
    binprobs_lat  :: Vector{Float64}
    binprobs_lon  :: Vector{Float64}
    temp_binprobs_lat :: Vector{Float64}
    temp_binprobs_lon :: Vector{Float64}

    function DynamicBayesianNetworkBehavior(
        model::DBNModel,
        simparams_lat::DBNSimParams=DBNSimParams(),
        simparams_lon::DBNSimParams=DBNSimParams()
        )

        retval = new()
        retval.model = model

        targets = get_targets(model)

        f_lat = get_target_lat(model, targets)
        retval.ind_lat = indexof(f_lat, model)
        retval.symbol_lat = symbol(f_lat)

        f_lon = get_target_lon(model, targets)
        retval.ind_lon = indexof(f_lon, model)
        retval.symbol_lon = symbol(f_lon)


        retval.simparams_lat = simparams_lat
        retval.simparams_lon = simparams_lon

        retval.indicators = get_indicators(model)
        retval.ordering = topological_sort_by_dfs(model.BN.dag)
        retval.observations = Dict{Symbol,Float64}()
        retval.assignment = Dict{Symbol,Int}()
        retval.logPs = Dict{Symbol,Float64}()

        for f in retval.indicators
            sym = symbol(f)
            retval.observations[sym] = NaN
            retval.assignment[sym] = 0
        end
        retval.logPs[retval.symbol_lat] = NaN
        retval.logPs[retval.symbol_lon] = NaN

        retval.ind_lat_in_discretizers = findfirst(model.BN.names, retval.symbol_lat)
        retval.ind_lon_in_discretizers = findfirst(model.BN.names, retval.symbol_lon)

        retval.binprobs_lat = Array(Float64, nlabels(model.discretizers[retval.ind_lat_in_discretizers]))
        retval.binprobs_lon = Array(Float64, nlabels(model.discretizers[retval.ind_lon_in_discretizers]))
        retval.temp_binprobs_lat = deepcopy(retval.binprobs_lat)
        retval.temp_binprobs_lon = deepcopy(retval.binprobs_lon)

        retval
    end
end

# function infer_action_lon_from_input_acceleration(sym::Symbol, accel::Float64, simlog::Matrix{Float64}, frameind::Int, logindexbase::Int)

#     if sym == :f_accel_250ms || sym == :f_accel_500ms
#         return accel
#     elseif sym == :f_des_speed_250ms || sym == :f_des_speed_500ms
#         return accel/Features.KP_DESIRED_SPEED
#     else
#         error("unknown longitudinal target $sym")
#     end
# end
# function infer_action_lat_from_input_turnrate(sym::Symbol, turnrate::Float64, simlog::Matrix{Float64}, frameind::Int, logindexbase::Int)

#     if sym == :f_turnrate_250ms || sym == :f_turnrate_500ms
#         return turnrate
#     elseif sym == :f_des_angle_250ms || sym == :f_des_angle_500ms
#         ϕ = simlog[frameind, logindexbase + LOG_COL_ϕ]
#         return (turnrate / Features.KP_DESIRED_ANGLE) + ϕ
#     else
#         error("unknown lateral target $sym")
#     end
# end

##############################################################

sample!(behavior::DynamicBayesianNetworkBehavior, assignment::Dict{Symbol, Int}) = sample!(behavior.model, assignment, behavior.ordering)
sample_and_lopP!(behavior::DynamicBayesianNetworkBehavior, assignment::Dict{Symbol, Int}, logPs::Dict{Symbol, Float64}=Dict{Symbol, Float64}()) = sample!(behavior.model, assignment, logPs, behavior.ordering)

function select_action(
    basics::FeatureExtractBasicsPdSet,
    behavior::DynamicBayesianNetworkBehavior,
    carind::Int,
    validfind::Int
    )

    model = behavior.model
    symbol_lat = behavior.symbol_lat
    symbol_lon = behavior.symbol_lon

    simparams_lat = behavior.simparams_lat
    simparams_lon = behavior.simparams_lon
    samplemethod_lat = simparams_lat.sampling_scheme
    samplemethod_lon = simparams_lon.sampling_scheme
    smoothing_lat = simparams_lat.smoothing
    smoothing_lon = simparams_lon.smoothing
    smoothcounts_lat = simparams_lat.smoothing_counts
    smoothcounts_lon = simparams_lon.smoothing_counts

    bmap_lat = model.discretizers[behavior.ind_lat_in_discretizers]
    bmap_lon = model.discretizers[behavior.ind_lon_in_discretizers]

    observations = behavior.observations
    assignment = behavior.assignment

    Features.observe!(observations, basics, carind, validfind, behavior.indicators)
    encode!(assignment, model, observations)
    sample!(model, assignment, behavior.ordering)

    bin_lat = assignment[symbol_lat]
    bin_lon = assignment[symbol_lon]

    action_lat = decode(bmap_lat, bin_lat, samplemethod_lat)
    action_lon = decode(bmap_lon, bin_lon, samplemethod_lon)

    @assert(!isinf(action_lat))
    @assert(!isinf(action_lon))
    @assert(!isnan(action_lat))
    @assert(!isnan(action_lon))

    # action_lat = clamp(action_lat, -0.05, 0.05) # TODO(tim): remove this
    # action_lon = clamp(action_lon, -3.0, 1.5) # TODO(tim): remove this

    (action_lat, action_lon)
end

function _calc_action_loglikelihood(
    behavior::DynamicBayesianNetworkBehavior,
    action_lat::Float64,
    action_lon::Float64,
    )

    model = behavior.model
    symbol_lat = behavior.symbol_lat
    symbol_lon = behavior.symbol_lon
    bmap_lat = model.discretizers[behavior.ind_lat_in_discretizers]
    bmap_lon = model.discretizers[behavior.ind_lon_in_discretizers]

    bin_lat = encode(bmap_lat, action_lat)
    bin_lon = encode(bmap_lon, action_lon)

    observations = behavior.observations # assumed to already be populated
    assignment   = behavior.assignment   # this will be overwritten
    logPs        = behavior.logPs        # this will be overwritten
    binprobs_lat = behavior.binprobs_lat # this will be overwritten
    binprobs_lon = behavior.binprobs_lon # this will be overwritten

    encode!(assignment, model, observations)

    # TODO(tim): put this back in; temporarily removed for debugging
    if is_parent(model, symbol_lon, symbol_lat) # lon -> lat
        calc_probability_distribution_over_assignments!(binprobs_lon, model, assignment, symbol_lon)
        fill!(binprobs_lat, 0.0)
        temp = behavior.temp_binprobs_lon
        for (i,p) in enumerate(binprobs_lon)
            assignment[symbol_lon] = i
            calc_probability_distribution_over_assignments!(temp, model, assignment, symbol_lat)
            for (j,v) in enumerate(temp)
                binprobs_lat[j] +=  v * p
            end
        end
    elseif is_parent(model, symbol_lat, symbol_lon) # lat -> lon
        calc_probability_distribution_over_assignments!(binprobs_lat, model, assignment, symbol_lat)
        fill!(binprobs_lon, 0.0)
        temp = behavior.temp_binprobs_lat
        for (i,p) in enumerate(binprobs_lat)
            assignment[symbol_lat] = i
            calc_probability_distribution_over_assignments!(temp, model, assignment, symbol_lon)
            for (j,v) in enumerate(temp)
                binprobs_lon[j] +=  v * p
            end
        end
    else # lat and lon are conditionally independent
        calc_probability_distribution_over_assignments!(binprobs_lat, model, assignment, symbol_lat)
        calc_probability_distribution_over_assignments!(binprobs_lon, model, assignment, symbol_lon)
    end

    P_bin_lat = binprobs_lat[bin_lat]
    P_bin_lon = binprobs_lon[bin_lon]

    p_within_bin_lat = calc_probability_for_uniform_sample_from_bin(P_bin_lat, bmap_lat, bin_lat)
    p_within_bin_lon = calc_probability_for_uniform_sample_from_bin(P_bin_lon, bmap_lon, bin_lon)

    # println("actions: ", action_lat, "  ", action_lon)

    log(p_within_bin_lat) + log(p_within_bin_lon)
end
function calc_action_loglikelihood(
    basics::FeatureExtractBasicsPdSet,
    behavior::DynamicBayesianNetworkBehavior,
    carind::Int,
    validfind::Int,
    action_lat::Float64,
    action_lon::Float64,
    )

    model = behavior.model
    symbol_lat = behavior.symbol_lat
    symbol_lon = behavior.symbol_lon
    bmap_lat = model.discretizers[findfirst(model.BN.names, symbol_lat)]
    bmap_lon = model.discretizers[findfirst(model.BN.names, symbol_lon)]

    if min(bmap_lat) ≤ action_lat ≤ max(bmap_lat) &&
       min(bmap_lon) ≤ action_lon ≤ max(bmap_lon)

        Features.observe!(behavior.observations, basics, carind, validfind, behavior.indicators)

        _calc_action_loglikelihood(behavior, action_lat, action_lon)
    else
        print_with_color(:red, STDOUT, "\nDynamicBayesianNetworkBehaviors calc_log_prob: HIT\n")
        print_with_color(:red, STDOUT, "validfind: $validfind\n")
        print_with_color(:red, STDOUT, "$(min(bmap_lat))  $action_lat $(max(bmap_lat))\n")
        print_with_color(:red, STDOUT, "$(min(bmap_lon))  $action_lon $(max(bmap_lon))\n")
        -Inf
    end
end
function calc_action_loglikelihood(
    behavior::DynamicBayesianNetworkBehavior,
    features::DataFrame,
    frameind::Integer,
    )

    action_lat = features[frameind, symbol(FUTUREDESIREDANGLE_250MS)]::Float64
    action_lon = features[frameind, symbol(FUTUREACCELERATION_250MS)]::Float64

    model = behavior.model
    symbol_lat = behavior.symbol_lat
    symbol_lon = behavior.symbol_lon
    bmap_lat = model.discretizers[findfirst(model.BN.names, symbol_lat)]
    bmap_lon = model.discretizers[findfirst(model.BN.names, symbol_lon)]


    # action_lat = clamp(action_lat, min(bmap_lat), max(bmap_lat))
    # action_lon = clamp(action_lon, min(bmap_lon), max(bmap_lon))

    if min(bmap_lat) ≤ action_lat ≤ max(bmap_lat) &&
       min(bmap_lon) ≤ action_lon ≤ max(bmap_lon)

        for name in keys(behavior.observations)
            behavior.observations[name] = features[frameind, name]
        end

        _calc_action_loglikelihood(behavior, action_lat, action_lon)
    else
        print_with_color(:red, STDOUT, "\nDynamicBayesianNetworkBehaviors calc_log_prob: HIT\n")
        print_with_color(:red, STDOUT, "frameind: $frameind\n")
        print_with_color(:red, STDOUT, "$(min(bmap_lat))  $action_lat $(max(bmap_lat))\n")
        print_with_color(:red, STDOUT, "$(min(bmap_lon))  $action_lon $(max(bmap_lon))\n")
        -Inf
    end
end