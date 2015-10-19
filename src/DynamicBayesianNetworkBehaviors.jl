module DynamicBayesianNetworkBehaviors

using AutomotiveDrivingModels
using NLopt
using Distributions
using BayesNets
using SmileExtra

import Graphs: topological_sort_by_dfs, in_neighbors, num_vertices, num_edges
import Discretizers: encode
import AutomotiveDrivingModels: ModelTargets, AbstractVehicleBehavior, select_action, calc_action_loglikelihood,
                                train, observe, _reverse_smoothing_sequential_moving_average

export
    DBNModel,
    DynamicBayesianNetworkBehavior,
    DBNSimParams,
    GraphLearningResult,
    ParentFeatures,

    DEFAULT_INDICATORS,
    DEFAULT_DISCRETIZERS,

    dbnmodel,
    build_bn,

    is_target_lat,
    is_target_lon,

    indexof,
    is_parent,
    parent_indeces,

    get_targets,
    get_target_lat,
    get_target_lon,
    get_indicators,
    get_indicators_for_target,
    get_indicators_lat,
    get_indicators_lon,

    find_target_indeces,

    get_num_vertices,
    get_num_edges,

    get_total_sample_count,
    get_bin_counts,
    get_marginal_probability,
    get_counts_for_assignment,
    get_emstats,

    encode,
    sample!,
    sample_and_logP!,
    calc_log_probability_of_assignment,
    calc_probability_distribution_over_assignments!,
    calc_probability_for_uniform_sample_from_bin,

    export_to_text,
    print_structure,

    model_adjacency_matrix,

    get_input_acceleration,
    get_input_turnrate,
    # infer_action_lon_from_input_acceleration,
    # infer_action_lat_from_input_turnrate,

    discretize,
    discretize_cleaned,
    drop_invalid_discretization_rows,
    convert_dataset_to_matrix,
    feature_indeces_in_net,
    calc_bincounts_array,

    select_action,
    calc_action_loglikelihood,
    train

immutable DBNModel
    BN            :: BayesNet
    statsvec      :: Vector{Matrix{Float64}} # each matrix is [r×q], nvar_instantiations × nparent_instantiations
    features      :: Vector{AbstractFeature}
    discretizers  :: Vector{AbstractDiscretizer}
    istarget      :: BitVector # whether a feature is a target feature
end

function dbnmodel{R<:Real, D<:AbstractDiscretizer, F<:AbstractFeature, G<:AbstractFeature}(
    BN:: BayesNet,
    statsvec::Vector{Matrix{R}},
    discretizerdict::Dict{Symbol, D},
    targets::Vector{F},
    indicators::Vector{G}
    )

    features = AbstractFeature[symbol2feature(sym) for sym in BN.names]
    discretizers = AbstractDiscretizer[discretizerdict[sym] for sym in BN.names]
    istarget = falses(length(features))
    for (i,f) in enumerate(features)
        if in(f, targets)
            istarget[i] = true
        elseif !in(f, indicators)
            error("Feature not in targets or indicators")
        end
    end

    statsvec_float = convert(Vector{Matrix{Float64}}, statsvec)
    DBNModel(BN, statsvec_float, features, discretizers, istarget)
end
function dbnmodel{R<:Real, D<:AbstractDiscretizer, F<:AbstractFeature, G<:AbstractFeature}(
    BN::BayesNet,
    statsvec::Vector{Matrix{R}},
    discretizers::Vector{D},
    targets::Vector{F},
    indicators::Vector{G}
    )

    features = AbstractFeature[symbol2feature(sym) for sym in BN.names]
    istarget = falses(length(features))
    for (i,f) in enumerate(features)
        if in(f, targets)
            istarget[i] = true
        elseif !in(f, indicators)
            error("Feature $(symbol(f)) neither in targets nor indicators")
        end
    end

    statsvec_float = convert(Vector{Matrix{Float64}}, statsvec)
    DBNModel(BN, statsvec_float, features, discretizers, istarget)
end
function dbnmodel{R<:Real, D<:AbstractDiscretizer, F<:AbstractFeature, G<:AbstractFeature}(
    adj::BitMatrix,
    statsvec::Vector{Matrix{R}},
    discretizerdict::Dict{Symbol, D},
    targets::Vector{F},
    indicators::Vector{G}
    )

    BN = build_bn(statsvec, targets, indicators, adj)
    dbnmodel(BN, statsvec, discretizerdict, targets, indicators)
end
function dbnmodel(modelstats::Dict{String, Any})

    discretizerdict = modelstats["binmaps"]
    targets    = modelstats["targets"]
    indicators = modelstats["indicators"]
    stats      = modelstats["statistics"]
    adj        = modelstats["adjacency"]

    dbnmodel(adj, stats, discretizerdict, convert(Vector{AbstractFeature}, targets),
                        convert(Vector{AbstractFeature}, indicators))
end
function dbnmodel(modelpstats_file::String)
    emstats = load(modelpstats_file)
    dbnmodel(emstats)
end

function build_bn{R<:Real, F<:AbstractFeature, G<:AbstractFeature}(
    statsvec   :: Vector{Matrix{R}},
    targets    :: Vector{F},
    indicators :: Vector{G},
    adj        :: BitMatrix
    )

    bnnames = [symbol(f)::Symbol for f in [targets, indicators]]
    @assert(length(unique(bnnames)) == length(bnnames)) # NOTE(tim): currently does not support targets also being indicators
    n_nodes = length(bnnames)
    @assert(n_nodes == size(adj, 1) == size(adj, 2))

    BN = BayesNet(bnnames)

    r_arr = Array(Int, n_nodes)
    for (node,node_sym) in enumerate(bnnames)
        stats = statsvec[node]
        r, q = size(stats) # r = num node instantiations, q = num parental instantiations
        states = [1:r]
        BN.domains[node] = DiscreteDomain(states)
        r_arr[node] = r
    end

    for (node,node_sym) in enumerate(bnnames)

        stats = statsvec[node]
        r, q = size(stats)
        states = [1:r]

        stats .+= 1 # NOTE(tim): adding uniform prior
        probabilities = stats ./ sum(stats,1)

        # set any parents & populate probability table
        n_parents = sum(adj[:,node])
        if n_parents > 0
            bnparents = bnnames[adj[:,node]]
            for pa in bnparents
                addEdge!(BN, pa, node_sym)
            end

            # populate probability table
            assignments = BayesNets.assignment_dicts(BN, bnparents)
            # parameterFunction = BayesNets.discrete_parameter_function(assignments, vec(probabilities), r)
            # setCPD!(BN, node_sym, CPDs.Discrete(states, parameterFunction))

            parameterlookup = BayesNets.discrete_parameter_dict(assignments, vec(probabilities), r)
            setCPD!(BN, node_sym, CPDs.DiscreteDictCPD(states, parameterlookup))
        else
            # no parents
            # setCPD!(BN, node_sym, CPDs.Discrete(states, vec(probabilities)))

            setCPD!(BN, node_sym, CPDs.DiscreteStaticCPD(states, vec(probabilities)))
        end

    end

    return BN
end

function is_target_lat(f::AbstractFeature)
    isa(f, Features.Feature_FutureTurnRate_250ms) ||
    isa(f, Features.Feature_FutureTurnRate_500ms) ||
    isa(f, Features.Feature_FutureDesiredAngle_250ms) ||
    isa(f, Features.Feature_FutureDesiredAngle_500ms)
end
function is_target_lon(f::AbstractFeature)
    isa(f, Features.Feature_FutureDesiredSpeed_250ms) ||
    isa(f, Features.Feature_FutureDesiredSpeed_500ms) ||
    isa(f, Features.Feature_FutureAcceleration_250ms) ||
    isa(f, Features.Feature_FutureAcceleration_500ms)
end

indexof(f::Symbol, model::DBNModel) = model.BN.index[f]
indexof(f::AbstractFeature, model::DBNModel) = model.BN.index[symbol(f)]
is_parent(model::DBNModel, parent::Int, child::Int) = in(parent, in_neighbors(child, model.BN.dag))
is_parent(model::DBNModel, parent::Symbol, child::Symbol) = is_parent(model, model.BN.index[parent], model.BN.index[child])
function parent_indeces(varindex::Int, model::DBNModel)
    parent_names = BayesNets.parents(model.BN, model.BN.names[varindex])
    retval = Array(Int, length(parent_names))
    for (i, name) in enumerate(parent_names)
        retval[i] = model.BN.index[name]
    end
    retval
end

get_targets(model::DBNModel) = model.features[model.istarget]
function get_target_lat(model::DBNModel, targets::Vector{AbstractFeature}=get_targets(model))
    ind = findfirst(f->is_target_lat(f), targets)
    targets[ind]
end
function get_target_lon(model::DBNModel, targets::Vector{AbstractFeature}=get_targets(model))
    ind = findfirst(f->is_target_lon(f), targets)
    targets[ind]
end

get_indicators(model::DBNModel) = model.features[!model.istarget]
function get_indicators_for_target(model::DBNModel, i::Int)
    indicator_indeces = parent_indeces(i, model)
    model.features[indicator_indeces]
end
function get_indicators_for_target(model::DBNModel, target::AbstractFeature)
    i = indexof(target, model)
    get_indicators_for_target(model, i)
end
get_indicators_lat(model::DBNModel) = get_indicators_for_target(model, get_target_lat(model))
get_indicators_lon(model::DBNModel) = get_indicators_for_target(model, get_target_lon(model))

function find_target_indeces{F<:AbstractFeature}(targetset::ModelTargets, features::Vector{F})
    ind_lat = findfirst(f->f==targetset.lat, features)
    ind_lon = findfirst(f->f==targetset.lon, features)
    @assert(ind_lat  != 0)
    @assert(ind_lon  != 0)
    (ind_lat, ind_lon)
end

get_num_vertices(model::DBNModel) = num_vertices(model.BN.dag)
get_num_edges(model::DBNModel) = num_edges(model.BN.dag)

get_total_sample_count(model::DBNModel) = sum(model.statsvec[1])
function get_bin_counts(model::DBNModel)
    n_nodes = get_num_vertices(model)
    r_arr = Array(Int, n_nodes)
    for i = 1 : n_nodes
        r_arr[i] = size(model.statsvec[i], 1)
    end
    r_arr
end
function get_marginal_probability(model::DBNModel, varindex::Int)
    binprobs = vec(sum(model.statsvec[varindex], 2))
    binprobs ./= sum(binprobs)
end
function get_counts_for_assignment(
    model::DBNModel,
    targetind::Int,
    parentindeces::Vector{Int},
    parentassignments::Vector{Int},
    bincounts::Vector{Int}
    )

    dims = tuple([bincounts[parentindeces]]...)
    subs = tuple(parentassignments...)
    j = sub2ind(dims, subs...)
    copy(model.statsvec[targetind][:,j])
end
function get_counts_for_assignment(
    model::DBNModel,
    targetind::Int,
    assignments::Dict{Symbol, Int},
    bincounts::Vector{Int}
    )

    parentindeces = parent_indeces(targetind, model)
    nparents = length(parentindeces)
    parentassignments = Array(Int, nparents)
    for (i,ind) in enumerate(parentindeces)
        parentassignments[i] = assignments[model.BN.names[ind]]
    end
    get_counts_for_assignment(model, targetind, parentindeces, parentassignments, bincounts)
end

function encode!(assignment::Dict{Symbol,Int}, model::DBNModel, observations::Dict{Symbol,Float64})
    # take each observation and bin it appropriately
    # returns a Dict{Symbol,Int}

    # TODO(tim): ensure order is correct
    for (i,istarget) in enumerate(model.istarget)
        if !istarget
            sym = model.BN.names[i]
            val = observations[sym]
            @assert(!isnan(val))
            assignment[sym] = encode(model.discretizers[i], val)
        end
    end
    assignment
end
function sample!(model::DBNModel, assignment::Dict{Symbol, Int}, ordering::Vector{Int}=topological_sort_by_dfs(model.BN.dag))
    #=
    Run through nodes in topological order, building the instantiation vector as we go
    We use nodes we already know to condition on the distribution for nodes we do not
    Modifies assignment to include newly sampled symbols
    =#

    # for name in model.BN.names[ordering]
    #     if !haskey(assignment, name)
    #         assignment[name] = BayesNets.rand(BayesNets.cpd(model.BN, name), assignment)
    #     end
    # end

    for name in model.BN.names[ordering]
        cpd = BayesNets.cpd(model.BN, name)

        p = probvec(cpd, assignment)
        r = rand()
        i = 1
        p_tot = 0.0
        while p_tot + p[i] < r && i < length(p)
            p_tot += p[i]
            i += 1
        end
        assignment[name] = cpd.domain[i]
    end

    assignment
end
function sample_and_logP!(
    model::DBNModel,
    assignment::Dict{Symbol, Int},
    logPs::Dict{Symbol, Float64},

    ordering::Vector{Int}=topological_sort_by_dfs(model.BN.dag),
    )

    for name in model.BN.names[ordering]
        cpd = BayesNets.cpd(model.BN, name)

        p = probvec(cpd, assignment)

        r = rand()
        i = 1
        p_tot = 0.0
        while p_tot + p[i] < r && i < length(p)
            p_tot += p[i]
            i += 1
        end

        assignment[name] = cpd.domain[i]
        logPs[name] = log(p[i])
    end

    logPs
end
function calc_log_probability_of_assignment(model::DBNModel, assignment::Dict{Symbol, Int}, sym::Symbol)
    # returns the discrete log probability of the given bin assignment
    cpd = BayesNets.cpd(model.BN, sym)
    b = assignment[sym]
    p = cpd.parameterFunction(assignment)
    log(p[b])
end
function calc_probability_distribution_over_assignments!(
    dest::Vector{Float64},
    model::DBNModel,
    assignment::Dict{Symbol, Int},
    target::Symbol
    )

    # NOTE (tim): cpd.parameterFunction(assignment) returns the actual probability vector, not a copy
    cpd = BayesNets.cpd(model.BN, target)
    copy!(dest, probvec(cpd, assignment))
    @assert(abs(sum(dest)-1.0) < 0.0001) # probability sums to 1.0
    dest
end
function calc_probability_for_uniform_sample_from_bin(
    discrete_prob::Float64,
    ::CategoricalDiscretizer, # disc
    ::Int # bin
    )

    discrete_prob
end
function calc_probability_for_uniform_sample_from_bin(
    discrete_prob::Float64,
    disc::LinearDiscretizer,
    bin::Int
    )

    # widthtot = totalwidth(disc)
    widthbin = binwidth(disc, bin)

    discrete_prob / widthbin # * widthtot
end
function calc_probability_for_uniform_sample_from_bin(
    discrete_prob::Float64,
    disc::HybridDiscretizer,
    bin::Int
    )

    if bin ≤ disc.lin.nbins
        calc_probability_for_uniform_sample_from_bin(discrete_prob, disc.lin, bin)
    else
        discrete_prob
    end
end

function export_to_text(model::DBNModel, filename::String)
    # export a bayesian network to an encounter definition file

    n = model.n # the number of nodes

    open(filename, "w") do fout

        # Labels: the node names
        ###############################################
        println(fout, "# labels")
        for i = 1 : n
            @printf(fout, "\"%s\"", string(model.labels[i]))
            if i < n
                @printf(fout, ", ")
            else
                @printf(fout, "\n")
            end
        end

        # G: the graphical structure
        ##################################################
        println(fout, "# G: graphical structure")
        for i = 1 : n
            for j = 1 : n
                @printf(fout, "%c ", model.G[i,j] ? '1' : '0')
            end
            @printf(fout, "\n")
        end

        # r: number of bins for each variable
        ##################################################
        println(fout, "# r: number of bins")
        for i = 1 : n
            @printf(fout, "%d ", model.r[i])
        end
        @printf(fout, "\n")

        # N: the sufficient statistics, in integer form
        ##################################################
        println(fout, "# N: sufficient statistics")
        for i = 1 : n
            stats = model.N[i]
            r, q = size(stats)
            for b = 1:q, a = 1:r
                @printf(fout, "%d ", stats[a,b])
            end
        end
        @printf(fout, "\n")

        # discretizers, '*' if discrete
        ##################################################
        @printf(fout, "# discretizers\n")
        for i = 1 : n
            bmap = model.discretizers[i]

            if isa(bmap, CategoricalDiscretizer)
                @printf(fout, "*")
            elseif isa(bmap, LinearDiscretizer)
                for edge in bmap.binedges
                    @printf(fout, "%f ", edge)
                end
            elseif isa(bmap, HybridDiscretizer)
                for edge in bmap.lin.binedges
                    @printf(fout, "%f ", edge)
                end
            else
                error("invalid bmap type $(typeof(bmap))")
            end
            @printf(fout, "\n")
        end
    end
end
function print_structure(model::DBNModel)
    target = get_target_lat(model)
    println("target lat: ", symbol(target))
    for indicator in get_indicators_lat(model, target)
        println("\t", symbol(indicator))
    end

    target = get_target_lon(model)
    println("target lon: ", symbol(target))
    for indicator in get_indicators_lon(model, target)
        println("\t", symbol(indicator))
    end
end

function model_adjacency_matrix(
    net_ind_lat :: Int,
    net_ind_lon :: Int,
    net_parent_indices_lat :: Vector{Int},
    net_parent_indices_lon :: Vector{Int},
    num_net_features :: Int
    )

    # [parent_index, child_index] = true for parent -> child edges

    adj = falses(num_net_features, num_net_features)

    adj[net_parent_indices_lat, net_ind_lat] = true
    adj[net_parent_indices_lon, net_ind_lon] = true

    adj
end

##############################################################

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

sample!(behavior::DynamicBayesianNetworkBehavior, assignment::Dict{Symbol, Int}) = sample!(behavior.model, assignment, behavior.ordering)
sample_and_lopP!(behavior::DynamicBayesianNetworkBehavior, assignment::Dict{Symbol, Int}, logPs::Dict{Symbol, Float64}=Dict{Symbol, Float64}()) = sample!(behavior.model, assignment, logPs, behavior.ordering)

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

export_to_text(behavior::DynamicBayesianNetworkBehavior) = export_to_text(behavior.model)

##############################################################

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

    action_lat = clamp(action_lat, -0.05, 0.05) # TODO(tim): remove this
    action_lon = clamp(action_lon, -3.0, 1.5) # TODO(tim): remove this

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

##############################################################

type ParentFeatures
    lat :: Vector{AbstractFeature}
    lon :: Vector{AbstractFeature}

    ParentFeatures() = new(AbstractFeature[], AbstractFeature[])
    ParentFeatures(lat::Vector{AbstractFeature}, lon::Vector{AbstractFeature}) = new(lat, lon)
end

type GraphLearningResult
    fileroot     :: String
    target_lat   :: AbstractFeature
    target_lon   :: AbstractFeature
    parents_lat  :: Vector{AbstractFeature}
    parents_lon  :: Vector{AbstractFeature}
    features     :: Vector{AbstractFeature}

    adj          :: BitMatrix               # this is of the size of the resulting network (ie, |f_inds|)
    stats        :: Vector{Matrix{Float64}} # NOTE(tim): this does not include prior counts
    bayescore    :: Float64

    function GraphLearningResult(
        basefolder     :: String,
        features       :: Vector{AbstractFeature},
        ind_lat        :: Int,
        ind_lon        :: Int,
        parentinds_lat :: Vector{Int}, # indices are given in terms of input features vector
        parentinds_lon :: Vector{Int},
        bayescore      :: Float64,
        r              :: AbstractVector{Int},
        d              :: AbstractMatrix{Int}
        )

        # 1 - build the set of features used in the net, ordered by [target_lat, target_lon, indicators]
        # 2 - pull the indeces for the parents of target_lat
        # 3 - pull the indeces for the parents of target_lon


        net_feature_indeces = feature_indeces_in_net(ind_lat, ind_lon, parentinds_lat, parentinds_lon)
        net_features = features[net_feature_indeces]
        net_ind_lat = 1
        net_ind_lon = 2
        net_parent_indices_lat = _find_index_mapping(parentinds_lat, net_feature_indeces)
        net_parent_indices_lon = _find_index_mapping(parentinds_lon, net_feature_indeces)
        num_net_features = length(net_feature_indeces)

        adj   = model_adjacency_matrix(net_ind_lat, net_ind_lon, net_parent_indices_lat, net_parent_indices_lon, num_net_features)
        stats = convert(Vector{Matrix{Float64}}, statistics(adj, r[net_feature_indeces], d[net_feature_indeces,:]))

        target_lat = features[ind_lat]
        target_lon = features[ind_lon]

        # need to permute these appropriately

        parents_lat = features[net_parent_indices_lat]
        parents_lon = features[net_parent_indices_lon]

        new(basefolder, target_lat, target_lon, parents_lat, parents_lon,
            net_features, adj, stats, bayescore)
    end
end

immutable ModelParams
    binmaps::Vector{AbstractDiscretizer} # in the same order as features
    parents_lat::Vector{Int} # indeces within features
    parents_lon::Vector{Int} # indeces within features
end
immutable ModelStaticParams
    ind_lat::Int
    ind_lon::Int
    features::Vector{AbstractFeature}
end
immutable ModelData
    continuous::Matrix{Float64} # NOTE(tim): this should never change
    discrete::Matrix{Int} # this should be overwritten as the discretization params change
    bincounts::Vector{Int} # this should be overwritten as the discretization params change

    function ModelData(continuous::Matrix{Float64}, modelparams::ModelParams, features)
        d = discretize(modelparams.binmaps, continuous, features)
        r = calc_bincounts_array(modelparams.binmaps)
        new(continuous, d, r)
    end
end

const DEFAULT_INDICATORS = [
                    YAW, SPEED, VELFX, VELFY, DELTA_SPEED_LIMIT,
                    D_CL, D_ML, D_MR, D_MERGE, D_SPLIT,
                    TIMETOCROSSING_RIGHT, TIMETOCROSSING_LEFT, TIMESINCELANECROSSING, ESTIMATEDTIMETOLANECROSSING,
                    N_LANE_L, N_LANE_R, HAS_LANE_L, HAS_LANE_R,
                    TURNRATE, TURNRATE_GLOBAL, ACC, ACCFX, ACCFY, A_REQ_STAYINLANE, LANECURVATURE,

                    HAS_FRONT, D_X_FRONT, D_Y_FRONT, V_X_FRONT, V_Y_FRONT, YAW_FRONT, TURNRATE_FRONT,
                    HAS_REAR,  D_X_REAR,  D_Y_REAR,  V_X_REAR,  V_Y_REAR,  YAW_REAR,  TURNRATE_REAR,
                               D_X_LEFT,  D_Y_LEFT,  V_X_LEFT,  V_Y_LEFT,  YAW_LEFT,  TURNRATE_LEFT,
                               D_X_RIGHT, D_Y_RIGHT, V_X_RIGHT, V_Y_RIGHT, YAW_RIGHT, TURNRATE_RIGHT,
                    A_REQ_FRONT, TTC_X_FRONT, TIMEGAP_X_FRONT,
                    A_REQ_REAR,  TTC_X_REAR,  TIMEGAP_X_REAR,
                    A_REQ_LEFT,  TTC_X_LEFT,  TIMEGAP_X_LEFT,
                    A_REQ_RIGHT, TTC_X_RIGHT, TIMEGAP_X_RIGHT,

                    SCENEVELFX,

                    TIME_CONSECUTIVE_BRAKE, TIME_CONSECUTIVE_ACCEL, TIME_CONSECUTIVE_THROTTLE,
                         PASTACC250MS,      PASTACC500MS,      PASTACC750MS,      PASTACC1S,
                    PASTTURNRATE250MS, PASTTURNRATE500MS, PASTTURNRATE750MS, PASTTURNRATE1S,
                       PASTVELFY250MS,    PASTVELFY500MS,    PASTVELFY750MS,    PASTVELFY1S,
                        PASTD_CL250MS,     PASTD_CL500MS,     PASTD_CL750MS,     PASTD_CL1S,

                         MAXACCFX500MS,     MAXACCFX750MS,     MAXACCFX1S,     MAXACCFX1500MS,     MAXACCFX2S,     MAXACCFX2500MS,     MAXACCFX3S,     MAXACCFX4S,
                         MAXACCFY500MS,     MAXACCFY750MS,     MAXACCFY1S,     MAXACCFY1500MS,     MAXACCFY2S,     MAXACCFY2500MS,     MAXACCFY3S,     MAXACCFY4S,
                      MAXTURNRATE500MS,  MAXTURNRATE750MS,  MAXTURNRATE1S,  MAXTURNRATE1500MS,  MAXTURNRATE2S,  MAXTURNRATE2500MS,  MAXTURNRATE3S,  MAXTURNRATE4S,
                        MEANACCFX500MS,    MEANACCFX750MS,    MEANACCFX1S,    MEANACCFX1500MS,    MEANACCFX2S,    MEANACCFX2500MS,    MEANACCFX3S,    MEANACCFX4S,
                        MEANACCFY500MS,    MEANACCFY750MS,    MEANACCFY1S,    MEANACCFY1500MS,    MEANACCFY2S,    MEANACCFY2500MS,    MEANACCFY3S,    MEANACCFY4S,
                     MEANTURNRATE500MS, MEANTURNRATE750MS, MEANTURNRATE1S, MEANTURNRATE1500MS, MEANTURNRATE2S, MEANTURNRATE2500MS, MEANTURNRATE3S, MEANTURNRATE4S,
                         STDACCFX500MS,     STDACCFX750MS,     STDACCFX1S,     STDACCFX1500MS,     STDACCFX2S,     STDACCFX2500MS,     STDACCFX3S,     STDACCFX4S,
                         STDACCFY500MS,     STDACCFY750MS,     STDACCFY1S,     STDACCFY1500MS,     STDACCFY2S,     STDACCFY2500MS,     STDACCFY3S,     STDACCFY4S,
                      STDTURNRATE500MS,  STDTURNRATE750MS,  STDTURNRATE1S,  STDTURNRATE1500MS,  STDTURNRATE2S,  STDTURNRATE2500MS,  STDTURNRATE3S,  STDTURNRATE4S,
                ]
const DEFAULT_DISCRETIZERS = Dict{Symbol,AbstractDiscretizer}()
    DEFAULT_DISCRETIZERS[:f_turnrate_250ms      ] = LinearDiscretizer([-0.025,-0.02,-0.015,-0.01,-0.005,0.005,0.01,0.015,0.02,0.025], Int)
    DEFAULT_DISCRETIZERS[:f_turnrate_500ms      ] = LinearDiscretizer([-0.025,-0.02,-0.015,-0.01,-0.005,0.005,0.01,0.015,0.02,0.025], Int)
    DEFAULT_DISCRETIZERS[:f_accel_250ms         ] = LinearDiscretizer([-5.00,-3.045338252335337,-2.0,-1.50593693678868,-0.24991770599882523,0.06206203400761478,0.2489269478410686,0.5,2.6], Int)
    DEFAULT_DISCRETIZERS[:f_accel_500ms         ] = LinearDiscretizer([-1.0,-0.25,-0.08,0.0,0.08,0.25,1.0], Int)
    DEFAULT_DISCRETIZERS[:f_des_angle_250ms     ] = LinearDiscretizer([-0.25,-0.05,-0.03555530775035057,-0.02059935606012066,-0.005175260331415599,0.00642120613623413,0.02143002425291505,0.05,0.25], Int)
    DEFAULT_DISCRETIZERS[:f_des_angle_500ms     ] = LinearDiscretizer([-0.025,-0.01,-0.005,0.005,0.01,0.025], Int)
    DEFAULT_DISCRETIZERS[:f_des_speed_250ms     ] = LinearDiscretizer([-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0]+29.06, Int)
    DEFAULT_DISCRETIZERS[:f_des_speed_500ms     ] = LinearDiscretizer([-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0]+29.06, Int)
    DEFAULT_DISCRETIZERS[:f_acc_control_250ms   ] = LinearDiscretizer([-0.2,0.0,0.05,0.2,0.4,0.6], Int)
    DEFAULT_DISCRETIZERS[:f_acc_control_500ms   ] = LinearDiscretizer([-0.2,0.0,0.05,0.2,0.4,0.6], Int)
    DEFAULT_DISCRETIZERS[:f_deltaY_250ms        ] = LinearDiscretizer([-0.4,-0.2,-0.1,0.1,0.2,0.4], Int)
    DEFAULT_DISCRETIZERS[:lanechange2s          ] = LinearDiscretizer([-0.5,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:yaw                   ] = LinearDiscretizer([-0.05,-0.02,-0.0065,0.0065,0.02,0.05], Int)
    DEFAULT_DISCRETIZERS[:speed                 ] = LinearDiscretizer([0,10,20,22.5,25,30], Int)
    DEFAULT_DISCRETIZERS[:delta_speed_limit     ] = LinearDiscretizer([-1.0,-0.5,-0.0,0.5,1.0], Int)
    DEFAULT_DISCRETIZERS[:posFy                 ] = LinearDiscretizer([-2.5,-1.5,-0.5,0.5,1.5,2.5], Int)
    DEFAULT_DISCRETIZERS[:velFx                 ] = LinearDiscretizer([0,10,20,22.5,25,30], Int)
    DEFAULT_DISCRETIZERS[:velFy                 ] = LinearDiscretizer([-0.6,-0.4,-0.2,-0.1,0.1,0.2,0.4,0.6], Int)
    DEFAULT_DISCRETIZERS[:turnrate              ] = LinearDiscretizer([-0.04,-0.015,-0.005,0.005,0.015,0.04], Int)
    DEFAULT_DISCRETIZERS[:turnrate_global       ] = LinearDiscretizer([-0.04,-0.015,-0.005,0.005,0.015,0.04], Int)
    DEFAULT_DISCRETIZERS[:acc                   ] = LinearDiscretizer([-1,-0.25,-0.08,0.08,0.25,1], Int)
    DEFAULT_DISCRETIZERS[:accFx                 ] = LinearDiscretizer([-1,-0.25,-0.08,0.08,0.25,1], Int)
    DEFAULT_DISCRETIZERS[:accFy                 ] = LinearDiscretizer([-0.5,-0.15,-0.05,0.05,0.15,0.5], Int)
    DEFAULT_DISCRETIZERS[:cl                    ] = CategoricalDiscretizer([1.0,2.0,3.0,4.0], Int)
    DEFAULT_DISCRETIZERS[:d_cl                  ] = LinearDiscretizer([-2.5,-1.5,-1,1,1.5,2.5], Int)
    DEFAULT_DISCRETIZERS[:d_ml                  ] = LinearDiscretizer([-4,-3,-2,-1.5,-1,0], Int)
    DEFAULT_DISCRETIZERS[:d_mr                  ] = LinearDiscretizer([0,1,1.5,2,3,4], Int)
    DEFAULT_DISCRETIZERS[:d_merge               ] = LinearDiscretizer([0.0,50,75,100], Int)
    DEFAULT_DISCRETIZERS[:d_split               ] = LinearDiscretizer([0.0,50,75,100], Int)
    DEFAULT_DISCRETIZERS[:ttcr_ml               ] = datalineardiscretizer([0.0,6,9.6,10], Int)
    DEFAULT_DISCRETIZERS[:ttcr_mr               ] = datalineardiscretizer([0.0,6,9.6,10], Int)
    DEFAULT_DISCRETIZERS[:est_ttcr              ] = datalineardiscretizer([-10.0, -9.6, -6,0,6,9.6,10], Int)
    DEFAULT_DISCRETIZERS[:scene_velFx           ] = LinearDiscretizer([0.0,10,20,22.5,25,30], Int)
    DEFAULT_DISCRETIZERS[:a_req_stayinlane      ] = LinearDiscretizer([0,0.05,0.15,2], Int)
    DEFAULT_DISCRETIZERS[:n_lane_right          ] = CategoricalDiscretizer([0.0,1.0,2.0,3.0,4.0], Int)
    DEFAULT_DISCRETIZERS[:n_lane_left           ] = CategoricalDiscretizer([0.0,1.0,2.0,3.0,4.0], Int)
    DEFAULT_DISCRETIZERS[:has_lane_left         ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:has_lane_right        ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:curvature             ] = LinearDiscretizer([-0.002,-0.0005,0.0005,0.002], Int)
    DEFAULT_DISCRETIZERS[:has_front             ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:d_y_front             ] = datalineardiscretizer([-2,-0.5,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:v_x_front             ] = datalineardiscretizer([-1.5,-0.25,-0.1,0.1,0.25,1.5], Int)
    DEFAULT_DISCRETIZERS[:d_x_front             ] = datalineardiscretizer([0.0,5,10,30,100], Int)
    DEFAULT_DISCRETIZERS[:v_y_front             ] = datalineardiscretizer([-3,-1.5,-0.5,0.5,1.5,3], Int)
    DEFAULT_DISCRETIZERS[:yaw_front             ] = datalineardiscretizer([-0.1,-0.02,0.02,0.1], Int)
    DEFAULT_DISCRETIZERS[:a_req_front           ] = datalineardiscretizer([-2,-1,-0.25,0], Int)
    DEFAULT_DISCRETIZERS[:turnrate_front        ] = datalineardiscretizer([-0.04,-0.015,-0.005,0.005,0.015,0.04], Int)
    DEFAULT_DISCRETIZERS[:ttc_x_front           ] = datalineardiscretizer([0.0,2,4,9,10], Int)
    DEFAULT_DISCRETIZERS[:timegap_x_front       ] = datalineardiscretizer([0.0,2,4,9,10], Int)
    DEFAULT_DISCRETIZERS[:has_rear              ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:d_y_rear              ] = datalineardiscretizer([-2,-0.5,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:v_x_rear              ] = datalineardiscretizer([-1.5,-0.25,-0.1,0.1,0.25,1.5], Int)
    DEFAULT_DISCRETIZERS[:d_x_rear              ] = datalineardiscretizer([0.0,5,10,30,100], Int)
    DEFAULT_DISCRETIZERS[:v_y_rear              ] = datalineardiscretizer([-3,-1.5,-0.5,0.5,1.5,3], Int)
    DEFAULT_DISCRETIZERS[:yaw_rear              ] = datalineardiscretizer([-0.1,-0.02,0.02,0.1], Int)
    DEFAULT_DISCRETIZERS[:a_req_rear            ] = datalineardiscretizer([0.0,1,2,10], Int)
    DEFAULT_DISCRETIZERS[:turnrate_rear         ] = datalineardiscretizer([-0.04,-0.015,-0.005,0.005,0.015,0.04], Int)
    DEFAULT_DISCRETIZERS[:ttc_x_rear            ] = datalineardiscretizer([0.0,2,4,8,10], Int)
    DEFAULT_DISCRETIZERS[:timegap_x_rear        ] = datalineardiscretizer([0.0,2,4,9,10], Int)
    DEFAULT_DISCRETIZERS[:d_y_left              ] = datalineardiscretizer([0.0,2,5,8], Int)
    DEFAULT_DISCRETIZERS[:v_x_left              ] = datalineardiscretizer([-1.5,-0.25,-0.1,0.1,0.25,1.5], Int)
    DEFAULT_DISCRETIZERS[:d_x_left              ] = datalineardiscretizer([-50.0,-20,-10,10,20,50], Int)
    DEFAULT_DISCRETIZERS[:v_y_left              ] = datalineardiscretizer([-3,-1.5,-0.5,0.5,1.5,3], Int)
    DEFAULT_DISCRETIZERS[:yaw_left              ] = datalineardiscretizer([-0.1,-0.02,0.02,0.1], Int)
    DEFAULT_DISCRETIZERS[:a_req_left            ] = datalineardiscretizer([-2,-1,-0.05,0.05,1,2], Int)
    DEFAULT_DISCRETIZERS[:turnrate_left         ] = datalineardiscretizer([-0.04,-0.015,-0.005,0.005,0.015,0.04], Int)
    DEFAULT_DISCRETIZERS[:ttc_x_left            ] = datalineardiscretizer([0.0,2,4,8,10], Int)
    DEFAULT_DISCRETIZERS[:timegap_x_left        ] = datalineardiscretizer([0.0,2,4,9,10], Int)
    DEFAULT_DISCRETIZERS[:d_y_right             ] = datalineardiscretizer([-8.0,-5,-2,0], Int)
    DEFAULT_DISCRETIZERS[:v_x_right             ] = datalineardiscretizer([-1.5,-0.25,-0.1,0.1,0.25,1.5], Int)
    DEFAULT_DISCRETIZERS[:d_x_right             ] = datalineardiscretizer([-50.0,-20,-10,10,20,50], Int)
    DEFAULT_DISCRETIZERS[:v_y_right             ] = datalineardiscretizer([-3,-1.5,-0.5,0.5,1.5,3], Int)
    DEFAULT_DISCRETIZERS[:yaw_right             ] = datalineardiscretizer([-0.1,-0.02,0.02,0.1], Int)
    DEFAULT_DISCRETIZERS[:a_req_right           ] = datalineardiscretizer([-2,-1,-0.05,0.05,1,2], Int)
    DEFAULT_DISCRETIZERS[:turnrate_right        ] = datalineardiscretizer([-0.04,-0.015,-0.005,0.005,0.015,0.04], Int)
    DEFAULT_DISCRETIZERS[:ttc_x_right           ] = datalineardiscretizer([0.0,2,4,8,10], Int)
    DEFAULT_DISCRETIZERS[:timegap_x_right       ] = datalineardiscretizer([0.0,2,4,9,10], Int)
    DEFAULT_DISCRETIZERS[:time_consecutive_brake] = datalineardiscretizer([0.0,0.251,0.501,1.001,2.0], Int)
    DEFAULT_DISCRETIZERS[:time_consecutive_accel] = datalineardiscretizer([0.0,0.251,0.501,1.001,2.0], Int)
    DEFAULT_DISCRETIZERS[:time_consecutive_throttle] = datalineardiscretizer([-2.0,-1.001,-0.501,-0.251,0.0,0.251,0.501,1.001,2.0], Int)
    DEFAULT_DISCRETIZERS[:timesincelanecrossing ] = datalineardiscretizer([0.0,0.501,10.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_f      ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_fr     ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_r      ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_br     ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_b      ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_bl     ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_l      ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_isoccupied_fl     ] = CategoricalDiscretizer([0.0,1.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_f            ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_fr           ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_r            ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_br           ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_b            ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_bl           ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_l            ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:osg_time_fl           ] = LinearDiscretizer([0.0,4.0,8.0,12.0], Int)
    DEFAULT_DISCRETIZERS[:pastacc250ms          ] = LinearDiscretizer([-1,-0.25,-0.08,0.08,0.25,1], Int)
    DEFAULT_DISCRETIZERS[:pastacc500ms          ] = LinearDiscretizer([-1,-0.25,-0.08,0.08,0.25,1], Int)
    DEFAULT_DISCRETIZERS[:pastacc750ms          ] = LinearDiscretizer([-1,-0.25,-0.08,0.08,0.25,1], Int)
    DEFAULT_DISCRETIZERS[:pastacc1s             ] = LinearDiscretizer([-1,-0.25,-0.08,0.08,0.25,1], Int)
    DEFAULT_DISCRETIZERS[:pastturnrate250ms     ] = LinearDiscretizer([-0.02,-0.01,-0.005,0.005,0.01,0.02], Int)
    DEFAULT_DISCRETIZERS[:pastturnrate500ms     ] = LinearDiscretizer([-0.02,-0.01,-0.005,0.005,0.01,0.02], Int)
    DEFAULT_DISCRETIZERS[:pastturnrate750ms     ] = LinearDiscretizer([-0.02,-0.01,-0.005,0.005,0.01,0.02], Int)
    DEFAULT_DISCRETIZERS[:pastturnrate1s        ] = LinearDiscretizer([-0.02,-0.01,-0.005,0.005,0.01,0.02], Int)
    DEFAULT_DISCRETIZERS[:pastvelFy250ms        ] = LinearDiscretizer([-1.0,-0.5,-0.1,0.1,0.5,1.0], Int)
    DEFAULT_DISCRETIZERS[:pastvelFy500ms        ] = LinearDiscretizer([-1.0,-0.5,-0.1,0.1,0.5,1.0], Int)
    DEFAULT_DISCRETIZERS[:pastvelFy750ms        ] = LinearDiscretizer([-1.0,-0.5,-0.1,0.1,0.5,1.0], Int)
    DEFAULT_DISCRETIZERS[:pastvelFy1s           ] = LinearDiscretizer([-1.0,-0.5,-0.1,0.1,0.5,1.0], Int)
    DEFAULT_DISCRETIZERS[:pastd_cl250ms         ] = LinearDiscretizer([-0.3,-0.2,-0.1,0.1,0.2,0.3], Int)
    DEFAULT_DISCRETIZERS[:pastd_cl500ms         ] = LinearDiscretizer([-0.5,-0.3,-0.15,0.15,0.3,0.5], Int)
    DEFAULT_DISCRETIZERS[:pastd_cl750ms         ] = LinearDiscretizer([-0.75,-0.5,-0.25,0.25,0.5,0.75], Int)
    DEFAULT_DISCRETIZERS[:pastd_cl1s            ] = LinearDiscretizer([-0.75,-0.5,-0.25,0.25,0.5,0.75], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx250ms         ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx500ms         ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx750ms         ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx1s            ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx1500ms        ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx2s            ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx2500ms        ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx3s            ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFx4s            ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy250ms         ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy500ms         ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy750ms         ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy1s            ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy1500ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy2s            ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy2500ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy3s            ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxaccFy4s            ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate250ms      ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate500ms      ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate750ms      ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate1s         ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate1500ms     ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate2s         ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate2500ms     ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate3s         ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:maxturnrate4s         ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx250ms        ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx500ms        ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx750ms        ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx1s           ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx1500ms       ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx2s           ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx2500ms       ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx3s           ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFx4s           ] = LinearDiscretizer([-1.5,-0.5,-0.15,0.15,0.5,1.5], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy100ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy150ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy200ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy250ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy500ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy750ms        ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy1s           ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy1500ms       ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy2s           ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy2500ms       ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy3s           ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanaccFy4s           ] = LinearDiscretizer([-2,-0.5,-0.15,0.15,0.5,2], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate250ms     ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate500ms     ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate750ms     ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate1s        ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate1500ms    ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate2s        ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate2500ms    ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate3s        ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:meanturnrate4s        ] = LinearDiscretizer([-0.1,-0.03,-0.004,0.004,0.03,0.1], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx250ms         ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx500ms         ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx750ms         ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx1s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx1500ms        ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx2s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx2500ms        ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx3s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFx4s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy250ms         ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy500ms         ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy750ms         ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy1s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy1500ms        ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy2s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy2500ms        ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy3s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdaccFy4s            ] = LinearDiscretizer([0,0.1,0.2,0.5], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate250ms      ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate500ms      ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate750ms      ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate1s         ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate1500ms     ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate2s         ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate2500ms     ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate3s         ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
    DEFAULT_DISCRETIZERS[:stdturnrate4s         ] = LinearDiscretizer([0,0.01,0.02,0.04], Int)
const NLOPT_SOLVER = :LN_SBPLX # :LN_COBYLA :LN_SBPLX :GN_DIRECT_L
const NLOPT_XTOL_REL = 1e-4
const BAYESIAN_SCORE_IMPROVEMENT_THRESOLD = 1e-1

function calc_bincounts(
    data_sorted_ascending::Vector{Float64},
    binedges::Vector{Float64} # this includes min and max edges
    )

    # println(data_sorted_ascending[1], " ≥ ", binedges[1])
    # println(data_sorted_ascending[end], " ≤ ", binedges[end])
    @assert(data_sorted_ascending[1] ≥ binedges[1])
    @assert(data_sorted_ascending[end] ≤ binedges[end])

    bincounts = zeros(Int, length(binedges)-1)

    i = 1
    for x in data_sorted_ascending
        while x > binedges[i+1]
            i += 1
        end
        bincounts[i] += 1
    end
    bincounts
end
function calc_bincounts_array{D<:AbstractDiscretizer}(binmaps::Vector{D})
    r = Array(Int, length(binmaps))
    for i = 1 : length(r)
        r[i] = nlabels(binmaps[i])
    end
    r
end
function discretize{D<:AbstractDiscretizer}(
    binmaps::Vector{D},
    data::Matrix{Float64}, # (nfeatures × nsamples)
    features::Vector{AbstractFeature}
    )

    N, M = size(data)
    mat = Array(Int, N, M)
    for (i,dmap) in enumerate(binmaps)
        for j = 1 : M
            value = data[i,j]

            @assert(!isna(value))
            # @assert(supports_encoding(dmap, value))
            # try
                mat[i,j] = encode(dmap, value)
            # catch
            #     println(dmap)
            #     println("feature: ", features[i], "  ", symbol(features[i]))
            #     println("value: ", value, " at index ($i,$j)")
            #     println(map(f->symbol(f), features[i-2:i+2]))
            #     println(data[i-5:i+5,j])
            #     error("Bad")
            # end
        end
    end

    mat
end
function discretize_cleaned{D<:AbstractDiscretizer, F<:AbstractFeature}(
    binmaps  :: Dict{Symbol, D},
    features :: Vector{F},
    data     :: DataFrame
    )

    # returns an M × N matrix where
    #    M = # of valid sample rows
    #    N = # of features

    # assumes that drop_invalid_discretization_rows! has already been run on data

    m = size(data, 1)
    mat = Array(Int, m, length(features))

    for (j,f) in enumerate(features)
        sym = symbol(f)
        dmap = binmaps[sym]
        for i = 1 : m
            value  = data[i, sym]
            @assert(!isna(value))
            @assert(supports_encoding(dmap, value))
            mat[i,j] = int(encode(dmap, value))::Int
        end
    end

    mat
end
function rediscretize!(data::ModelData, modelparams::ModelParams, variable_index::Int)
    binmap = modelparams.binmaps[variable_index]
    data.discrete[variable_index,:] = encode(binmap, data.continuous[variable_index,:])
    data
end
function drop_invalid_discretization_rows{D<:AbstractDiscretizer, F<:AbstractFeature}(
    binmaps  :: Dict{Symbol, D},
    features :: Vector{F},
    data     :: DataFrame
    )

    m = size(data, 1)
    is_valid = trues(m)
    for i = 1 : m
        for (j,f) in enumerate(features)
            sym = symbol(f)
            value  = data[i, sym]
            dmap = binmaps[sym]
            @assert(!isna(value))
            if !supports_encoding(dmap, value)
                is_valid[i] = false
                break
            end
        end
    end

    data[is_valid, :]
end
function drop_invalid_discretization_rows{D<:AbstractDiscretizer, F<:AbstractFeature}(
    binmaps  :: Dict{Symbol, D},
    features :: Vector{F},
    target_indeces :: Vector{Int}, # these cannot be Inf
    data     :: DataFrame
    )

    m = size(data, 1)

    is_valid = trues(m)
    for i = 1 : m
        for (j,f) in enumerate(features)
            sym = symbol(f)
            value = data[i, sym]::Float64
            dmap = binmaps[sym]
            if isnan(value) ||
               (isinf(value) && in(j, target_indeces)) ||
               !supports_encoding(dmap, value)

                is_valid[i] = false
                break
            end
        end
    end

    data[is_valid, :]
end
function convert_dataset_to_matrix{F<:AbstractFeature}(
    dataframe::DataFrame,
    features::Vector{F}
    )

    # creates a Matrix{Float64} which is nfeatures × nrows
    # rows are ordered in the same order as the features vector

    nrows = nrow(dataframe)
    nfeatures = length(features)

    mat = Array(Float64, nfeatures, nrows)
    for j = 1 : nrows
        for (i,f) in enumerate(features)
            sym = symbol(f)
            mat[i,j] = dataframe[j,sym]
        end
    end
    mat
end

function get_emstats(res::GraphLearningResult, binmapdict::Dict{Symbol, AbstractDiscretizer})
    emstats = Dict{String, Any}()
    emstats["binmaps"] = binmapdict
    emstats["targets"] = [res.target_lat, res.target_lon]
    emstats["indicators"] = res.features[3:end]
    emstats["statistics"] = res.stats
    emstats["adjacency"] = res.adj
    emstats
end
function get_emstats{D<:AbstractDiscretizer, F<:AbstractFeature}(res::GraphLearningResult, binmaps::Vector{D}, features::Vector{F})
    binmapdict = Dict{Symbol,AbstractDiscretizer}()
    for (b,f) in zip(binmaps, features)
        binmapdict[symbol(f)] = b
    end
    get_emstats(res, binmapdict)
end

function _find_index_mapping{T}(original::Vector{T}, target::Vector{T})
    # computes the vector{Int} for indeces in target for values in original
    # all values must be present in target
    # example: original = [:a, :b, :c]
    #          target   = [:d, :c, :a, :b]
    #          retval   = [3, 4, 2]

    retval = Array(Int, length(original))
    for (i,v) in enumerate(original)
        retval[i] = findfirst(target, v)
        @assert(retval[i] != 0)
    end
    retval
end
function feature_indeces_in_net(
    find_lat::Int,
    find_lon::Int,
    parents_lat::Vector{Int},
    parents_lon::Vector{Int}
    )

    net_feature_indeces = [find_lat,find_lon]
    append!(net_feature_indeces, sort!(unique([parents_lat, parents_lon])))
    net_feature_indeces
end

function get_parent_indeces{F<:AbstractFeature}(parents::ParentFeatures, features::Vector{F})
    parents_lat = find(f->in(f, parents.lat), features)
    parents_lon = find(f->in(f, parents.lon), features)
    (parents_lat, parents_lon)
end
function calc_structure_change(orig::Vector{AbstractFeature}, res::Vector{AbstractFeature})

    added = AbstractFeature[]
    removed = AbstractFeature[]
    unchanged = AbstractFeature[]

    for f_new in res
        if in(f_new, orig)
            push!(unchanged, f_new)
        else
            push!(added, f_new)
        end
    end

    for f_old in orig
        if !in(f_old, unchanged)
            push!(removed, f_old)
        end
    end

    (added, removed, unchanged)
end

function get_starting_opt_vector(
    starting_bins_lat::Vector{Float64},
    starting_bins_lon::Vector{Float64},
    extrema_lat::(Float64, Float64),
    extrema_lon::(Float64, Float64)
    )

    unit_range_lat = bins_actual_to_unit_range(starting_bins_lat[2:end-1], extrema_lat...)
    unit_range_lon = bins_actual_to_unit_range(starting_bins_lon[2:end-1], extrema_lon...)
    [unit_range_lat, unit_range_lon]
end
function bins_unit_range_to_actual(bin_inner_edges::Vector{Float64}, bin_lo::Float64, bin_hi::Float64)
    width = bin_hi - bin_lo
    retval = bin_inner_edges .* width
    retval .+= bin_lo
    retval
end
function bins_unit_range_to_actual!(bin_inner_edges::Vector{Float64}, bin_lo::Float64, bin_hi::Float64)
    width = bin_hi - bin_lo
    for i = 1 : length(bin_inner_edges)
        bin_inner_edges[i] *= width
        bin_inner_edges[i] += bin_lo
    end
    bin_inner_edges
end
function bins_actual_to_unit_range(bin_inner_edges::Vector{Float64}, bin_lo::Float64, bin_hi::Float64)
    width = bin_hi - bin_lo
    retval = bin_inner_edges .- bin_lo
    retval ./= width
    retval
end
function bins_actual_to_unit_range!(bin_inner_edges::Vector{Float64}, bin_lo::Float64, bin_hi::Float64)
    width = bin_hi - bin_lo
    for i = 1 : length(bin_inner_edges)
        bin_inner_edges[i] -= bin_lo
        bin_inner_edges[i] /= width
    end
    bin_inner_edges
end

function get_bin_logprobability(binmap::LinearDiscretizer, bin::Int)
    bin_width = binmap.binedges[bin+1] - binmap.binedges[bin]
    -log(bin_width)
end
function get_bin_logprobability(binmap::CategoricalDiscretizer, bin::Int)
    # values sampled from a categorical discretizer have no pdf
    0.0
end
function get_bin_logprobability(binmap::HybridDiscretizer, bin::Int)
    if bin ≤ binmap.disc.lin.nbins
        get_bin_logprobability(binmap.lin, bin)
    else
        get_bin_logprobability(binmap.cat, bin - binmap.lin.nbins)
    end
end

function _pre_optimize_categorical_binning(
    data_sorted_ascending::Vector{Float64},
    nbins::Int, # number of bins in resulting discretization
    extrema::(Float64, Float64), # upper and lower bounds
    ncandidate_bins::Int # number of evenly spaced candidate binedges to consider in the pretraining phase
    )

    lo, hi = extrema
    ncenter_bins = nbins-1
    candidate_binedges = linspace(lo, hi, ncandidate_bins+2)
    bincounts = calc_bincounts(data_sorted_ascending, candidate_binedges)
    candidate_binedges = candidate_binedges[2:end-1]
    ncandidate_binedges = length(candidate_binedges)

    binedge_assignments = [1:ncenter_bins]

    binedges = Array(Float64, nbins+1)
    binedges[1] = lo
    binedges[end] = hi
    binedges[2:nbins] = candidate_binedges[binedge_assignments]

    counts = Array(Int, nbins)
    for i = 1 : nbins-1
        counts[i] = bincounts[i]
    end
    counts[nbins] = 0
    for i = nbins : ncandidate_binedges+1
        counts[nbins] += bincounts[i]
    end

    best_score = calc_categorical_score(counts, binedges)
    best_binedges = copy(binedges)

    # println("counts:            ", counts)
    # println("starting binedges: ", binedges)
    # println("starting score:    ", best_score)

    n_trials = binomial(ncandidate_binedges, ncenter_bins)-1
    while n_trials > 0
        n_trials -= 1

        if binedge_assignments[ncenter_bins] < ncandidate_binedges
            # move the previous bin into the past one
            binedge_assignments[ncenter_bins] += 1

            a = binedge_assignments[ncenter_bins]
            counts[nbins] -= bincounts[a]
            counts[nbins-1] += bincounts[a]
            binedges[ncenter_bins+1] = candidate_binedges[a]
        else
            i = ncenter_bins - 1
            while i > 1 && binedge_assignments[i] == binedge_assignments[i+1] - 1
                i -= 1
            end

            binedge_assignments[i] += 1
            a = binedge_assignments[i]
            counts[i+1] -= bincounts[a]
            counts[i] += bincounts[a]
            binedges[i+1] = candidate_binedges[a]

            while i < ncenter_bins
                i += 1
                binedge_assignments[i] = binedge_assignments[i-1]+1
                counts[i] = bincounts[binedge_assignments[i]]
                binedges[i+1] = candidate_binedges[binedge_assignments[i]]
            end

            counts[nbins] = 0
            for i in binedge_assignments[i]+1 : ncandidate_binedges+1
                counts[nbins] += bincounts[i]
            end
        end

        score = calc_categorical_score(counts, binedges)
        # @assert(calc_bincounts(data_sorted_ascending, binedges) == counts)
        # @assert(calc_categorical_score(data_sorted_ascending, binedges) == score)

        if score > best_score
            # println("better score: ", score)
            # println("better edges: ", binedges)
            best_score = score
            copy!(best_binedges, binedges)
        end

        # println(binedge_assignments, "  ", score, "  ", counts)
    end

    # println(best_binedges, "  ", best_score)

    best_binedges
end
function _optimize_categorical_binning_nlopt!(binedges::Vector{Float64}, data_sorted_ascending::Vector{Float64}; ε::Float64=1e-5)

    bin_lo, bin_hi, nbins = binedges[1], binedges[end], length(binedges)-1
    starting_opt_vector = bins_actual_to_unit_range(binedges[2:nbins], bin_lo, bin_hi)

    optimization_objective(x::Vector, grad::Vector) = begin
        if length(grad) > 0
            warn("TRYING TO COMPUTE GRADIENT!") # do nothing - this is Nonlinear
        end

        width = bin_hi - bin_lo
        for i = 1 : nbins-1
            binedges[i+1] = max((x[i] * width) + bin_lo, binedges[i]+ε)
        end

        calc_categorical_score(data_sorted_ascending, binedges)
    end

    n = length(starting_opt_vector)
    opt = Opt(NLOPT_SOLVER, n)
    xtol_rel!(opt, NLOPT_XTOL_REL)
    lower_bounds!(opt, zeros(Float64, n))
    upper_bounds!(opt, ones(Float64, n))
    max_objective!(opt, optimization_objective)

    # println("before: ", optimization_objective(starting_opt_vector, Int[]))
    maxf, maxx, ret = optimize(opt, starting_opt_vector)
    # println(maxf, "  ", maxx, "  ", ret)
    binedges
end
function _optimize_categorical_binning(
    data_sorted_ascending::Vector{Float64},
    nbins::Int, # number of bins in resulting discretization
    extrema::(Float64, Float64), # upper and lower bounds
    ncandidate_bins::Int # number of evenly spaced candidate binedges to consider in the pretraining phase
    )

    the_binedges = _pre_optimize_categorical_binning(data_sorted_ascending, nbins, extrema, ncandidate_bins)
    _optimize_categorical_binning_nlopt!(the_binedges, data_sorted_ascending)
end

function calc_categorical_loglikelihood(
    data_sorted_ascending::Vector{Float64},
    binedges::Vector{Float64} # this includes min and max edges
    )

    if data_sorted_ascending[1] < binedges[1] ||
       data_sorted_ascending[end] > binedges[end]
       return -Inf
    end

    m = length(data_sorted_ascending)
    # width_tot = binedges[end] - binedges[1]
    logl = - m*log(m) # + m*log(width_tot)
    i = 2
    total = 0
    for x in data_sorted_ascending
        while x > binedges[i]
            if total > 0
                bin_width = binedges[i] - binedges[i-1]
                logl += total * log(total / bin_width)
            end
            total = 0
            i += 1
        end
        total += 1
    end
    bin_width = binedges[i] - binedges[i-1]
    logl += total * log(total / bin_width)

    logl
end
function calc_categorical_score(
    data_sorted_ascending::Vector{Float64},
    binedges::Vector{Float64} # this includes min and max edges
    )

    if data_sorted_ascending[1] < binedges[1] ||
       data_sorted_ascending[end] > binedges[end]
       return -Inf
    end

    logl = 0.0
    i = 2
    total = 0
    for x in data_sorted_ascending
        while x > binedges[i]
            if total > 0
                bin_width = binedges[i] - binedges[i-1]
                logl += total * log(total / bin_width)
            end
            total = 0
            i += 1
        end
        total += 1
    end
    bin_width = binedges[i] - binedges[i-1]
    logl += total * log(total / bin_width)

    logl
end
function calc_categorical_score(
    counts::Vector{Int},
    binedges::Vector{Float64} # this includes min and max edges
    )

    logl = 0.0

    for (i,total) in enumerate(counts)
        if total > 0
            bin_width = binedges[i+1] - binedges[i]
            logl += total * log(total / bin_width)
        end
    end

    logl
end
function calc_bayesian_score(
    data::ModelData,
    modelparams::ModelParams,
    staticparams::ModelStaticParams
    )

    # NOTE: this does not compute the score components for the indicator variables

    log_bayes_score_component(staticparams.ind_lat, modelparams.parents_lat, data.bincounts, data.discrete) +
        log_bayes_score_component(staticparams.ind_lon, modelparams.parents_lon, data.bincounts, data.discrete)
end
function calc_discretize_score(
    binmap::AbstractDiscretizer,
    stats::Matrix{Int} # NOTE(tim): should not include prior
    )

    count_orig = sum(stats)
    count_new = count_orig + 2*length(stats)
    count_ratio = count_orig / count_new
    # log_count_ratio = log(count_ratio)
    # println(log_count_ratio)

    # orig_score = 0.0
    # for i = 1 : size(stats, 1)

    #     count = 0
    #     for j = 1 : size(stats, 2)
    #         count += stats[i,j]
    #     end

    #     P = get_bin_logprobability(binmap, i)
    #     orig_score += count*P
    # end

    score = 0.0
    for i = 1 : size(stats, 1)

        total = 0
        for j = 1 : size(stats, 2)
            total += stats[i,j]
        end
        count_modified = total + size(stats, 2)

        P = get_bin_logprobability(binmap, i)
        score += count_modified*count_ratio*P
    end

    # println("orig score: ", orig_score)
    # println("new  score: ", score)

    score
end
function calc_discretize_score(
    data::ModelData,
    modelparams::ModelParams,
    staticparams::ModelStaticParams
    )

    score = 0.0

    binmap = modelparams.binmaps[staticparams.ind_lat]
    stats = SmileExtra.statistics(staticparams.ind_lat, modelparams.parents_lat,
                       data.bincounts, data.discrete)
    score += calc_discretize_score(binmap, stats)
    @assert(!isinf(score))

    binmap = modelparams.binmaps[staticparams.ind_lon]
    stats = SmileExtra.statistics(staticparams.ind_lon, modelparams.parents_lon,
                       data.bincounts, data.discrete)
    score += calc_discretize_score(binmap, stats)
    @assert(!isinf(score))

    score
end
function calc_component_score(
    target_index::Int,
    target_parents::Vector{Int},
    target_binmap::AbstractDiscretizer,
    data::ModelData,
    stats::AbstractMatrix{Int}
    )

    calc_discretize_score(target_binmap, stats) +
        log_bayes_score_component(target_index, target_parents,
                                  data.bincounts, data.discrete)
end
function calc_component_score(
    target_index::Int,
    target_parents::Vector{Int},
    target_binmap::AbstractDiscretizer,
    data::ModelData
    )

    stats = SmileExtra.statistics(target_index, target_parents,
                       data.bincounts, data.discrete)

    calc_component_score(target_index, target_parents, target_binmap, data, stats)
end
function calc_component_score(
    target_index::Int,
    target_parents::Vector{Int},
    target_binmap::AbstractDiscretizer,
    data::ModelData,
    score_cache::Dict{Vector{Int}, Float64}
    )

    stats = SmileExtra.statistics(target_index, target_parents,
                       data.bincounts, data.discrete)

    calc_discretize_score(target_binmap, stats) +
        log_bayes_score_component(target_index, target_parents,
                                  data.bincounts, data.discrete, score_cache)
end
function calc_complete_score(
    data::ModelData,
    modelparams::ModelParams,
    staticparams::ModelStaticParams
    )

    bayesian_score = calc_bayesian_score(data, modelparams, staticparams)
    discretize_score = calc_discretize_score(data, modelparams, staticparams)

    @assert(!isinf(bayesian_score))
    @assert(!isinf(discretize_score))

    bayesian_score + discretize_score
end

function optimize_categorical_binning!(
    data::Vector{Float64}, # datapoints (will be sorted)
    nbins::Int, # number of bins in resulting discretization
    extrema::(Float64, Float64), # upper and lower bounds
    ncandidate_bins::Int # number of evenly spaced candidate binedges to consider in the pretraining phase
    )

    #=
    Optimizes the discretization of a 1D continuous variable
    into a set of discrete, uniform-width bins
    =#

    @assert(!isempty(data))
    @assert(nbins > 1)
    @assert(extrema[1] < extrema[2])
    ncandidate_bins = max(nbins, ncandidate_bins)

    _optimize_categorical_binning(sort!(data), nbins, extrema, ncandidate_bins)
end
function optimize_structure!(
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelData;
    forced::(Vector{Int}, Vector{Int})=(Int[], Int[]), # lat, lon
    verbosity::Integer=0,
    max_parents::Integer=6
    )

    binmaps = modelparams.binmaps
    parents_lat = deepcopy(modelparams.parents_lat)
    parents_lon = deepcopy(modelparams.parents_lon)

    forced_lat, forced_lon = forced
    parents_lat = sort(unique([parents_lat, forced_lat]))
    parents_lon = sort(unique([parents_lon, forced_lon]))

    features = staticparams.features
    ind_lat = staticparams.ind_lat
    ind_lon = staticparams.ind_lon
    binmap_lat = modelparams.binmaps[ind_lat]
    binmap_lon = modelparams.binmaps[ind_lon]

    n_targets = 2
    n_indicators = length(features)-n_targets

    chosen_lat = map(i->in(n_targets+i, parents_lat), [1:n_indicators])
    chosen_lon = map(i->in(n_targets+i, parents_lon), [1:n_indicators])

    score_cache_lat = Dict{Vector{Int}, Float64}()
    score_cache_lon = Dict{Vector{Int}, Float64}()

    score_lat = calc_component_score(ind_lat, parents_lat, binmap_lat, data, score_cache_lat)
    score_lon = calc_component_score(ind_lon, parents_lon, binmap_lon, data, score_cache_lon)
    score = score_lat + score_lon

    if verbosity > 0
        println("Starting Score: ", score)
    end

    n_iter = 0
    score_diff = 1.0
    while score_diff > 0.0
        n_iter += 1

        selected_lat = false
        selected_index = 0
        new_parents_lat = copy(parents_lat)
        new_parents_lon = copy(parents_lon)
        score_diff = 0.0

        # check edges for indicators -> lat
        if length(parents_lat) < max_parents
            for i = 1 : n_indicators
                # add edge if it does not exist
                if !chosen_lat[i]
                    new_parents = sort!(push!(copy(parents_lat), n_targets+i))
                    new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, score_cache_lat) - score_lat
                    if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                        selected_lat = true
                        score_diff = new_score_diff
                        new_parents_lat = new_parents
                    end
                end
            end
        elseif verbosity > 0
            warn("DBNB: optimize_structure: max parents lat reached")
        end
        for (idx, i) in enumerate(parents_lat)
            # remove edge if it does exist
            if !in(features[i], forced_lat)
                new_parents = deleteat!(copy(parents_lat), idx)
                new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, score_cache_lat) - score_lat
                if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                    selected_lat = true
                    score_diff = new_score_diff
                    new_parents_lat = new_parents
                end
            end
        end

        # check edges for indicators -> lon
        if length(parents_lon) < max_parents
            for i = 1 : n_indicators
                # add edge if it does not exist
                if !chosen_lon[i]
                    new_parents = sort!(push!(copy(parents_lon), n_targets+i))
                    new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, score_cache_lon) - score_lon
                    if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                        selected_lat = false
                        score_diff = new_score_diff
                        new_parents_lon = new_parents
                    end
                end
            end
        elseif verbosity > 0
            warn("DBNB: optimize_structure: max parents lon reached")
        end
        for (idx, i) in enumerate(parents_lon)
            # remove edge if it does exist
            if !in(features[i], forced_lon)
                new_parents = deleteat!(copy(parents_lon), idx)
                new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, score_cache_lon) - score_lon
                if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                    selected_lat = false
                    score_diff = new_score_diff
                    new_parents_lon = new_parents
                end
            end
        end

        # check edge between lat <-> lon
        if !in(ind_lon, parents_lat) && !in(ind_lat, parents_lon)
            # lon -> lat
            if length(parents_lat) < max_parents
                new_parents = unshift!(copy(parents_lat), ind_lon)
                new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, score_cache_lat) - score_lat
                if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                    selected_lat = true
                    score_diff = new_score_diff
                    new_parents_lat = new_parents
                end
            end

            # lat -> lon
            if length(parents_lon) < max_parents
                new_parents = unshift!(copy(parents_lon), ind_lat)
                new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, score_cache_lon) - score_lon
                if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                    selected_lat = false
                    score_diff = new_score_diff
                    new_parents_lon = new_parents
                end
            end
        elseif in(ind_lon, parents_lat) && !in(features[ind_lon], forced_lat)

            # try edge removal
            new_parents = deleteat!(copy(parents_lat), ind_lat)
            new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, score_cache_lat) - score_lat
            if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                selected_lat = true
                score_diff = new_score_diff
                new_parents_lat = new_parents
            end

            # try edge reversal (lat -> lon)
            if length(parents_lon) < max_parents
                new_parents = unshift!(copy(parents_lon), ind_lat)
                new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, score_cache_lon) - score_lon
                if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                    selected_lat = false
                    score_diff = new_score_diff
                    new_parents_lon = new_parents
                end
            end
        elseif in(ind_lat, parents_lon)  && !in(features[ind_lat], forced_lon)

            # try edge removal
            new_parents = deleteat!(copy(parents_lon), ind_lat)
            new_score_diff = calc_component_score(ind_lon, new_parents, binmap_lon, data, score_cache_lon) - score_lon
            if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                selected_lat = false
                score_diff = new_score_diff
                new_parents_lon = new_parents
            end

            # try edge reversal (lon -> lat)
            if length(parents_lat) < max_parents
                new_parents = unshift!(copy(parents_lat), ind_lon)
                new_score_diff = calc_component_score(ind_lat, new_parents, binmap_lat, data, score_cache_lat) - score_lat
                if new_score_diff > score_diff + BAYESIAN_SCORE_IMPROVEMENT_THRESOLD
                    selected_lat = true
                    score_diff = new_score_diff
                    new_parents_lat = new_parents
                end
            end
        end

        # select best
        if score_diff > 0.0
            if selected_lat
                parents_lat = new_parents_lat
                chosen_lat = map(k->in(n_targets+k, parents_lat), [1:n_indicators])
                score += score_diff
                score_lat += score_diff
                if verbosity > 0
                    println("changed lat:", map(f->symbol(f), features[parents_lat]))
                    println("new score: ", score)
                end
            else
                parents_lon = new_parents_lon
                chosen_lon = map(k->in(n_targets+k, parents_lon), [1:n_indicators])
                score += score_diff
                score_lon += score_diff
                if verbosity > 0
                    println("changed lon:", map(f->symbol(f), features[parents_lon]))
                    println("new score: ", score)
                end
            end
        end
    end

    empty!(modelparams.parents_lat)
    empty!(modelparams.parents_lon)
    append!(modelparams.parents_lat, parents_lat)
    append!(modelparams.parents_lon, parents_lon)

    modelparams
end
function optimize_target_bins!(
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelData
    )

    binmaps = modelparams.binmaps
    parents_lat = modelparams.parents_lat
    parents_lon = modelparams.parents_lon

    ind_lat = staticparams.ind_lat
    ind_lon = staticparams.ind_lon
    binmap_lat = modelparams.binmaps[ind_lat]
    binmap_lon = modelparams.binmaps[ind_lon]

    nbins_lat = nlabels(binmap_lat)
    nbins_lon = nlabels(binmap_lon)

    extrema_lat = extrema(binmap_lat)
    extrema_lon = extrema(binmap_lon)

    truncated_gaussian_lat = Truncated(Normal(0.0, 0.02), extrema_lat[1], extrema_lat[2])
    truncated_gaussian_lon = Truncated(Normal(0.0, 0.5), extrema_lon[1], extrema_lon[2])

    starting_bins_lat = quantile(truncated_gaussian_lat, linspace(0.0,1.0,nbins_lat+1))
    starting_bins_lon = quantile(truncated_gaussian_lon, linspace(0.0,1.0,nbins_lon+1))
    starting_opt_vector = get_starting_opt_vector(starting_bins_lat, starting_bins_lon, extrema_lat, extrema_lon)
    binmap_lat.binedges[:] = starting_bins_lat[:]
    binmap_lon.binedges[:] = starting_bins_lon[:]

    function overwrite_model_params!(x::Vector; ε::Float64=eps(Float64))
        # override binedges
        binedges_lat = bins_unit_range_to_actual(x[1:nbins_lat-1], binmap_lat.binedges[1], binmap_lat.binedges[end])
        for i = 1 : nbins_lat-1
            binmap_lat.binedges[i+1] = max(binedges_lat[i], binmap_lat.binedges[i] + ε)
        end
        for i = nbins_lat : -1 : 2
            binmap_lat.binedges[i] = min(binmap_lat.binedges[i], binmap_lat.binedges[i+1]-ε)
        end

        binedges_lon = bins_unit_range_to_actual(x[nbins_lat:end], binmap_lon.binedges[1], binmap_lon.binedges[end])
        for i = 1 : nbins_lon-1
            binmap_lon.binedges[i+1] = max(binedges_lon[i], binmap_lon.binedges[i] + ε)
        end
        for i = nbins_lon : -1 : 2
            binmap_lon.binedges[i] = min(binmap_lon.binedges[i], binmap_lon.binedges[i+1]-ε)
        end

        # rediscretize
        rediscretize!(data, modelparams, ind_lat)
        rediscretize!(data, modelparams, ind_lon)

        nothing
    end
    function optimization_objective(x::Vector, grad::Vector)
        if length(grad) > 0
            # do nothing - this is Nonlinear
            warn("TRYING TO COMPUTE GRADIENT!")
        end

        overwrite_model_params!(x)

        # compute score
        score = calc_component_score(ind_lat, parents_lat, binmap_lat, data) +
                calc_component_score(ind_lon, parents_lon, binmap_lon, data)

        # println(x, " -> ", score)

        score
    end

    n = length(starting_opt_vector)
    opt = Opt(NLOPT_SOLVER, n)
    xtol_rel!(opt, NLOPT_XTOL_REL)
    lower_bounds!(opt, zeros(Float64, n))
    upper_bounds!(opt, ones(Float64, n))
    max_objective!(opt, optimization_objective)

    maxf, maxx, ret = optimize(opt, starting_opt_vector)

    overwrite_model_params!(maxx)
end
function optimize_indicator_bins!(
    binmap::AbstractDiscretizer,
    index::Int,
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelData
    )

    nothing
end
function optimize_indicator_bins!(
    binmap::LinearDiscretizer,
    index::Int,
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelData
    )

    nbins = nlabels(binmap)
    if nbins ≤ 1
        return nothing
    end

    bin_lo, bin_hi = extrema(binmap)

    binmaps = modelparams.binmaps
    ind_lat = staticparams.ind_lat
    ind_lon = staticparams.ind_lon
    parents_lat = modelparams.parents_lat
    parents_lon = modelparams.parents_lon
    binmap_lat = modelparams.binmaps[ind_lat]
    binmap_lon = modelparams.binmaps[ind_lon]
    nbins_lat = nlabels(binmap_lat)
    nbins_lon = nlabels(binmap_lon)

    starting_opt_vector = bins_actual_to_unit_range(binmap.binedges[2:nbins], bin_lo, bin_hi)

    function overwrite_model_params!(x::Vector; ε::Float64=eps(Float64))
        # override binedges
        bin_edges = bins_unit_range_to_actual(x, bin_lo, bin_hi)

        for i = 1 : nbins-1
            binmap.binedges[i+1] = max(bin_edges[i], binmap.binedges[i] + ε)
        end
        for i = nbins : -1 : 2
            binmap.binedges[i] = min(binmap.binedges[i], binmap.binedges[i+1]-ε)
        end

        # rediscretize
        data.discrete[index,:] = encode(binmap, data.continuous[index,:])

        nothing
    end
    function optimization_objective_both_parents(x::Vector, grad::Vector)
        if length(grad) > 0
            # do nothing - this is Nonlinear
            warn("TRYING TO COMPUTE GRADIENT!")
        end

        overwrite_model_params!(x)

        score = calc_component_score(ind_lat, parents_lat, binmap_lat, data) +
                calc_component_score(ind_lon, parents_lon, binmap_lon, data)

        score
    end
    function optimization_objective_lat(x::Vector, grad::Vector)
        if length(grad) > 0
            # do nothing - this is Nonlinear
            warn("TRYING TO COMPUTE GRADIENT!")
        end

        overwrite_model_params!(x)

        calc_component_score(ind_lat, parents_lat, binmap_lat, data)
    end
    function optimization_objective_lon(x::Vector, grad::Vector)
        if length(grad) > 0
            # do nothing - this is Nonlinear
            warn("TRYING TO COMPUTE GRADIENT!")
        end

        overwrite_model_params!(x)

        calc_component_score(ind_lon, parents_lon, binmap_lon, data)
    end

    n = length(starting_opt_vector)
    opt = Opt(NLOPT_SOLVER, n)
    xtol_rel!(opt, NLOPT_XTOL_REL)
    lower_bounds!(opt, zeros(Float64, n))
    upper_bounds!(opt, ones(Float64, n))

    in_lat = in(index, parents_lat)
    in_lon = in(index, parents_lon)
    if in_lat && in_lon
        max_objective!(opt, optimization_objective_both_parents)
    elseif in_lat
        max_objective!(opt, optimization_objective_lat)
    elseif in_lon
        max_objective!(opt, optimization_objective_lon)
    else
        error("target is not an indicator variable")
    end

    # enforce ordering of bins, f(x,g) ≤ 0
    # for i = 1 : nbins-2
    #     inequality_constraint!(opt, (x,g) -> x[i] - x[i+1])
    # end

    maxf, maxx, ret = optimize(opt, starting_opt_vector)

    overwrite_model_params!(maxx)
end
function optimize_indicator_bins!(
    binmap::HybridDiscretizer,
    index::Int,
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelData
    )



    lin = binmap.lin
    nbins = nlabels(lin)
    if nbins ≤ 1
        return nothing
    end

    bin_lo, bin_hi = extrema(lin)
    binmaps = modelparams.binmaps
    ind_lat = staticparams.ind_lat
    ind_lon = staticparams.ind_lon
    parents_lat = modelparams.parents_lat
    parents_lon = modelparams.parents_lon
    binmap_lat = modelparams.binmaps[ind_lat]
    binmap_lon = modelparams.binmaps[ind_lon]
    nbins_lat = nlabels(binmap_lat)
    nbins_lon = nlabels(binmap_lon)

    starting_opt_vector = bins_actual_to_unit_range(lin.binedges[2:nbins], bin_lo, bin_hi)

    function overwrite_model_params!(x::Vector; ε::Float64=eps(Float64))
        # override binedges
        bin_edges = bins_unit_range_to_actual(x, bin_lo, bin_hi)


        for i = 1 : nbins-1
            lin.binedges[i+1] = max(bin_edges[i], lin.bin_edges[i] + ε)
        end
        for i = nbins : -1 : 2
            lin.binedges[i] = min(lin.bin_edges[i], lin.bin_edges[i+1]-ε)
        end

        # rediscretize
        data.discrete[index,:] = encode(binmap, data.continuous[index,:])

        nothing
    end
    function optimization_objective_both_parents(x::Vector, grad::Vector)
        if length(grad) > 0
            # do nothing - this is Nonlinear
            warn("TRYING TO COMPUTE GRADIENT!")
        end

        overwrite_model_params!(x)

        score = calc_component_score(ind_lat, parents_lat, binmap_lat, data) +
                calc_component_score(ind_lon, parents_lon, binmap_lon, data)

        score
    end
    function optimization_objective_lat(x::Vector, grad::Vector)
        if length(grad) > 0
            # do nothing - this is Nonlinear
            warn("TRYING TO COMPUTE GRADIENT!")
        end

        overwrite_model_params!(x)

        calc_component_score(ind_lat, parents_lat, binmap_lat, data)
    end
    function optimization_objective_lon(x::Vector, grad::Vector)
        if length(grad) > 0
            # do nothing - this is Nonlinear
            warn("TRYING TO COMPUTE GRADIENT!")
        end

        overwrite_model_params!(x)

        calc_component_score(ind_lon, parents_lon, binmap_lon, data)
    end

    n = length(starting_opt_vector)
    opt = Opt(NLOPT_SOLVER, n)
    xtol_rel!(opt, NLOPT_XTOL_REL)
    lower_bounds!(opt, zeros(Float64, n))
    upper_bounds!(opt, ones(Float64, n))

    in_lat = in(index, parents_lat)
    in_lon = in(index, parents_lon)
    if in_lat && in_lon
        max_objective!(opt, optimization_objective_both_parents)
    elseif in_lat
        max_objective!(opt, optimization_objective_lat)
    elseif in_lon
        max_objective!(opt, optimization_objective_lon)
    else
        error("target is not an indicator variable")
    end

    # enforce ordering of bins, f(x,g) ≤ 0
    # for i = 1 : nbins-2
    #     inequality_constraint!(opt, (x,g) -> x[i] - x[i+1])
    # end

    maxf, maxx, ret = optimize(opt, starting_opt_vector)

    overwrite_model_params!(maxx)
end
function optimize_parent_bins!(
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelData
    )

    binmaps = modelparams.binmaps
    parents_lat = modelparams.parents_lat
    parents_lon = modelparams.parents_lon

    for parent_index in unique([parents_lat, parents_lon])
        binmap = binmaps[parent_index]
        optimize_indicator_bins!(binmap, parent_index, modelparams, staticparams, data)
    end

    nothing
end

function train(::Type{DynamicBayesianNetworkBehavior}, trainingframes::DataFrame;
    starting_structure::ParentFeatures=ParentFeatures(),
    forced::ParentFeatures=ParentFeatures(),
    targetset::ModelTargets=ModelTargets(FUTUREDESIREDANGLE_250MS, FUTUREACCELERATION_250MS),
    indicators::Vector{AbstractFeature}=copy(DEFAULT_INDICATORS),
    discretizerdict::Dict{Symbol, AbstractDiscretizer}=deepcopy(DEFAULT_DISCRETIZERS),
    ncandidate_bins::Int=20,
    verbosity::Int=0,
    preoptimize_target_bins::Bool=true,
    preoptimize_indicator_bins::Bool=true,
    optimize_structure::Bool=true,
    optimize_target_bins::Bool=true,
    optimize_parent_bins::Bool=true,
    max_parents::Int=6,
    args::Dict=Dict{Symbol,Any}()
    )

    nbins_lat = -1
    nbins_lon = -1
    for (k,v) in args
        if k == :starting_structure
            starting_structure = v
        elseif k == :forced
            forced = v
        elseif k == :targetset
            targetset = v
        elseif k == :indicators
            indicators = copy(v)
        elseif k == :discretizerdict
            discretizerdict = v
        elseif k == :verbosity
            verbosity = v
        elseif k == :preoptimize_target_bins
            preoptimize_target_bins = v
        elseif k == :preoptimize_parent_bins || k == :preoptimize_indicator_bins
            preoptimize_indicator_bins = v
        elseif k == :optimize_structure
            optimize_structure = v
        elseif k == :optimize_target_bins
            optimize_target_bins = v
        elseif k == :optimize_parent_bins
            optimize_parent_bins = v
        elseif k == :ncandidate_bins
            ncandidate_bins = v
        elseif k == :nbins_lat
            nbins_lat = v
        elseif k == :nbins_lon
            nbins_lon = v
        elseif k == :max_parents
            max_parents = v
        else
            warn("Train DynamicBayesianNetworkBehavior: ignoring $k")
        end
    end

    targets = [targetset.lat, targetset.lon]

    features = [targets, indicators]
    ind_lat, ind_lon = find_target_indeces(targetset, features)
    parents_lat, parents_lon = get_parent_indeces(starting_structure, features)
    forced_lat, forced_lon = get_parent_indeces(forced, features)

    parents_lat = sort(unique([parents_lat, forced_lat]))
    parents_lon = sort(unique([parents_lon, forced_lon]))

    modelparams = ModelParams(map(f->discretizerdict[symbol(f)], features), parents_lat, parents_lon)
    staticparams = ModelStaticParams(ind_lat, ind_lon, features)

    # println("size: ", size(trainingframes))
    continuous_dataframe = deepcopy(trainingframes)
    continuous_dataframe = drop_invalid_discretization_rows(discretizerdict, features, [ind_lat, ind_lon], continuous_dataframe)
    continuous_data = convert_dataset_to_matrix(continuous_dataframe, features)

    datavec = Array(Float64, size(continuous_data, 2))
    if preoptimize_target_bins
        if verbosity > 0
            println("Optimizing Target Bins"); tic()
        end

        ###################
        # lat

        datavec[:] = continuous_data[ind_lat,:]
        disc::LinearDiscretizer = modelparams.binmaps[ind_lat]
        extremes = (disc.binedges[1], disc.binedges[end])
        nbins = nlabels(disc)
        if nbins_lat != -1
            nbins = nbins_lat
            disc = LinearDiscretizer(linspace(extremes[1], extremes[2], nbins+1))
            modelparams.binmaps[ind_lat] = disc
        end

        for (i,x) in enumerate(datavec)
            @assert(extremes[1] ≤ x ≤ extremes[2])
        end
        disc.binedges[:] = optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins)

        ###################
        # lon

        datavec[:] = continuous_data[ind_lon,:]
        disc = modelparams.binmaps[ind_lon]::LinearDiscretizer
        extremes = (disc.binedges[1], disc.binedges[end])
        nbins = nlabels(disc)
        if nbins_lon != -1
            nbins = nbins_lon
            disc = LinearDiscretizer(linspace(extremes[1], extremes[2], nbins+1))
            modelparams.binmaps[ind_lon] = disc
        end

        for (i,x) in enumerate(datavec)
            @assert(extremes[1] ≤ x ≤ extremes[2])
        end
        disc.binedges[:] = optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins)

        if verbosity > 0
            toc()
        end
    end

    # is_indicator_valid = trues(length(indicators))
    if preoptimize_indicator_bins

        if verbosity > 0
            println("Optimizing Indicator Bins"); tic()
        end

        for i in 3:length(features)
            # println(symbol(staticparams.features[i]), "  ", typeof(modelparams.binmaps[i]))
            disc2 = modelparams.binmaps[i]
            if isa(disc2, LinearDiscretizer)
                datavec[:] = continuous_data[i,:]
                nbins = nlabels(disc2)
                extremes = (disc2.binedges[1], disc2.binedges[end])
                for (k,x) in enumerate(datavec)
                    datavec[k] = clamp(x, extremes[1], extremes[2])
                end
                disc2.binedges[:] = optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins)
            elseif isa(disc2, HybridDiscretizer)
                datavec[:] = continuous_data[i,:]
                sort!(datavec)
                k = findfirst(value->isinf(value) || isnan(value), datavec)
                if k != 1 # skip if the entire array is Inf (such as if we only have freeflow data, d_x_front will be all Inf)
                    if k == 0
                        k = length(datavec)
                    else
                        k -= 1
                    end
                    nbins = nlabels(disc2.lin)
                    extremes = (disc2.lin.binedges[1], disc2.lin.binedges[end])
                    for j in 1 : k
                        datavec[j] = clamp(datavec[j], extremes[1], extremes[2])
                    end
                    disc2.lin.binedges[:] = optimize_categorical_binning!(datavec[1:k], nbins, extremes, ncandidate_bins)
                # else
                #     is_indicator_valid[i] = false
                end
            end
        end

        if verbosity > 0
            toc()
        end
    end

    # make sure to do discretization AFTER the preoptimization
    data = ModelData(continuous_data, modelparams, features)

    starttime = time()
    iter = 0
    score = calc_complete_score(data, modelparams, staticparams)

    if optimize_structure || optimize_target_bins || optimize_parent_bins
        score_diff = Inf
        SCORE_DIFF_THRESHOLD = 10.0
        while score_diff > SCORE_DIFF_THRESHOLD

            iter += 1
            verbosity == 0 || @printf("\nITER %d: %.4f (Δ%.4f) t=%.0f\n", iter, score, score_diff, time()-starttime)

            if optimize_structure
                optimize_structure!(modelparams, staticparams, data, forced=(forced_lat, forced_lon), max_parents=max_parents, verbosity=verbosity)
            end
            if optimize_target_bins
                optimize_target_bins!(modelparams, staticparams, data)
            end
            if optimize_parent_bins
                optimize_parent_bins!(modelparams, staticparams, data)
            end

            score_new = calc_complete_score(data, modelparams, staticparams)
            score_diff = score_new - score
            score = score_new
        end
    end

    if verbosity > 0
        println("\nELAPSED TIME: ", time()-starttime, "s")
        println("FINAL: ", score)

        println("\nStructure:")
        println("lat: ", map(f->symbol(f), features[modelparams.parents_lat]))
        println("lon: ", map(f->symbol(f), features[modelparams.parents_lon]))

        println("\nTarget Bins:")
        println("lat: ", modelparams.binmaps[ind_lat].binedges)
        println("lon: ", modelparams.binmaps[ind_lon].binedges)

        println("\nIndicator Bins:")
        if !isempty(modelparams.parents_lat) && !isempty(modelparams.parents_lon)
            for parent_index in unique([modelparams.parents_lat, modelparams.parents_lon])
                sym = symbol(features[parent_index])
                if isa(modelparams.binmaps[parent_index], LinearDiscretizer)
                    println(sym, "\t", modelparams.binmaps[parent_index].binedges)
                elseif isa(modelparams.binmaps[parent_index], HybridDiscretizer)
                    println(sym, "\t", modelparams.binmaps[parent_index].lin.binedges)
                end
            end
        else
            println("[empty]")
        end
    end

    binmapdict = Dict{Symbol, AbstractDiscretizer}()
    for (b,f) in zip(modelparams.binmaps, staticparams.features)
        binmapdict[symbol(f)] = b
    end

    res = GraphLearningResult("trained", staticparams.features, staticparams.ind_lat, staticparams.ind_lon,
                              modelparams.parents_lat, modelparams.parents_lon, NaN,
                              data.bincounts, data.discrete)
    model = dbnmodel(get_emstats(res, binmapdict))

    DynamicBayesianNetworkBehavior(model, DBNSimParams(), DBNSimParams())
end

end # module