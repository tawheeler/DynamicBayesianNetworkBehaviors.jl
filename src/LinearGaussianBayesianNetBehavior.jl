using OnlineStats

immutable LinearGaussianStats
    # μ = wᵀx where x[end] = 1
    w::Vector{Float64} # [nparents + 1]
    σ::Float64
end
type LinearGaussianNode
    #=
    Defines a variable in a Linear Gaussian BN

    There is one linear gaussian for each combination of discrete parents
    NaNable parents have two states: NaN or continuous, so have 2 effective "bins"
    =#

    index::Int # index of this node in the assignment list
    stats::Vector{LinearGaussianStats} # one for each discrete parent config, same order as q in DMU
    parents_disc::Vector{Int} # index, in assignment, of discrete parents
    parents_cont::Vector{Int} # index, in assignment, of continuous parents, may include other nodes's vars
    # parents_nan ::Vector{Int} # indeces of NaNable parents

    parental_assignments_disc::Vector{Int} # discrete assignment
    parental_assignments_cont::Vector{Float64} # continuous assignment (last entry is 1.0)
    parent_instantiation_counts_disc::Vector{Int} # number of possible instantiations for each discrete parent
end

"""
The ordering of the parental instantiations in discrete networks follows the convention
defined in Decision Making Under Uncertainty.

Suppose a variable has three discrete parents. The first parental instantiation
assigns all parents to their first bin. The second will assign the first
parent (as defined in `parents`) to its second bin and the other parents
to their first bin. The sequence continues until all parents are instantiated
to their last bins.

This is a directly copy from Base.sub2ind but allows for passing a vector instead of separate items

Note that this does NOT check bounds
"""
function sub2ind_vec{T<:Integer}(dims::Tuple{Vararg{Integer}}, I::AbstractVector{T})
    N = length(dims)
    @assert(N == length(I))

    ex = I[N] - 1
    for i in N-1:-1:1
        if i > N
            ex = (I[i] - 1 + ex)
        else
            ex = (I[i] - 1 + dims[i]*ex)
        end
    end

    ex + 1
end

function _get_normal(
    node::LinearGaussianNode,
    assignment_disc::Vector{Int},    # note: full array into which we must index with node.parents_disc
    assignment_cont::Vector{Float64} # note: full array into which we must index with node.parents_cont
    )

    j = 1
    if !isempty(node.parents_disc)
        for (i,v) in enumerate(node.parents_disc)
            node.parental_assignments_disc[i] = assignment_disc[v]
        end
        j = sub2ind_vec(node.parent_instantiation_counts_disc, node.parental_assignments_disc)
    end

    LG_stats = node.stats[j]

    for (i,v) in enumerate(node.parents_cont)
        node.parental_assignments_cont[i] = assignment_cont[v]
    end

    μ = dot(LG_stats.A, node.parental_assignments_cont)
    normal = Normal(μ, LG_stats.σ)
end

#
#
#
#
#

type LinearGaussianBayesianNetBehavior <: AbstractVehicleBehavior

    targets::ModelTargets
    extractor_disc :: FeaturesNew.FeatureSubsetExtractor
    extractor_cont :: FeaturesNew.FeatureSubsetExtractor
    # extractor_nan  :: FeaturesNew.FeatureSubsetExtractor
    clamper_cont :: FeaturesNew.DataClamper
    clamper_act :: FeaturesNew.DataClamper

    sample_lat_first::Bool # used to ensure topological ordering is preserved
    node_lat::LinearGaussianNode
    node_lon::LinearGaussianNode
    assignment_disc::Vector{Int} # [d]
    assignment_cont::Vector{Float64} # [c + 2], preallocated memory, with cont | lat | lon

    function LinearGaussianBayesianNetBehavior(
        targets::ModelTargets{FeaturesNew.AbstractFeature},
        extractor_disc::FeaturesNew.FeatureSubsetExtractor,
        extractor_cont::FeaturesNew.FeatureSubsetExtractor,
        # extractor_nan ::FeaturesNew.FeatureSubsetExtractor,
        clamper_cont::FeaturesNew.DataClamper,
        clamper_act::FeaturesNew.DataClamper,
        sample_lat_first::Bool,
        node_lat::LinearGaussianNode,
        node_lon::LinearGaussianNode,
        )

        retval = new()

        retval.targets = deepcopy(targets)
        retval.extractor_disc = extractor_disc
        retval.extractor_cont = extractor_cont
        # retval.extractor_nan = extractor_nan
        retval.clamper_cont = clamper_cont
        retval.clamper_act = clamper_act
        retval.sample_lat_first = sample_lat_first
        retval.node_lat = node_lat
        retval.node_lon = node_lon
        retval.assignment_disc = Array(Int, length(retval.extractor_disc.indicators))
        retval.assignment_cont = Array(Float64, length(retval.extractor_cont.indicators))

        retval
    end
end
function Base.print(io::IO, LB::LinearGaussianBayesianNetBehavior)

    var_symbols = [map(f->symbol(f), LB.extractor_disc.indicators);
                   map(f->symbol(f), LB.extractor_cont.indicators);
                   symbol(LB.targets.lat); symbol(LB.targets.lon)
                  ]

    println(io, "Linear Gaussian Bayesian Network Behavior")
    println(io, "\ttargets: ", LB.targets)
    println(io, "\tfeatures: ")
    println(io, "\t\tdiscrete:   ", map(f->symbol(f), LB.extractor_disc.indicators))
    println(io, "\t\tcontinuous: ", map(f->symbol(f), LB.extractor_cont.indicators))
    # println(io, "\t\tNaNable: ", map(f->symbol(f), LB.extractor_nan.indicators))
    println(io, "\tnumber of gaussians for lat: ", length(LB.node_lat.stats))
    println(io, "\tnumber of gaussians for lon: ", length(LB.node_lat.stats))
    println(io, "\tparents lat: ", var_symbols[LB.node_lat.parents_disc],
                                   var_symbols[LB.node_lat.parents_cont])
    println(io, "\tparents lon: ", var_symbols[LB.node_lon.parents_disc],
                                   var_symbols[LB.node_lat.parents_cont])
    println(io, "\tsample lat first: ", LB.sample_lat_first)
end

type LB_TrainParams <: AbstractVehicleBehaviorTrainParams

    targets::ModelTargets
    indicators::Vector{FeaturesNew.AbstractFeature} # list of all potential indicators

    ridge_regression_constant::Float64
    min_σ_lat::Float64 # minimum standard deviation for lateral target
    min_σ_lon::Float64 # minimum standard deviation for longitudinal target

    max_parents::Int # maximum number of parents per node
    verbosity::Int

    function LB_TrainParams(;
        targets::ModelTargets = ModelTargets{FeaturesNew.AbstractFeature}(FeaturesNew.FUTUREDESIREDANGLE,
                                                                          FeaturesNew.FUTUREACCELERATION),
        indicators::Union{Vector{AbstractFeature}, Vector{FeaturesNew.AbstractFeature}} = [
                            POSFY, YAW, SPEED, DELTA_SPEED_LIMIT, VELFX, VELFY, SCENEVELFX, TURNRATE,
                            D_CL, D_ML, D_MR, TIMETOCROSSING_LEFT, TIMETOCROSSING_RIGHT,
                            N_LANE_L, N_LANE_R, HAS_LANE_L, HAS_LANE_R, ACC, ACCFX, ACCFY,
                            A_REQ_STAYINLANE,
                            HAS_FRONT, D_X_FRONT, D_Y_FRONT, V_X_FRONT, V_Y_FRONT, TTC_X_FRONT,
                            A_REQ_FRONT, TIMEGAP_X_FRONT,
                         ],
        ridge_regression_constant::Float64=0.001,
        min_σ_lat::Float64=1e-4,
        min_σ_lon::Float64=1e-5,
        max_parents::Int=4,
        verbosity::Int=0,
        )

        @assert(!in(targets.lat, indicators))
        @assert(!in(targets.lon, indicators))

        retval = new()

        retval.targets = targets
        retval.indicators = indicators
        retval.min_σ_lat = min_σ_lat
        retval.min_σ_lon = min_σ_lon

        retval.max_parents = max_parents
        retval.verbosity = verbosity

        retval
    end
end
function Base.print(io::IO, p::LB_TrainParams)
    println(io, "LB Train Params")
    println(io, "\ttargets: ", targets)
    println(io, "\tindicators: ", map(f->symbol(f), θ.indicators))
    println(io, "\tridge_regression_constant: ", ridge_regression_constant)
    println(io, "\tmin_σ_lat:      ", min_σ_lat)
    println(io, "\tmin_σ_lon:      ", min_σ_lon)
    println(io, "\tmax_parents:    ", max_parents)
    println(io, "\tn_PCA_features: ", n_PCA_features)
end

type LB_PreallocatedData <: AbstractVehicleBehaviorPreallocatedData

    Y::Matrix{Float64} # [2 × m] # two targets (lat, lon)
    X_disc::Matrix{Int} # [d × m]
    X_cont::Matrix{Float64} # [c × m]
    # X_NaN::Matrix{Float64} # [n × m]

    features_disc  :: Vector{FeaturesNew.AbstractFeature}
    features_cont  :: Vector{FeaturesNew.AbstractFeature}
    clamper_cont   :: FeaturesNew.DataClamper
    clamper_act    :: FeaturesNew.DataClamper

    function GMR_PreallocatedData(dset::ModelTrainingData2, params::LB_TrainParams)

        retval = new()

        targets = params.targets
        indicators = params.indicators
        trainingframes = dset.dataframe
        nframes = nrow(trainingframes)
        nindicators = length(indicators)

        X = Array(Float64, nindicators, nframes)
        Y = Array(Float64, 2, nframes)
        pull_design_and_target_matrices!(X, Y, trainingframes, targets, indicators)

        ###########################

        features_disc_index = find(f->isint(f), indicators)
        features_cont_index = find(f->!isint(f), indicators)

        retval.Y = Y
        retval.X_disc = X[features_disc_index, :]
        retval.X_cont = X[features_cont_index, :]
        retval.features_disc = indicators[features_disc_index]
        retval.features_cont = indicators[features_cont_index]

        ###########################

        retval.clamper_cont = DataClamper(retval.X_cont)
        retval.clamper_act = DataClamper(retval.Y)

        retval
    end
end
function preallocate_learning_data(
    dset::ModelTrainingData2,
    params::LB_TrainParams)

    LB_PreallocatedData(dset, params)
end

function _cast_discrete_to_int!(behavior::LinearGaussianBayesianNetBehavior)
    for (i,v) in enumerate(behavior.extractor_disc.x)
        behavior.assignment_disc = round(Int, v)
    end
    behavior
end
function _copy_to_assignment!(behavior::LinearGaussianBayesianNetBehavior)
    copy!(behavior.assignment_cont, 1, behavior.clamper_cont.x, 1, length(behavior.clamper_cont.x))

    # temporarily set NaN values to 0.0
    for i in 1:length(behavior.assignment_cont)
        if isnan(x[i])
            x[i] = 0.0
        end
    end

    behavior
end
function _sample_from_node!(behavior::LinearGaussianBayesianNetBehavior, node::LinearGaussianNode)
    normal = _get_normal(node,
                         behavior.assignment_cont,
                         behavior.assignment_disc)
    action_lat = rand(normal)
    behavior.assignment_cont[node.index] = action_lat

    action_lat
end
function _process_obs(behavior::LinearGaussianBayesianNetBehavior)
    FeaturesNew.process!(behavior.clamper_cont) # TODO - ensure that clamper_cont is tied to extractor_cont
    _cast_discrete_to_int!(behavior)
    _copy_to_assignment!(behavior)
    behavior
end
function _observe_on_runlog(
    behavior::LinearGaussianBayesianNetBehavior,
    runlog::RunLog,
    sn::StreetNetwork,
    colset::UInt,
    frame::Int
    )

    FeaturesNew.observe!(behavior.extractor_disc, runlog, sn, colset, frame)
    FeaturesNew.observe!(behavior.extractor_cont, runlog, sn, colset, frame)
    _process_obs(behavior)
end
function _observe_on_dataframe(
    behavior::LinearGaussianBayesianNetBehavior,
    features::DataFrame,
    frameind::Integer,
    )

    FeaturesNew.observe!(behavior.extractor_disc, features, frameind)
    FeaturesNew.observe!(behavior.extractor_cont, features, frameind)
    _process_obs(behavior)
end
function _set_and_process_action!(behavior::LinearGaussianBayesianNetBehavior, action_lat::Float64, action_lon::Float64)
    behavior.clamper_act.x[1] = action_lat
    behavior.clamper_act.x[2] = action_lon
    FeaturesNew.process!(behavior.clamper_act)
    behavior.assignment_cont[behavior.node_lat.index] = action_lat
    behavior.assignment_cont[behavior.node_lon.index] = action_lon
    behavior
end

function select_action(
    behavior::LinearGaussianBayesianNetBehavior,
    runlog::RunLog,
    sn::StreetNetwork,
    colset::UInt,
    frame::Int
    )

    _observe_on_runlog(behavior, runlog, sn, colset, frame)

    if behavior.sample_lat_first
        behavior.clamper_act.x[1] = _sample_from_node!(behavior, behavior.node_lat)
        behavior.clamper_act.x[2] = _sample_from_node!(behavior, behavior.node_lon)
    else
        behavior.clamper_act.x[2] = _sample_from_node!(behavior, behavior.node_lon)
        behavior.clamper_act.x[1] = _sample_from_node!(behavior, behavior.node_lat)
    end

    process!(behavior.clamper_act) # clamp
    action_lat = behavior.clamper_act.x[1]
    action_lon = behavior.clamper_act.x[2]

    (action_lat, action_lon)
end

function _calc_action_loglikelihood(behavior::LinearGaussianBayesianNetBehavior)

    # NOTE: observation and setting action must have already occured

    logl = 0.0
    normal = _get_normal(behavior.node_lat, behavior.assignment_cont, behavior.assignment_disc)
    logl += logpdf(normal, behavior.clamper_act[1])

    normal = _get_normal(behavior.node_lon, behavior.assignment_cont, behavior.assignment_disc)
    logl += logpdf(normal, behavior.clamper_act[2])

    logl
end
function calc_action_loglikelihood(
    behavior::LinearGaussianBayesianNetBehavior,
    runlog::RunLog,
    sn::StreetNetwork,
    colset::UInt,
    frame::Int,
    action_lat::Float64,
    action_lon::Float64,
    )


    _observe_on_runlog(behavior, runlog, sn, colset, frame)
    _set_and_process_action!(behavior, action_lat, action_lon)
    _calc_action_loglikelihood(behavior)
end
function calc_action_loglikelihood(
    behavior::LinearGaussianBayesianNetBehavior,
    features::DataFrame,
    frameind::Integer,
    )

    action_lat = features[frameind, symbol(behavior.targets.lat)]::Float64
    action_lon = features[frameind, symbol(behavior.targets.lon)]::Float64

    _observe_on_runlog(behavior, features, frameind)
    _set_and_process_action!(behavior, action_lat, action_lon)
    _calc_action_loglikelihood(behavior)
end

#
#
#
#
#

type NodeInTraining
    index::Int # index of this node in Y
    parents_disc::Vector{Int} # index, in X_disc, of discrete parents
    parents_cont::Vector{Int} # index, in X_cont, of continuous parents, may include other nodes's vars
    target_as_parent::Int # if != 0, index of other target in Y

    NodeInTraining(index::Int) = new(index, Int[], Int[], 0)
end
function Base.hash(node::NodeInTraining, h::UInt=one(UInt))
    hash(node.index, hash(node.parents_disc, hash(node.parents_cont, hash(node.target_as_parent, h))))
end
function Base.(:(==))(A::NodeInTraining, B::NodeInTraining)
    A.index == B.index &&
    A.target_as_parent == B.target_as_parent &&
    A.parents_disc == B.parents_disc &&
    A.parents_cont == B.parents_cont
end

function _get_component_score(
    node::NodeInTraining,
    Y::Matrix{Float64},
    X_disc::Matrix{Float64},
    X_cont::Matrix{Float64},
    disc_parent_instantiations::Vector{Int},
    λ::Float64,
    )

    # Using BIC score
    # see: “Ideal Parent” Structure Learning for Continuous Variable Bayesian Networks
    #
    # BIC = max_θ  l(D|G, θ) - log(m)/2 * dim(G)
    #
    #   where max_θ  l(D|G, θ) is found using ridge regression
    #         and dim(G) is the number of parameters in G
    #
    #  Calling this function causes it to return:
    #    max_θ  l(D|G, θ) - log(m)/2*dim(node)
    #
    #   which can be added to other components to get the full BIC score

    m = nrow(Y)
    @assert(nrow(X_disc) == m)
    @assert(nrow(X_cont) == m)

    nparents_disc = length(node.parents_disc)
    nparents_cont = length(node.parents_cont)

    logl = NaN
    dim_node = -1

    if nparents_cont > 0

        if nparents_disc > 0

            # perform ridge-regression and stdev calc online
            #   Xᵀy = (λI + XᵀX)w

            parent_instantiation_counts_disc = Array(Int, nparents_disc)
            n_disc_parent_instantiations = 1
            for (i,p) in enumerate(node.parents_disc)
                parent_instantiation_counts_disc[i] = disc_parent_instantiations[p]
                n_disc_parent_instantiations *= disc_parent_instantiations[p]
            end

            # LHS is the Xᵀy vector [nparents+1]
            # RHS is the λI + XᵀX matrix [nparents+1 × nparents+1]

            LHS_arr = Array(Vector{Float64}, n_disc_parent_instantiations)
            RHS_arr = Array(Matrix{Float64}, n_disc_parent_instantiations)
            σ_var_arr = Array(Variance, n_disc_parent_instantiations)
            for i in 1 : n_disc_parent_instantiations
                LHS_arr[i] = zeros(Float64, nparents_cont+1)
                RHS_arr[i] = diagm(fill(λ, nparents_cont+1))
                σ_var_arr[i] = Variance()
            end

            parental_assignments_disc = Array(Int, nparents_disc)
            x = Array(Float64, nparents_cont+1)
            x[end] = 1.0

            for i in 1 : m
                # pull contimuous data
                y = Y[node.index, i]
                for (j,p) in enumerate(parents_cont)
                    x[j] = X_cont[p,i]
                end

                # identify which one to update
                for (j,p) in enumerate(parents_disc)
                    parental_assignments_disc[j] = X_disc[p,i]
                end
                k = sub2ind_vec(parent_instantiation_counts_disc, parental_assignments_disc)

                lhs = LHS_arr[k]
                rhs = RHS_arr[k]
                fit!(σ_var_arr[k], y)

                # update ridge regression
                for j in 1 : nparents_cont+1
                    lhs[j] += x[j]*y

                    for q in 1 : nparents_cont
                        rhs[j,q] += x[j]*x[q]
                    end
                end
            end

            # solve ridge regressions
            #  w = (λI + XᵀX)⁻¹ Xᵀy
            w_values = Array(Vector{Float64}, n_disc_parent_instantiations)
            σ_values = Array(Float64, n_disc_parent_instantiations)
            for i in 1 : n_disc_parent_instantiations
                σ_var = σ_var_arr[i]
                if nobs(σ_var) > 1
                    w_values[i] = RHS_arr[k] \ lhs_arr[k]
                    σ_values[i] = std(σ_var)
                else
                    w_values[i] = zeros(nparents_cont+1)
                    σ_values[i] = 0.001 # default standard deviation
                end
            end

            # compute the log likelihood
            logl = 0.0
            for i in 1 : m

                # pull continuous data
                y = Y[node.index, i]
                for (j,p) in enumerate(parents_cont)
                    x[j] = X_cont[p,i]
                end

                # identify which one to update
                for (j,p) in enumerate(parents_disc)
                    parental_assignments_disc[j] = X_disc[p,i]
                end
                k = sub2ind_vec(parent_instantiation_counts_disc, parental_assignments_disc)

                # calc normal and logl

                μ = dot(w_values[k], x)
                σ = σ_values[k]
                logl += logpdf(Normal(μ, σ), y)
            end

            dim_node = n_disc_parent_instantiations * (nparents_cont+2) # each parent instantiation has nparents+1 for the mean and 1 for the stdev
        else
            # no discrete parents

            # solve a single linear regression problem:
            #   Xᵀy = (λI + XᵀX)w

            # LHS is the Xᵀy vector [nparents+1]
            # RHS is the λI + XᵀX matrix [nparents+1 × nparents+1]

            lhs = zeros(Float64, nparents_cont+1)
            rhs = diagm(fill(λ, nparents_cont+1))
            σ_var = Variance()

            x = Array(Float64, nparents_cont+1)
            x[end] = 1.0

            for i in 1 : m

                # pull continuous data
                y = Y[node.index, i]
                for (j,p) in enumerate(parents_cont)
                    x[j] = X_cont[p,i]
                end

                # update stdev
                fit!(σ_var, y)

                # update ridge regression
                for j in 1 : nparents_cont+1
                    lhs[j] += x[j]*y

                    for q in 1 : nparents_cont
                        rhs[j,q] += x[j]*x[q]
                    end
                end
            end

            # solve ridge regressions
            #  w = (λI + XᵀX)⁻¹ Xᵀy

            @assert(nobs(σ_var) > 1)
            w = RHS_arr[k] \ lhs_arr[k]
            σ = std(σ_var)

            # compute the log likelihood
            logl = 0.0
            for i in 1 : m


                # pull continuous data
                y = Y[node.index, i]
                for (j,p) in enumerate(parents_cont)
                    x[j] = X_cont[p,i]
                end

                # calc normal and logl
                μ = dot(w, x)
                logl += logpdf(Normal(μ, σ), y)
            end

            dim_node = nparents_cont+2 # nparents+1 for the mean and 1 for the stdev
        end
    else # no continuous parents (μ is fixed)

        if nparents_disc > 0

            parent_instantiation_counts_disc = Array(Int, nparents_disc)
            n_disc_parent_instantiations = 1
            for (i,p) in enumerate(node.parents_disc)
                parent_instantiation_counts_disc[i] = disc_parent_instantiations[p]
                n_disc_parent_instantiations *= disc_parent_instantiations[p]
            end

            # NOTE: var can give us mean too
            var_arr = Array(Variance, n_disc_parent_instantiations)
            for i in 1 : n_disc_parent_instantiations
                var_arr[i] = Variance()
            end

            parental_assignments_disc = Array(Int, nparents_disc)

            for i in 1 : m

                y = Y[node.index, i]

                # identify which one to update
                for (j,p) in enumerate(parents_disc)
                    parental_assignments_disc[j] = X_disc[p,i]
                end
                k = sub2ind_vec(parent_instantiation_counts_disc, parental_assignments_disc)

                # update online stat
                fit!(var_arr[k], y)
            end

            μ_values = Array(Float64, n_disc_parent_instantiations)
            σ_values = Array(Float64, n_disc_parent_instantiations)
            for i in 1 : n_disc_parent_instantiations
                var = var_arr[i]
                if nobs(σ_var) > 1
                    μ_values[i] = mean(var)
                    σ_values[i] = std(var)
                elseif nobs(σ_var) > 0
                    μ_values[i] = mean(var)
                    σ_values[i] = 0.001 # default standard deviation
                else
                    μ_values[i] = 0.0
                    σ_values[i] = 0.001 # default standard deviation
                end
            end

            # compute the log likelihood
            logl = 0.0
            for i in 1 : m

                y = Y[node.index, i]

                # identify which one to update
                for (j,p) in enumerate(parents_disc)
                    parental_assignments_disc[j] = X_disc[p,i]
                end
                k = sub2ind_vec(parent_instantiation_counts_disc, parental_assignments_disc)

                # calc normal and logl
                μ = μ_values[k]
                σ = σ_values[k]
                logl += logpdf(Normal(μ, σ), y)
            end

            dim_node = n_disc_parent_instantiations * 2 # each parent instantiation has 1 for mean and 1 for stdev
        else
            # no discrete parents either

            var = Variance()
            fit!(var, Y[node.index, :])

            μ = mean(var)
            σ = std(var)
            normal = Normal(μ, σ)

            # compute the log likelihood
            logl = 0.0
            for i in 1 : m
                logl += logpdf(normal, Y[node.index, i])
            end

            dim_node = 2 # 1 for mean and 1 for stdev
        end
    end

    logl - log(m)*dim_node/2
end
function _get_component_score!(
    score_cache::Dict{NodeInTraining, Float64},
    node::NodeInTraining,
    Y::Matrix{Float64},
    X_disc::Matrix{Float64},
    X_cont::Matrix{Float64},
    disc_parent_instantiations::Vector{Int},
    λ::Float64,
    )

    if !haskey(score_cache, node)
        score_cache[node] = _get_component_score(node, Y, X_disc, X_cont,
                                                 disc_parent_instantiations, λ)
    end
    score_cache[node]
end

function _greedy_hillclimb_iter_on_node(
    node::NodeInTraining,
    score_cache::Dict{NodeInTraining, Float64},
    Y::Matrix{Float64},
    X_disc::Matrix{Float64},
    X_cont::Matrix{Float64},
    max_parents::Int
    )

    n_disc =size(X_disc, 1)
    n_cont =size(X_cont, 1)

    start_score = score_cache[node]
    best_score = start_score
    parent_to_add = -1
    add_discrete = true

    parents_disc_orig = deepcopy(node.parents_disc)
    parents_cont_orig = deepcopy(node.parents_cont)

    nparents_currently = length(parents_disc_orig) + length(parents_cont_orig)
    if nparents_currently ≥ max_parents
        (0.0, parent_to_add, add_discrete)
    end

    # try adding discrete edges
    for p in 1 : n_disc
        node.parents_disc = sort!(unique(push!(deepcopy(parents_disc_orig), p)))
        new_score = _get_component_score!(score_cache, node, Y, X_disc, X_cont)

        if new_score > best_score
            best_score = new_score
            parent_to_add = p
            add_discrete = true
        end
    end

    # try adding continuous edges
    for p in 1 : n_cont
        node.parents_cont = sort!(unique(push!(deepcopy(parents_cont_orig), p)))
        new_score = _get_component_score!(score_cache, node, Y, X_disc, X_cont)

        if new_score > best_score
            best_score = new_score
            parent_to_add = p
            add_discrete = true
        end
    end

    node.parents_disc = parents_disc_orig
    node.parents_cont = parents_cont_orig

    Δscore = best_score - start_score

    (Δscore, parent_to_add, add_discrete)
end
function _apply_node_change!(
    node::NodeInTraining,
    parent_to_add::Int,
    add_discrete::Bool
    )

    if add_discrete
        @assert(!in(node.parents_disc, parent_to_add))
        sort!(push!(node.parents_disc, parent_to_add))
    else
        @assert(!in(node.parents_cont, parent_to_add))
        sort!(push!(node.parents_cont, parent_to_add))
    end

    node
end
function _build_linear_gaussian_node(
    node::NodeInTraining,
    Y::Matrix{Float64},
    X_disc::Matrix{Float64},
    X_cont::Matrix{Float64},
    ind_old_to_new_disc::Dict{Int, Int}(),
    ind_old_to_new_cont::Dict{Int, Int}(),
    disc_parent_instantiations::Vector{Int},
    )

    index = length(ind_old_to_new_cont) + node.index

    parents_disc = Array(Int, length(node.parents_disc))
    for (i,p) in enumerate(node.parents_disc)
        parents_disc[i] = ind_old_to_new_disc[p]
    end

    parents_cont = Array(Int, length(node.parents_cont))
    for (i,p) in enumerate(node.parents_cont)
        parents_cont[i] = ind_old_to_new_cont[p]
    end

    m = nrow(Y)
    @assert(nrow(X_disc) == m)
    @assert(nrow(X_cont) == m)

    nparents_disc = length(parents_disc)
    nparents_cont = length(parents_cont)

    stats = LinearGaussianStats[]
    parental_assignments_disc = Array(Int, nparents_disc)
    parental_assignments_cont = Array(Float64, nparents_cont+1)
    parent_instantiation_counts_disc = Array(Int, nparents_disc)

    if nparents_cont > 0
        if nparents_disc > 0

            # perform ridge-regression and stdev calc online
            #   Xᵀy = (λI + XᵀX)w

            n_disc_parent_instantiations = 1
            for (i,p) in enumerate(node.parents_disc)
                parent_instantiation_counts_disc[i] = disc_parent_instantiations[p]
                n_disc_parent_instantiations *= disc_parent_instantiations[p]
            end

            # LHS is the Xᵀy vector [nparents+1]
            # RHS is the λI + XᵀX matrix [nparents+1 × nparents+1]

            LHS_arr = Array(Vector{Float64}, n_disc_parent_instantiations)
            RHS_arr = Array(Matrix{Float64}, n_disc_parent_instantiations)
            σ_var_arr = Array(Variance, n_disc_parent_instantiations)
            for i in 1 : n_disc_parent_instantiations
                LHS_arr[i] = zeros(Float64, nparents_cont+1)
                RHS_arr[i] = diagm(fill(λ, nparents_cont+1))
                σ_var_arr[i] = Variance()
            end

            x = Array(Float64, nparents_cont+1)
            x[end] = 1.0

            for i in 1 : m
                # pull contimuous data
                y = Y[node.index, i]
                for (j,p) in enumerate(parents_cont)
                    x[j] = X_cont[p,i]
                end

                # identify which one to update
                for (j,p) in enumerate(parents_disc)
                    parental_assignments_disc[j] = X_disc[p,i]
                end
                k = sub2ind_vec(parent_instantiation_counts_disc, parental_assignments_disc)

                lhs = LHS_arr[k]
                rhs = RHS_arr[k]
                fit!(σ_var_arr[k], y)

                # update ridge regression
                for j in 1 : nparents_cont+1
                    lhs[j] += x[j]*y

                    for q in 1 : nparents_cont
                        rhs[j,q] += x[j]*x[q]
                    end
                end
            end

            # solve ridge regressions
            #  w = (λI + XᵀX)⁻¹ Xᵀy

            stats = Array(LinearGaussianStats, n_disc_parent_instantiations)

            for i in 1 : n_disc_parent_instantiations
                σ_var = σ_var_arr[i]
                if nobs(σ_var) > 1
                    w = RHS_arr[k] \ lhs_arr[k]
                    σ = std(σ_var)
                    stats[i] =  LinearGaussianStats(w, σ)
                else
                    w = zeros(nparents_cont+1)
                    σ = 0.001 # default standard deviation
                    stats[i] =  LinearGaussianStats(w, σ)
                end
            end
        else
            # no discrete parents

            # solve a single linear regression problem:
            #   Xᵀy = (λI + XᵀX)w

            # LHS is the Xᵀy vector [nparents+1]
            # RHS is the λI + XᵀX matrix [nparents+1 × nparents+1]

            lhs = zeros(Float64, nparents_cont+1)
            rhs = diagm(fill(λ, nparents_cont+1))
            σ_var = Variance()

            x = Array(Float64, nparents_cont+1)
            x[end] = 1.0

            for i in 1 : m

                # pull continuous data
                y = Y[node.index, i]
                for (j,p) in enumerate(parents_cont)
                    x[j] = X_cont[p,i]
                end

                # update stdev
                fit!(σ_var, y)

                # update ridge regression
                for j in 1 : nparents_cont+1
                    lhs[j] += x[j]*y

                    for q in 1 : nparents_cont
                        rhs[j,q] += x[j]*x[q]
                    end
                end
            end

            # solve ridge regressions
            #  w = (λI + XᵀX)⁻¹ Xᵀy

            @assert(nobs(σ_var) > 1)
            w = RHS_arr[k] \ lhs_arr[k]
            σ = std(σ_var)
            stats = LinearGaussianStats[LinearGaussianStats(w, σ)]
        end
    else # no continuous parents (μ is fixed)
        if nparents_disc > 0

            n_disc_parent_instantiations = 1
            for (i,p) in enumerate(node.parents_disc)
                parent_instantiation_counts_disc[i] = disc_parent_instantiations[p]
                n_disc_parent_instantiations *= disc_parent_instantiations[p]
            end

            # NOTE: var can give us mean too
            var_arr = Array(Variance, n_disc_parent_instantiations)
            for i in 1 : n_disc_parent_instantiations
                var_arr[i] = Variance()
            end

            for i in 1 : m

                y = Y[node.index, i]

                # identify which one to update
                for (j,p) in enumerate(parents_disc)
                    parental_assignments_disc[j] = X_disc[p,i]
                end
                k = sub2ind_vec(parent_instantiation_counts_disc, parental_assignments_disc)

                # update online stat
                fit!(var_arr[k], y)
            end

            stats = Array(LinearGaussianStats, n_disc_parent_instantiations)
            for i in 1 : n_disc_parent_instantiations
                var = var_arr[i]
                if nobs(σ_var) > 1
                    w = Float64[mean(var)]
                    σ = std(var)
                    stats[i] = LinearGaussianStats(w, σ)
                elseif nobs(σ_var) > 0
                    w = Float64[mean(var)]
                    σ = 0.001 # default standard deviation
                    stats[i] = LinearGaussianStats(w, σ)
                else
                    μ = Float64[0.0]
                    σ = 0.001 # default standard deviation
                    stats[i] = LinearGaussianStats(w, σ)
                end
            end
        else
            # no discrete parents either

            var = Variance()
            fit!(var, Y[node.index, :])

            w = Float64[mean(var)]
            σ = std(var)
            stats = LinearGaussianStats[LinearGaussianStats(w, σ)]
        end
    end

    LinearGaussianNode(index, stats, parents_disc, parents_cont,
                       parental_assignments_disc,
                       parental_assignments_cont,
                       parent_instantiation_counts_disc,
                    )
end

function train(
    training_data::ModelTrainingData2,
    preallocated_data::LB_PreallocatedData,
    params::LB_TrainParams,
    fold::Int,
    fold_assignment::FoldAssignment,
    match_fold::Bool,
    )

    Y = copy_matrix_fold(preallocated_data.Y, fold, fold_assignment, match_fold)
    X_disc = copy_matrix_fold(preallocated_data.X_disc, fold, fold_assignment, match_fold)
    X_cont = copy_matrix_fold(preallocated_data.X_cont, fold, fold_assignment, match_fold)

    @assert(findfirst(v->isnan(v), Y) == 0)
    @assert(findfirst(v->isinf(v), Y) == 0)

    # -----------------------------------------
    # run structure learning
    #  - always add the next best edge

    node_lat = NodeInTraining(1)
    node_lon = NodeInTraining(2)

    score_cache = Dict{NodeInTraining, Float64}() # node -> (component_score)


    best_logl = _get_component_score!(score_cache, node_lat, Y, X_disc, X_cont) +
                _get_component_score!(score_cache, node_lon, Y, X_disc, X_cont)

    finished = false
    while !finished

        Δscore_lat, parent_to_add_lat, add_discrete_lat =
            _greedy_hillclimb_iter_on_node(node_lat, score_cache, Y, X_disc, X_cont, params.max_parents)
        Δscore_lon, parent_to_add_lon, add_discrete_lon =
            _greedy_hillclimb_iter_on_node(node_lon, score_cache, Y, X_disc, X_cont, params.max_parents)

        if max(Δscore_lat, Δscore_lon) < 0.001
            # no improvement
            finished = true
        else
            if Δscore_lat > Δscore_lon
                _apply_node_change!(node_lat, parent_to_add_lat, add_discrete_lat)
            else
                _apply_node_change!(node_lon, parent_to_add_lon, add_discrete_lon)
            end
        end
    end

    # -----------------------------------------
    # now build the model

    ind_old_disc = sort!(unique([node_lat.parents_disc; node_lon.parents_disc]))
    extractor_disc = FeatureSubsetExtractor(params.indicators[ind_old_disc])
    ind_old_to_new_disc = Dict{Int,Int}()
    for (o,n) in enumerate(ind_old_disc)
        ind_old_to_new_disc[o] = n
    end

    ind_old_cont = sort!(unique([node_lat.parents_cont; node_lon.parents_cont]))
    extractor_cont = FeatureSubsetExtractor(params.indicators[ind_old_cont])
    clamper_cont = get_clamper_subset(preallocated_data.clamper_cont, ind_old_cont, extractor_cont.x)
    ind_old_to_new_cont = Dict{Int,Int}()
    for (o,n) in enumerate(ind_old_cont)
        ind_old_to_new_cont[o] = n
    end


    model_node_lat = _build_linear_gaussian_node(node_lat, Y, X_disc, X_cont,
                                                 ind_old_to_new_disc, ind_old_to_new_cont)
    model_node_lon = _build_linear_gaussian_node(node_lon, Y, X_disc, X_cont,
                                                 ind_old_to_new_disc, ind_old_to_new_cont)

    sample_lat_first = true # TODO: update this once we can learn targets as parents



    LinearGaussianBayesianNetBehavior(
        params.targets, extractor_disc, extractor_cont,
        clamper_cont, preallocated_data.clamper_act,
        sample_lat_first, model_node_lat, model_node_lon
        )
end

end