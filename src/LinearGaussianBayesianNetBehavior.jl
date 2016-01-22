
immutable LinearGaussianStats
    # μ = Ax where x[end] = 1
    A::Vector{Float64} # [nparents + 1]
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
end

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

function select_action(
    behavior::LinearGaussianBayesianNetBehavior,
    runlog::RunLog,
    sn::StreetNetwork,
    colset::UInt,
    frame::Int
    )

    FeaturesNew.observe!(behavior.extractor_disc, runlog, sn, colset, frame)
    FeaturesNew.observe!(behavior.extractor_cont, runlog, sn, colset, frame)
    FeaturesNew.process!(behavior.clamper_cont) # TODO - ensure that clamper_cont is tied to extractor_cont
    _cast_discrete_to_int!(behavior)
    _copy_to_assignment!(behavior)

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