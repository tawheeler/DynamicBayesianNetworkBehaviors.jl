module DynamicBayesianNetworkBehaviors

using AutomotiveDrivingModels
using NLopt
using Distributions
using BayesNets
using SmileExtra

import LightGraphs: topological_sort_by_dfs, in_neighbors, nv, ne
import Discretizers: encode
import AutomotiveDrivingModels:
    ModelTargets,
    AbstractVehicleBehavior,
    AbstractVehicleBehaviorPreallocatedData,
    AbstractVehicleBehaviorTrainParams,

    preallocate_learning_data,
    select_action,
    calc_action_loglikelihood,
    train,
    observe,
    _reverse_smoothing_sequential_moving_average,
    is_in_fold

export
    DBNModel,
    DynamicBayesianNetworkBehavior,
    DBNSimParams,
    GraphLearningResult,
    ParentFeatures,
    BN_TrainParams,

    DirichletPrior,
    UniformPrior,
    BDeuPrior,

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

    get_input_acceleration,
    get_input_turnrate,
    # infer_action_lon_from_input_acceleration,
    # infer_action_lat_from_input_turnrate,

    discretize,
    discretize_cleaned,
    drop_invalid_discretization_rows,
    convert_dataset_to_matrix,
    calc_bincounts_array,

    select_action,
    calc_action_loglikelihood,
    train

##############################################################

include("dirichlet_priors.jl")
include("DBNModel.jl")
include("behaviormodel.jl")
include("default_params.jl")
include("learning.jl")

end # module
