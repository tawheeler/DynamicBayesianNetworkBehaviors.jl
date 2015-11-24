type ParentFeatures
    lat :: Vector{AbstractFeature}
    lon :: Vector{AbstractFeature}

    ParentFeatures() = new(AbstractFeature[], AbstractFeature[])
    ParentFeatures(lat::Vector{AbstractFeature}, lon::Vector{AbstractFeature}) = new(lat, lon)
end

type GraphLearningResult
    fileroot     :: AbstractString
    target_lat   :: AbstractFeature
    target_lon   :: AbstractFeature
    parents_lat  :: Vector{AbstractFeature}
    parents_lon  :: Vector{AbstractFeature}
    features     :: Vector{AbstractFeature}

    adj          :: BitMatrix               # this is of the size of the resulting network (ie, |f_inds|)
    stats        :: Vector{Matrix{Float64}} # NOTE(tim): this does not include prior counts
    bayescore    :: Float64

    function GraphLearningResult(
        basefolder     :: AbstractString,
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


        net_feature_indeces = _feature_indeces_in_net(ind_lat, ind_lon, parentinds_lat, parentinds_lon)
        net_features = features[net_feature_indeces]
        net_ind_lat = 1
        net_ind_lon = 2
        net_parent_indices_lat = _find_index_mapping(parentinds_lat, net_feature_indeces)
        net_parent_indices_lon = _find_index_mapping(parentinds_lon, net_feature_indeces)
        num_net_features = length(net_feature_indeces)

        # println("net_feature_indeces:    ", net_feature_indeces)
        # println("net_features:           ", net_features)
        # println("net_parent_indices_lat: ", net_parent_indices_lat)
        # println("net_parent_indices_lon: ", net_parent_indices_lon)
        # println("num_net_features:       ", num_net_features)

        adj   = construct_model_adjacency_matrix(net_ind_lat, net_ind_lon, net_parent_indices_lat, net_parent_indices_lon, num_net_features)
        stats = convert(Vector{Matrix{Float64}}, SmileExtra.statistics(adj, r[net_feature_indeces], d[net_feature_indeces,:]))

        # println("adj: ")
        # println(adj)

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
    continuous::Matrix{Float64} # [nfeatures×nsamples] never overwritten
    discrete::Matrix{Int}       # [nfeatures×nsamples] is overwritten as the discretization params change
    bincounts::Vector{Int}      # [nfeatures]          is overwritten as the discretization params change

    function ModelData(continuous::Matrix{Float64}, modelparams::ModelParams, features)
        d = discretize(modelparams.binmaps, continuous, features)
        r = calc_bincounts_array(modelparams.binmaps)
        new(continuous, d, r)
    end
end


type ModelPreallocatedData <: VehicleBehaviorPreallocatedData
    #=
    Allocates the full size of continuous and discrete corresponding to the full dataset
    It then copies data into it based on the given fold assignment
    =#

    continuous::Matrix{Float64} # [nsamples×nfeatures]
    discrete::Matrix{Int}       # [nsamples×nfeatures]
    bincounts::Vector{Int}      # [nfeatures]
    rowcount::Int               # number of populated entries

    function ModelPreallocatedData(dset::ModelTrainingData, args::Dict{Symbol,Any})
        nsamples = nrow(dset.dataframe)

        nfeatures = length(get(args,:indicators,DEFAULT_INDICATORS)) + 2

        continuous = Array(Float64, nsamples, nfeatures)
        discrete = Array(Int, nsamples, nfeatures)
        bincounts = Array(Int, nfeatures)
        rowcount = 0
        new(continuous, discrete, bincounts, rowcount)
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
    DEFAULT_DISCRETIZERS[:f_accel_250ms         ] = LinearDiscretizer([-5.00,-3.045338252335337,-2.0,-1.50593693678868,-0.24991770599882523,0.06206203400761478,0.2489269478410686,0.5,3.0], Int)
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

    # creates a Matrix{Float64} which is nfeatures × nsamples
    # rows are ordered in the same order as the features vector

    nsamples = nrow(dataframe)
    nfeatures = length(features)

    mat = Array(Float64, nfeatures, nsamples)
    for j = 1 : nsamples
        for (i,f) in enumerate(features)
            sym = symbol(f)
            mat[i,j] = dataframe[j,sym]
        end
    end
    mat
end

function get_emstats(res::GraphLearningResult, binmapdict::Dict{Symbol, AbstractDiscretizer})
    emstats = Dict{AbstractString, Any}()
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
function _feature_indeces_in_net(
    find_lat::Int,
    find_lon::Int,
    parents_lat::Vector{Int},
    parents_lon::Vector{Int}
    )

    net_feature_indeces = [find_lat,find_lon]
    for p in sort!(unique([parents_lat; parents_lon]))
        if !in(p, net_feature_indeces)
            push!(net_feature_indeces, p)
        end
    end
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
    extrema_lat::Tuple{Float64, Float64},
    extrema_lon::Tuple{Float64, Float64}
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
    extrema::Tuple{Float64, Float64}, # upper and lower bounds
    ncandidate_bins::Int # number of evenly spaced candidate binedges to consider in the pretraining phase
    )

    lo, hi = extrema
    ncenter_bins = nbins-1
    candidate_binedges = collect(linspace(lo, hi, ncandidate_bins+2))
    bincounts = calc_bincounts(data_sorted_ascending, candidate_binedges)
    candidate_binedges = candidate_binedges[2:end-1]
    ncandidate_binedges = length(candidate_binedges)

    binedge_assignments = collect(1:ncenter_bins)

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
    extrema::Tuple{Float64, Float64}, # upper and lower bounds
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
function SmileExtra.statistics(
    targetind::Int,
    parents::AbstractVector{Int},
    data::ModelPreallocatedData,
    )

    bincounts = data.bincounts
    discrete_data = data.discrete
    rowcount = data.rowcount

    q = 1
    if !isempty(parents)
        Np = length(parents)
        q  = prod(bincounts[parents])
        stridevec = fill(1, Np)
        for k = 2:Np
            stridevec[k] = stridevec[k-1] * bincounts[parents[k-1]]
        end
        js = (discrete_data[1:rowcount,parents] - 1) * stridevec + 1
    else
        js = fill(1, rowcount)
    end

    full(sparse(vec(discrete_data[1:rowcount,targetind]), vec(js), 1, bincounts[targetind], q))
end
function SmileExtra.log_bayes_score_component{I<:Integer}(
    i::Int,
    parents::AbstractVector{I},
    data::ModelPreallocatedData,
    )

    #=
    Computes the bayesian score component for the given target variable index
        This assumes a unit dirichlet prior (alpha)

    INPUT:
        i       - index of the target variable
        parents - list of indeces of parent variables (should not contain self)
        data    - will use data.discrete and data.bincounts

    OUTPUT:
        the log bayesian score, Float64
    =#

    r = data.bincounts
    d = data.discrete
    rowcount = data.rowcount

    #=
        r       - list of instantiation counts accessed by variable index
                  r[1] gives number of discrete states variable 1 can take on
        d       - matrix of sample counts
                  d[k,j] gives the number of times the target variable took on its kth instantiation
                  given the jth parental instantiation
                  NOTE: that is the opposite convention (transpose) of the original log_bayes_score_component function!
                  NOTE: that we only use the first 'data.rowcount' samples!
    =#

    alpha = fill(1, r[i], isempty(parents) ? 1 : prod(r[parents]))

    nfeatures = size(d,2)

    if !isempty(parents)
        Np = length(parents)
        stridevec = fill(1, Np)
        for k = 2:Np
            stridevec[k] = stridevec[k-1] * r[parents[k-1]]
        end
        js = (d[1:rowcount,parents] - 1) * stridevec + 1
    else
        js = fill(1, rowcount)
    end

    N = sparse(vec(d[1:rowcount,i]), vec(js), 1, size(alpha)...) # note: duplicates are added together

    return sum(lgamma(alpha + N)) - sum(lgamma(alpha)) + sum(lgamma(sum(alpha,1))) - sum(lgamma(sum(alpha,1) + sum(N,1)))::Float64
end
function SmileExtra.log_bayes_score_component{I<:Integer}(
    i::Int,
    parents::AbstractVector{I},
    data::ModelPreallocatedData,
    cache::Dict{Vector{Int}, Float64}
    )

    if haskey(cache, parents)
        return cache[parents]
    end
    return (cache[parents] = log_bayes_score_component(i, parents, data))
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
function calc_bayesian_score(
    data::ModelPreallocatedData,
    modelparams::ModelParams,
    staticparams::ModelStaticParams
    )

    # NOTE: this does not compute the score components for the indicator variables

    log_bayes_score_component(staticparams.ind_lat, modelparams.parents_lat, data) +
        log_bayes_score_component(staticparams.ind_lon, modelparams.parents_lon, data)
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
function calc_discretize_score(
    data::ModelPreallocatedData,
    modelparams::ModelParams,
    staticparams::ModelStaticParams
    )

    score = 0.0

    binmap = modelparams.binmaps[staticparams.ind_lat]
    stats = SmileExtra.statistics(staticparams.ind_lat, modelparams.parents_lat, data)
    score += calc_discretize_score(binmap, stats)
    @assert(!isinf(score))

    binmap = modelparams.binmaps[staticparams.ind_lon]
    stats = SmileExtra.statistics(staticparams.ind_lon, modelparams.parents_lon, data)
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
    data::ModelPreallocatedData,
    stats::AbstractMatrix{Int}
    )

    calc_discretize_score(target_binmap, stats) +
        log_bayes_score_component(target_index, target_parents, data)
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
    data::ModelPreallocatedData
    )

    stats = SmileExtra.statistics(target_index, target_parents, data)
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
function calc_component_score(
    target_index::Int,
    target_parents::Vector{Int},
    target_binmap::AbstractDiscretizer,
    data::ModelPreallocatedData,
    score_cache::Dict{Vector{Int}, Float64}
    )

    stats = SmileExtra.statistics(target_index, target_parents, data)
    calc_discretize_score(target_binmap, stats) +
        log_bayes_score_component(target_index, target_parents, data, score_cache)
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
function calc_complete_score(
    data::ModelPreallocatedData,
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
    extrema::Tuple{Float64, Float64}, # upper and lower bounds
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
    data::Union{ModelData, ModelPreallocatedData};
    forced::Tuple{Vector{Int}, Vector{Int}}=(Int[], Int[]), # lat, lon
    verbosity::Integer=0,
    max_parents::Integer=6
    )

    binmaps = modelparams.binmaps
    parents_lat = deepcopy(modelparams.parents_lat)
    parents_lon = deepcopy(modelparams.parents_lon)

    forced_lat, forced_lon = forced
    parents_lat = sort(unique([parents_lat; forced_lat]))
    parents_lon = sort(unique([parents_lon; forced_lon]))

    features = staticparams.features
    ind_lat = staticparams.ind_lat
    ind_lon = staticparams.ind_lon
    binmap_lat = modelparams.binmaps[ind_lat]
    binmap_lon = modelparams.binmaps[ind_lon]

    n_targets = 2
    n_indicators = length(features)-n_targets

    chosen_lat = map(i->in(n_targets+i, parents_lat), 1:n_indicators)
    chosen_lon = map(i->in(n_targets+i, parents_lon), 1:n_indicators)

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
                chosen_lat = map(k->in(n_targets+k, parents_lat), 1:n_indicators)
                score += score_diff
                score_lat += score_diff
                if verbosity > 0
                    println("changed lat:", map(f->symbol(f), features[parents_lat]))
                    println("new score: ", score)
                end
            else
                parents_lon = new_parents_lon
                chosen_lon = map(k->in(n_targets+k, parents_lon), 1:n_indicators)
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
    data::Union{ModelData, ModelPreallocatedData},
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
    data::Union{ModelData, ModelPreallocatedData},
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
    binmap::LinearDiscretizer,
    index::Int,
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelPreallocatedData
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

        for i in 1 : nbins-1
            binmap.binedges[i+1] = max(bin_edges[i], binmap.binedges[i] + ε)
        end
        for i in nbins : -1 : 2
            binmap.binedges[i] = min(binmap.binedges[i], binmap.binedges[i+1]-ε)
        end

        # rediscretize
        for i in 1 : data.rowcount
            data.discrete[i,index] = encode(binmap, data.continuous[i,index])
        end

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
function optimize_indicator_bins!(
    binmap::HybridDiscretizer,
    index::Int,
    modelparams::ModelParams,
    staticparams::ModelStaticParams,
    data::ModelPreallocatedData
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
        for i in 1 : data.rowcount
            data.discrete[i,index] = encode(binmap, data.continuous[i,index])
        end

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

trains_with_nona(::DynamicBayesianNetworkBehavior) = false
trains_with_nona(::Type{DynamicBayesianNetworkBehavior}) = false
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

        for j in 1 : length(datavec)
            datavec[j] = continuous_data[ind_lat,j]
        end

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
        copy!(disc.binedges, optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins))

        ###################
        # lon

        for j in 1 : length(datavec)
            datavec[j] = continuous_data[ind_lon,j]
        end

        disc = modelparams.binmaps[ind_lon]::LinearDiscretizer
        extremes = (disc.binedges[1], disc.binedges[end])
        nbins = nlabels(disc)
        if nbins_lon != -1
            nbins = nbins_lon
            disc = LinearDiscretizer(linspace(extremes[1], extremes[2], nbins+1))
            modelparams.binmaps[ind_lon] = disc
        end

        for (i,x) in enumerate(datavec)
            if !(extremes[1] ≤ x ≤ extremes[2])
                println(extremes[1], " ≤ ", x, " ≤ ", extremes[2])
            end
            @assert(extremes[1] ≤ x ≤ extremes[2])
        end
        copy!(disc.binedges, optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins))

        if verbosity > 0
            toc()
        end
    end

    # is_indicator_valid = trues(length(indicators))
    if preoptimize_indicator_bins

        if verbosity > 0
            println("Optimizing Indicator Bins"); tic()
        end

        for i in 3 : length(features)
            # println(symbol(staticparams.features[i]), "  ", typeof(modelparams.binmaps[i]))
            disc2 = modelparams.binmaps[i]
            if isa(disc2, LinearDiscretizer)

                for j in 1 : length(datavec)
                    datavec[j] = continuous_data[i,j]
                end

                nbins = nlabels(disc2)
                extremes = (disc2.binedges[1], disc2.binedges[end])
                for (k,x) in enumerate(datavec)
                    datavec[k] = clamp(x, extremes[1], extremes[2])
                end
                copy!(disc2.binedges, optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins))

            elseif isa(disc2, HybridDiscretizer)

                for j in 1 : length(datavec)
                    datavec[j] = continuous_data[i,j]
                end

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
                    copy!(disc2.lin.binedges, optimize_categorical_binning!(datavec[1:k], nbins, extremes, ncandidate_bins))
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
function train(
    ::Type{DynamicBayesianNetworkBehavior},
    training_data::ModelTrainingData,
    fold::Int,
    fold_assignment::FoldAssignment,
    match_fold::Bool,
    preallocated_data::ModelPreallocatedData;

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
    features = [targets; indicators]

    ind_lat, ind_lon = find_target_indeces(targetset, features)
    parents_lat, parents_lon = get_parent_indeces(starting_structure, features)
    forced_lat, forced_lon = get_parent_indeces(forced, features)

    parents_lat = sort(unique([parents_lat; forced_lat]))
    parents_lon = sort(unique([parents_lon; forced_lon]))

    modelparams = ModelParams(map(f->discretizerdict[symbol(f)], features), parents_lat, parents_lon)
    staticparams = ModelStaticParams(ind_lat, ind_lon, features)

    ############################################################
    # pull the dataset
    #  - drop invalid discretization rows
    #  - check that everything is within the folds

    @assert(length(fold_assignment.frame_assignment) == nrow(training_data.dataframe))

    rowcount = 0
    for (i,a) in enumerate(fold_assignment.frame_assignment)
        if is_in_fold(fold, a, match_fold)
            rowcount += 1
            for (j,f) in enumerate(features)
                sym = symbol(f)
                dmap = discretizerdict[sym]
                value = training_data.dataframe[i, sym]::Float64
                @assert(!isnan(value))
                if supports_encoding(dmap, value) &&
                 !(isinf(value) && (j == ind_lat || j == ind_lon))
                    preallocated_data.continuous[rowcount, j] = value
                else
                    rowcount -= 1
                    break
                end
            end
        end
    end
    preallocated_data.rowcount = rowcount

    ############################################################

    datavec = Array(Float64, rowcount)
    if preoptimize_target_bins
        if verbosity > 0
            println("Optimizing Target Bins"); tic()
        end

        ###################
        # lat

        for i in 1 : rowcount
            datavec[i] = preallocated_data.continuous[i,ind_lat]
        end

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
        copy!(disc.binedges, optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins))

        ###################
        # lon

        for i in 1 : rowcount
            datavec[i] = preallocated_data.continuous[i,ind_lon]
        end

        disc = modelparams.binmaps[ind_lon]::LinearDiscretizer
        extremes = (disc.binedges[1], disc.binedges[end])
        nbins = nlabels(disc)
        if nbins_lon != -1
            nbins = nbins_lon
            disc = LinearDiscretizer(linspace(extremes[1], extremes[2], nbins+1))
            modelparams.binmaps[ind_lon] = disc
        end

        for (i,x) in enumerate(datavec)
            if !(extremes[1] ≤ x ≤ extremes[2])
                println(extremes[1], " ≤ ", x, " ≤ ", extremes[2])
            end
            @assert(extremes[1] ≤ x ≤ extremes[2])
        end
        copy!(disc.binedges, optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins))

        if verbosity > 0
            toc()
        end
    end

    # is_indicator_valid = trues(length(indicators))
    if preoptimize_indicator_bins

        if verbosity > 0
            println("Optimizing Indicator Bins"); tic()
        end

        for i in 3 : length(features) # skip the target features
            # println(symbol(staticparams.features[i]), "  ", typeof(modelparams.binmaps[i]))
            disc2 = modelparams.binmaps[i]
            if isa(disc2, LinearDiscretizer)

                for j in 1 : rowcount
                    datavec[j] = preallocated_data.continuous[j,ind_lon]
                end

                nbins = nlabels(disc2)
                extremes = (disc2.binedges[1], disc2.binedges[end])
                for (k,x) in enumerate(datavec)
                    datavec[k] = clamp(x, extremes[1], extremes[2])
                end
                copy!(disc2.binedges, optimize_categorical_binning!(datavec, nbins, extremes, ncandidate_bins))

            elseif isa(disc2, HybridDiscretizer)

                for j in 1 : rowcount
                    datavec[j] = preallocated_data.continuous[j,ind_lon]
                end

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
                    copy!(disc2.lin.binedges, optimize_categorical_binning!(datavec[1:k], nbins, extremes, ncandidate_bins))
                # else
                #     is_indicator_valid[i] = false
                end
            end
        end

        if verbosity > 0
            toc()
        end
    end

    ############################################################################
    # Discretize preallocated_data.continuous → preallocated_data.discrete
    # Compute bincounts

    for (j,dmap) in enumerate(modelparams.binmaps)
        preallocated_data.bincounts[j] = nlabels(dmap)
        for i in 1 : preallocated_data.rowcount
            value = preallocated_data.continuous[i,j]
            preallocated_data.discrete[i,j] = encode(dmap, value)
        end
    end

    ############################################################################

    starttime = time()
    iter = 0
    score = calc_complete_score(preallocated_data, modelparams, staticparams)

    if optimize_structure || optimize_target_bins || optimize_parent_bins
        score_diff = Inf
        SCORE_DIFF_THRESHOLD = 10.0
        while score_diff > SCORE_DIFF_THRESHOLD

            iter += 1
            verbosity == 0 || @printf("\nITER %d: %.4f (Δ%.4f) t=%.0f\n", iter, score, score_diff, time()-starttime)

            if optimize_structure
                optimize_structure!(modelparams, staticparams, preallocated_data, forced=(forced_lat, forced_lon), max_parents=max_parents, verbosity=verbosity)
            end
            if optimize_target_bins
                optimize_target_bins!(modelparams, staticparams, preallocated_data)
            end
            if optimize_parent_bins
                optimize_parent_bins!(modelparams, staticparams, preallocated_data)
            end

            score_new = calc_complete_score(preallocated_data, modelparams, staticparams)
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
                              preallocated_data.bincounts, preallocated_data.discrete[1:preallocated_data.rowcount,:]')
    model = dbnmodel(get_emstats(res, binmapdict))

    DynamicBayesianNetworkBehavior(model, DBNSimParams(), DBNSimParams())
end