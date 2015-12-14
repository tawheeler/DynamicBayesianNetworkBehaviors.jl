type GraphLearningResult
    fileroot     :: AbstractString
    target_lat   :: Union{AbstractFeature, FeaturesNew.AbstractFeature}
    target_lon   :: Union{AbstractFeature, FeaturesNew.AbstractFeature}
    parents_lat  :: Union{Vector{AbstractFeature}, Vector{FeaturesNew.AbstractFeature}}
    parents_lon  :: Union{Vector{AbstractFeature}, Vector{FeaturesNew.AbstractFeature}}
    features     :: Union{Vector{AbstractFeature}, Vector{FeaturesNew.AbstractFeature}}

    adj          :: BitMatrix               # this is of the size of the resulting network (ie, |f_inds|)
    stats        :: Vector{Matrix{Float64}} # NOTE(tim): this does not include prior counts
    bayescore    :: Float64

    function GraphLearningResult(
        basefolder     :: AbstractString,
        features       :: Union{Vector{AbstractFeature}, Vector{FeaturesNew.AbstractFeature}},
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