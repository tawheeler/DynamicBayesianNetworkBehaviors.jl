immutable DBNModel
    BN            :: BayesNet
    statsvec      :: Vector{Matrix{Float64}} # each matrix is [r×q], nvar_instantiations × nparent_instantiations
    features      :: Union{Vector{AbstractFeature},Vector{FeaturesNew.AbstractFeature}}
    discretizers  :: Vector{AbstractDiscretizer}
    istarget      :: BitVector # whether a feature is a target feature
end

function Base.print(io::IO, model::DBNModel)

    target_lat = get_target_lat(model)
    target_lon = get_target_lon(model)

    println(io, "DBNModel")
    println(io, "\t", symbol(target_lat), " <- ", map(f->symbol(f), get_indicators_for_target(model, target_lat)))
    println(io, "\t", symbol(target_lon), " <- ", map(f->symbol(f), get_indicators_for_target(model, target_lon)))
end

function dbnmodel{R<:Real, D<:AbstractDiscretizer}(
    BN:: BayesNet,
    statsvec::Vector{Matrix{R}},
    discretizerdict::Dict{Symbol, D},
    targets::Vector{AbstractFeature},
    indicators::Vector{AbstractFeature}
    )

    features = AbstractFeature[symbol2feature(node.name) for node in BN.nodes]
    discretizers = AbstractDiscretizer[discretizerdict[node.name] for node in BN.nodes]
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
function dbnmodel{R<:Real, D<:AbstractDiscretizer}(
    BN:: BayesNet,
    statsvec::Vector{Matrix{R}},
    discretizerdict::Dict{Symbol, D},
    targets::Vector{FeaturesNew.AbstractFeature},
    indicators::Vector{FeaturesNew.AbstractFeature}
    )

    features = FeaturesNew.AbstractFeature[FeaturesNew.symbol2feature(sym) for sym in names(BN)]
    discretizers = AbstractDiscretizer[discretizerdict[sym] for sym in names(BN)]
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
function dbnmodel{R<:Real, D<:AbstractDiscretizer}(
    BN::BayesNet,
    statsvec::Vector{Matrix{R}},
    discretizers::Vector{D},
    targets::Vector{AbstractFeature},
    indicators::Vector{AbstractFeature},
    )

    features = AbstractFeature[symbol2feature(sym) for sym in names(BN)]
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
function dbnmodel{R<:Real, D<:AbstractDiscretizer}(
    BN::BayesNet,
    statsvec::Vector{Matrix{R}},
    discretizers::Vector{D},
    targets::Vector{FeaturesNew.AbstractFeature},
    indicators::Vector{FeaturesNew.AbstractFeature},
    )

    features = FeaturesNew.AbstractFeature[symbol2feature(sym) for sym in names(BN)]
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
function dbnmodel{R<:Real, D<:AbstractDiscretizer}(
    adj::BitMatrix, # adj[i,j] = true means i → j
    statsvec::Vector{Matrix{R}},
    discretizerdict::Dict{Symbol, D},
    targets::Vector{AbstractFeature},
    indicators::Vector{AbstractFeature},
    )

    BN = build_bn(statsvec, targets, indicators, adj)
    dbnmodel(BN, statsvec, discretizerdict, targets, indicators)
end
function dbnmodel{R<:Real, D<:AbstractDiscretizer}(
    adj::BitMatrix, # adj[i,j] = true means i → j
    statsvec::Vector{Matrix{R}},
    discretizerdict::Dict{Symbol, D},
    targets::Vector{FeaturesNew.AbstractFeature},
    indicators::Vector{FeaturesNew.AbstractFeature},
    )

    BN = build_bn(statsvec, targets, indicators, adj)
    dbnmodel(BN, statsvec, discretizerdict, targets, indicators)
end
function dbnmodel(modelstats::Dict{AbstractString, Any})

    discretizerdict = modelstats["binmaps"]
    targets    = modelstats["targets"]
    indicators = modelstats["indicators"]
    stats      = modelstats["statistics"]
    adj        = modelstats["adjacency"] # adj[i,j] = true means i → j

    if eltype(targets) <: AbstractFeature
        dbnmodel(adj, stats, discretizerdict, convert(Vector{AbstractFeature}, targets),
                            convert(Vector{AbstractFeature}, indicators))
    else
        dbnmodel(adj, stats, discretizerdict, convert(Vector{FeaturesNew.AbstractFeature}, targets),
                            convert(Vector{FeaturesNew.AbstractFeature}, indicators))
    end
end
function dbnmodel(modelpstats_file::AbstractString)
    emstats = load(modelpstats_file)
    dbnmodel(emstats)
end

function build_bn{R<:Real}(
    statsvec   :: Vector{Matrix{R}},
    targets    :: Vector{FeaturesNew.AbstractFeature},
    indicators :: Vector{FeaturesNew.AbstractFeature}, # technically all of the non-target indicators
    adj        :: BitMatrix
    )

    @assert(size(adj,1) == size(adj,2)) # is square

    n_targets = length(targets)
    n_indicators = length(indicators)

    bnnames = Array(Symbol, n_targets + n_indicators)
    for (i,f) in enumerate(targets)
        bnnames[i] = symbol(f)
    end
    for (i,f) in enumerate(indicators)
        bnnames[i+n_targets] = symbol(f)
    end

    @assert(length(unique(bnnames)) == length(bnnames))
    n_nodes = length(bnnames)
    @assert(n_nodes == size(adj, 1) == size(adj, 2))

    BN = BayesNet(bnnames)

    r_arr = Array(Int, n_nodes)
    for (i,node_sym) in enumerate(bnnames)
        stats = statsvec[i]
        r, q = size(stats) # r = num node instantiations, q = num parental instantiations
        states = collect(1:r)
        BN.nodes[i].domain = BayesNets.CPDs.DiscreteDomain(states)
        r_arr[i] = r
    end

    for (node,node_sym) in enumerate(bnnames)

        stats = statsvec[node]
        r, q = size(stats)
        states = collect(1:r)

        stats .+= 1 # NOTE(tim): adding uniform prior
        probabilities = stats ./ sum(stats,1)

        # set any parents & populate probability table
        n_parents = sum(adj[:,node])
        if n_parents > 0
            bnparents = bnnames[adj[:,node]]
            for pa in bnparents
                add_edge!(BN, pa, node_sym)
            end

            # populate probability table
            assignments = BayesNets.assignment_dicts(BN, bnparents)
            # parameterFunction = BayesNets.discrete_parameter_function(assignments, vec(probabilities), r)
            # setCPD!(BN, node_sym, CPDs.Discrete(states, parameterFunction))

            parameterlookup = BayesNets.discrete_parameter_dict(assignments, vec(probabilities), r)
            set_CPD!(BN, node_sym, CPDs.DiscreteDict(states, parameterlookup))
        else
            # no parents
            # setCPD!(BN, node_sym, CPDs.Discrete(states, vec(probabilities)))

            set_CPD!(BN, node_sym, CPDs.DiscreteStatic(states, vec(probabilities)))
        end

    end

    return BN
end
function build_bn{R<:Real}(
    statsvec   :: Vector{Matrix{R}},
    targets    :: Vector{AbstractFeature},
    indicators :: Vector{AbstractFeature}, # technically all of the non-target indicators
    adj        :: BitMatrix
    )

    @assert(size(adj,1) == size(adj,2)) # is square

    n_targets = length(targets)
    n_indicators = length(indicators)

    bnnames = Array(Symbol, n_targets + n_indicators)
    for (i,f) in enumerate(targets)
        bnnames[i] = symbol(f)
    end
    for (i,f) in enumerate(indicators)
        bnnames[i+n_targets] = symbol(f)
    end

    @assert(length(unique(bnnames)) == length(bnnames))
    n_nodes = length(bnnames)
    @assert(n_nodes == size(adj, 1) == size(adj, 2))

    BN = BayesNet(bnnames)

    r_arr = Array(Int, n_nodes)
    for (node,node_sym) in enumerate(bnnames)
        stats = statsvec[node]
        r, q = size(stats) # r = num node instantiations, q = num parental instantiations
        states = collect(1:r)
        BN.domains[node] = DiscreteDomain(states)
        r_arr[node] = r
    end

    for (node,node_sym) in enumerate(bnnames)

        stats = statsvec[node]
        r, q = size(stats)
        states = collect(1:r)

        stats .+= 1 # NOTE(tim): adding uniform prior
        probabilities = stats ./ sum(stats,1)

        # set any parents & populate probability table
        n_parents = sum(adj[:,node])
        if n_parents > 0
            bnparents = bnnames[adj[:,node]]
            for pa in bnparents
                add_edge!(BN, pa, node_sym)
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
is_target_lat(f::FeaturesNew.AbstractFeature) = isa(f, FeaturesNew.Feature_FutureDesiredAngle)
function is_target_lon(f::AbstractFeature)
    isa(f, Features.Feature_FutureDesiredSpeed_250ms) ||
    isa(f, Features.Feature_FutureDesiredSpeed_500ms) ||
    isa(f, Features.Feature_FutureAcceleration_250ms) ||
    isa(f, Features.Feature_FutureAcceleration_500ms)
end
is_target_lon(f::FeaturesNew.AbstractFeature) = isa(f, FeaturesNew.Feature_FutureAcceleration)

indexof(f::Symbol, model::DBNModel) = model.BN.name_to_index[f]
indexof(f::AbstractFeature, model::DBNModel) = model.BN.name_to_index[symbol(f)]
indexof(f::FeaturesNew.AbstractFeature, model::DBNModel) = model.BN.name_to_index[symbol(f)]
is_parent(model::DBNModel, parent::Int, child::Int) = in(parent, in_neighbors(model.BN.dag, child))
is_parent(model::DBNModel, parent::Symbol, child::Symbol) = is_parent(model, model.BN.name_to_index[parent], model.BN.name_to_index[child])
function parent_indeces(varindex::Int, model::DBNModel)
    parent_names = BayesNets.parents(model.BN, names(model.BN)[varindex])
    retval = Array(Int, length(parent_names))
    for (i, name) in enumerate(parent_names)
        retval[i] = model.BN.name_to_index[name]
    end
    retval
end

get_targets(model::DBNModel) = model.features[model.istarget]
function get_target_lat(model::DBNModel, targets::Vector{AbstractFeature}=get_targets(model))
    ind = findfirst(f->is_target_lat(f), targets)
    targets[ind]
end
function get_target_lat(model::DBNModel, targets::Vector{FeaturesNew.AbstractFeature}=get_targets(model))
    ind = findfirst(f->is_target_lat(f), targets)
    targets[ind]
end
function get_target_lon(model::DBNModel, targets::Vector{AbstractFeature}=get_targets(model))
    ind = findfirst(f->is_target_lon(f), targets)
    targets[ind]
end
function get_target_lon(model::DBNModel, targets::Vector{FeaturesNew.AbstractFeature}=get_targets(model))
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
function get_indicators_for_target(model::DBNModel, target::FeaturesNew.AbstractFeature)
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

get_num_vertices(model::DBNModel) = nv(model.BN.dag)
get_num_edges(model::DBNModel) = ne(model.BN.dag)

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

    dims = tuple(bincounts[parentindeces]...)
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
        parentassignments[i] = assignments[model.BN.nodes[ind].name]
    end
    get_counts_for_assignment(model, targetind, parentindeces, parentassignments, bincounts)
end

function encode!(assignment::Dict{Symbol,Int}, model::DBNModel, observations::Dict{Symbol,Float64})
    # take each observation and bin it appropriately
    # returns a Dict{Symbol,Int}

    # TODO(tim): ensure order is correct
    for (i,istarget) in enumerate(model.istarget)
        if !istarget
            sym = model.BN.nodes[i].name
            val = observations[sym]
            try
                encode(model.discretizers[i], val)
            catch
                println("failed to encode ", sym, " of value ", val, "  ", model.discretizers[i])
            end
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

    for node in model.BN.nodes[ordering]
        name = node.name
        cpd = BayesNets.cpd(model.BN, name)

        p = BayesNets.probvec(cpd, assignment)
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
function sample_unset!(model::DBNModel, assignment::Dict{Symbol, Int}, ordering::Vector{Int}=topological_sort_by_dfs(model.BN.dag); missing_value::Int=-1)
    #=
    Run through nodes in topological order, building the instantiation vector as we go
    We use nodes we already know to condition on the distribution for nodes we do not
    Modifies assignment to include newly sampled symbols
    Only sample values set to missing_value
    =#

    for node in model.BN.nodes[ordering]

        name = node.name

        if get(assignment, name, missing_value) == missing_value

            cpd = BayesNets.cpd(model.BN, name)

            p = BayesNets.probvec(cpd, assignment)
            r = rand()
            i = 1
            p_tot = 0.0
            while p_tot + p[i] < r && i < length(p)
                p_tot += p[i]
                i += 1
            end
            assignment[name] = cpd.domain[i]
        end
    end

    assignment
end
function sample_and_logP!(
    model::DBNModel,
    assignment::Dict{Symbol, Int},
    logPs::Dict{Symbol, Float64},

    ordering::Vector{Int}=topological_sort_by_dfs(model.BN.dag),
    )

    for node in model.BN.nodes[ordering]
        name = node.name

        cpd = BayesNets.cpd(model.BN, name)

        p = BayesNets.probvec(cpd, assignment)

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
    copy!(dest, BayesNets.probvec(cpd, assignment))
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

function export_to_text(model::DBNModel, filename::AbstractString)
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

function construct_model_adjacency_matrix(
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