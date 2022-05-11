using Distances
using NearestNeighbors
using SharedArrays
using Distributed

function faSTICS(data, minpt, xi, eps, pworkers, dist=euclidean)
#
#   inputs:
#       data - MxD array, M samples of D dimensional data
#       minpt, xi, eps - optics clustering parameters
#       distance - distance measure function
#   outputs:
#       labels - labels assigned to each sample
#
#
    n = size(data)[1]
    core = fill(0.0, n); # array to save core distances of each point
    labels = SharedArray(fill(-2, n)); # output array
    numlabels = SharedArray(fill(0,pworkers)); # array to save number of clusters identified by each thread

    # Partition data to distribute amoung processors
    workerAssignments = partition!(data, pworkers);
    

    # main loop, compute OPTICS clusters for each partition on seperate threads
    @sync @distributed for worker in 1:pworkers

        # Compute labels for each thread and save it to the shared labels array
        labels[workerAssignments .== worker] = ExtractLabels(ExtractClustering(OPTICS(data[workerAssignments .== worker, :], core[workerAssignments .== worker], eps, minpt, dist, xi, sum(workerAssignments.==worker))))
        
        #update number of identified clusters for this partition
        if length(labels[workerAssignments .== worker])>0
            numlabels[worker] += maximum(labels[workerAssignments .== worker])
        end
    end

    # Correct labels
    for worker in 2:pworkers
        labels[workerAssignments .== worker .& labels .!= -1] .+= sum(numlabels[1:worker-1])
    end

    # recompute OPTICS labels on points not previously clustered and update
    newLabs = ExtractLabels(ExtractClustering(OPTICS(data[labels .== -1, :], core[labels .== -1], epsi, minpts, dist, xi, length(core[labels .== -1]))))
    labels[labels .== -1] = newLabs .+ (newLabs.!=-1)*numlabels[end]

    return labels

end



@everywhere function partition!(data, pworkers)
#
#   inputs:
#       data - MxD array, M samples of D dimensional data
#       minpt, eps - optics clustering parameters
#       distance - distance measure function
#       core - array to store core distances
#   output:
#       workerassignment - array 
#
    n = size(data)[1]

    batchsize = ceil(Int, n/pworkers) # number of points to put in each partition
    available = fill(true, n) # boolean array tracking which points have not been assigned a partition
    workerassignment = fill(0, n) # output
    idx = collect(1:n)  # indexing array
    tree = BallTree(data')

     for worker in 1:pworkers
        nextCenter = idx[available][1] # arbitrarily choose the next "center point" of the partition
        neighbors= knn(tree, data[nextCenter, :], batchsize) # find the batchsize nearest neighbors
        workerassignment[neighbors] .= worker # assign those neighbors to this worker
        available[neighbors] .= false
    end
    
    return SharedArray(workerassignment)
end

function OPTICS(data, ε::Real, minpts::Int, dist, xi::Real)


    n = size(data)[1]
    idxer = 1:n

    order = fill(-1,n) # clustering order
    processed = fill(false, n)
    reachability = fill(Inf, n) # reachability distance
    index = Array{Bool}(undef, n)

    tree = BallTree(data')   
    
    # calculate core distances
    core = getCoreDist(data, tree, minpts)
    core[core.>ε] .= Inf
    

    current = 1 # keeps track of the current point being expanded
    for orderingIdx in 1:n
        # pick the next object to process to be the next unprocessed
        # object with the smallest reachability distance
        index = idxer[.!processed]
        current = index[argmin(reachability[index])]
        processed[current] = true
        order[orderingIdx] = current

        if core[current] != Inf
            setReachDist!(core, reachability, current, processed, data, tree, dist, ε)
        end
    end
    return order, reachability, minpts, xi
end

function getCoreDist(data, tree, minpts)
    # calculate the distance from the minpts-th nearest neighbor for each point      
    core = [a[end] for a in knn(tree, data', minpts+1, true)[2]]
    return core
end

function setReachDist!(core, reachability, current, processed, data, tree, dist, ε)

    # get all unprocessed neighbors within ε of the current center point
    neighbors = inrange(tree, data[current, :], ε, true)
    unprocessed = neighbors[processed[neighbors] .== false]
    if size(unprocessed)[1]>0
        # find distance between each such neighbor and the current center point
        distances = [dist(data[current, :], data[a, :]) for a in unprocessed]
        # compute reachability distance to be max of the core distance and these distances
        newReaches = max.(distances, core[current]) 

        # update reachability distance for points whose reachability distance improves this
        improvedNeighbors = newReaches .< reachability[unprocessed]
        reachability[unprocessed[improvedNeighbors]] = newReaches[improvedNeighbors]
    end

end



function ExtractClustering((order, reachability, minpts, ξ))

    n = length(order)
    r = reachability[order]

    SetOfSteepDownAreas = []
    SetOfClusters = []
    mib = 0.0
    ξ_compliment = 1-ξ

    ratio = [r[1:(end-1)] ./ r[2:end]; Inf]
    SteepDown = ratio .>= 1/ξ_compliment
    SteepUp = ratio .<= ξ_compliment
    down = ratio .>= 1
    up = ratio .<= 1

    index = 1
    while index < n

        mib = max(mib, r[index])

        if SteepDown[index]
            filterUpdate!(SetOfSteepDownAreas, ξ_compliment, mib, r)

            D_start = index
            # expand down area
            laststeep = index
            index += 1
            nonsteep = 0
            while nonsteep < minpts && index <= n
                if down[index]
                    if !SteepDown[index]
                        nonsteep += 1
                    else
                        laststeep = index
                    end
                    index += 1
                else
                    break
                end
            end
            D_end = laststeep
            index = D_end+1
            push!(SetOfSteepDownAreas, [(D_start, D_end), 0.0])
            # end expand down area
            mib = r[index-1]
           
        elseif SteepUp[index]
            filterUpdate!(SetOfSteepDownAreas, ξ_compliment, mib, r)
            
            U_start = index
            # expand down area
            laststeep = index
            index += 1
            nonsteep = 0
            while nonsteep < minpts && index <= n
                if up[index]
                    if !SteepUp[index]
                        nonsteep += 1
                    else
                        laststeep = index
                    end
                    index += 1
                else
                    break
                end
            end
            U_end = laststeep
            index = U_end+1
            # end expand down area
            mib = r[index]

            U_clusters = []
            for D in SetOfSteepDownAreas
                c_start = D[1][1]
                c_end = U_end
                # check definition 11
                
                D_max = r[D[1][1]]
                if D_max * ξ_compliment >= r[c_end+1]
                    while r[c_start+1] > r[c_end+1] && c_start < D[1][2]
                        c_start+=1
                    end
                elseif r[c_end+1]*ξ_compliment >= D_max
                    while r[c_end-1] > D_max && c_end > U_start
                        c_end -= 1
                    end
                end

                #c_start, c_end = correctPred(r, pred, order, c_start, c_end)

                if isnothing(c_start)
                    continue
                end

                # criteria 3.a, 1, 2
                if c_end - c_start + 1 >= minpts && c_start <= D[1][2] && c_end >= U_start
                    push!(U_clusters, (c_start, c_end))
                end
            end
            # perm = sortperm([p[2]-p[1] for p in U_clusters])
            append!(SetOfClusters, reverse(U_clusters))#[perm]))# reverse(U_clusters))
        else
            index +=1 
        end

    end
    return SetOfClusters, n
end


function ExtractLabels((Clusters, n))
    labels = fill(-1, n)
    label = 0
    for c in Clusters
        if sum(labels[c[1]:(c[2]+1)] .!= -1)==0
            labels[c[1]:c[2]] .= label
            label +=1
        end
    end
    return labels
end

function filterUpdate!(SSDA, ξ_compliment, mib, r)
    if isnan(mib)
        SSDA = []
    end
    SSDA = SSDA[[mib <= r[sda[1][1]] * ξ_compliment for sda in SSDA]]
    for sda in SSDA
        sda[2] = max(sda[2], mib)
    end
end



@everywhere function centerpoint(data)
    out = Array{Float64}(undef, size(data)[2]+1)
    for i in 1:size(data)[2]
        out[i] = median(data[:,i])
    end
    return out
end