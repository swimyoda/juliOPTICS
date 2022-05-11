using PyCall
using BenchmarkTools
using Plots
using Distributed
using Tables
using CSV
using DataFrames
@everywhere include("juliOPTICS.jl")
clusters = pyimport("sklearn.cluster")

pworkers = 9
if nworkers()<pworkers
    addprocs(pworkers-nworkers())
else
    while nworkers() > pworkers
        rmprocs(nworkers())
    end
end

@everywhere using SharedArrays
@everywhere using Distances
@everywhere minpts = 20
@everywhere epsi = 0.1
@everywhere xi = 0.1
@everywhere dist = euclidean



#
# Reachability plot experiment
#

n = 3000
data1 = [[[4, -1] + 0.1 * randn(2) for i in 1:n/3];
         [[3, -2] + 1.6 * randn(2) for i in 1:n/3];
         [[5, 6] + 2 * randn(2) for i in 1:n/3] ]
data = mapreduce(permutedims, vcat, data1)

order, reachability_j = OPTICS(data, epsi, minpts, euclidean, xi)
reachability_j = reachability_j[order]
t = clusters.OPTICS(min_samples=minpts, xi=xi, max_eps=epsi, metric="euclidean", predecessor_correction=false).fit(data)
reachability_p = t.reachability_[t.ordering_.+1]
plot(reachability_j, labels="juliOPTICS", ylab="Reachability Distance", title="Computed Reachability Plots")
plot!(reachability_p, labels="sklearn.Cluster.OPTICS")


#
# Timing experiment
#

for i in 3 .*10 .^(2:4)
    n=i
    data1 = [[[4, -1] + 0.1 * randn(2) for i in 1:n/3];
                [[3, -2] + 1.6 * randn(2) for i in 1:n/3];
                [[5, 6] + 2 * randn(2) for i in 1:n/3] ]
    data = mapreduce(permutedims, vcat, data1)
    println(i)
    println("python")
    @btime clusters.OPTICS(min_samples=minpts, xi=xi, max_eps=epsi, metric="euclidean", predecessor_correction=false).fit_predict(data);
    println("serial")
    @btime labels = ExtractLabels(ExtractClustering(OPTICS(data, epsi, minpts, euclidean, xi)));
    println("faSTICS")
    @btime faSTICS(data, minpts, xi, epsi, pworkers);
end


#
#   Plotting timing experiment
#
 file = CSV.File("C:\\Users\\gallag00n\\Documents\\MIT Courses\\Spring 2022\\18.337\\project\\timetrials.csv")
time = DataFrame(file)
time.jspeedup = time.python ./ time.serial
time.pspeedup = time.python ./ time.parallel


plot(time.speed[time.data .==300], time.pspeedup[time.data .==300], color=:red, yaxis=:log, labels = "faSTICS, n=300")

plot!(time.speed[time.data .==30000], time.jspeedup[time.data .==30000], xaxis=:log, color=:blue, yaxis=:log, labels = "juliOPTICS, n=30,000", legend=true)

plot!(time.speed[time.data .==3000], time.pspeedup[time.data .==3000], xaxis=:log, color=:green, yaxis=:log, labels = "faSTICS, n=3,000")

plot!(time.speed[time.data .==30000], time.pspeedup[time.data .==30000], title="Speedup compared to SKLearn.Cluster", xlab="Number of threads available",ylab="Speedup relative python",xaxis=:log, color=:purple, yaxis=:log, labels = "faSTICS, n=30,000", legend=:bottomright)





#
#   Clustering experiment
#

n = 3000
data1 = [[[4, -1] + 0.1 * randn(2) for i in 1:n/3];
[[3, -2] + 1.6 * randn(2) for i in 1:n/3];
[[5, 6] + 2 * randn(2) for i in 1:n/3] ]
data = mapreduce(permutedims, vcat, data1)

labels = faSTICS(data, minpts, epsi, xi, pworkers);
labels[labels.==-2].=-1
c = labels
x = data[:,1]
y = data[:,2]

b1 = scatter(x[c.==-1], y[c.==-1], labels="Noise",title=string("faSTICS Clustering Results: p=",pworkers," threads"))
for col in unique(c[c.!=-1])
    scatter!(b1, x[c.==col], y[c.==col])#, color=:orange)
    println(col)
end
plot(b1, legend=true)#, legend=false, title="faSTICS Clustering Results: p=8 threads")

labels2 = clusters.OPTICS(min_samples=minpts, xi=xi, max_eps=epsi, metric="euclidean", predecessor_correction=false).fit_predict(data);
 
x1 = x
y1 = y
c2 = labels2
b = scatter(x1[c2.==-1], y1[c2.==-1],labels="Noise", title="SKLearn.Cluster.OPTICS Clustering Results")
for col in unique(c2[c2.!=-1])
    scatter!(b, x1[c2.==col], y1[c2.==col])
end
plot(b)
plot(b1, b, layout=@layout([b1;b]), legend=false)








