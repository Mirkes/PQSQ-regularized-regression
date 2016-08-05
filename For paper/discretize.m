function [bins] = discretize(X,edges)

[n,bins] = histc(X,edges);