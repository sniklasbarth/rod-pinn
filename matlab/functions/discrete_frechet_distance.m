function d = discrete_frechet_distance(P, Q)
%DISCRETE_FRECHET_DISTANCE Discrete Frechet distance between the polygonal
% curves P (n x dim) and Q (m x dim) (Eiter and Mannila, 1994).
n = size(P,1);
m = size(Q,1);
D = sqrt(max(sum(P.^2,2) + sum(Q.^2,2)' - 2*(P*Q'), 0)); % pairwise distances
ca = zeros(n,m);
ca(1,1) = D(1,1);
for i = 2:n
    ca(i,1) = max(ca(i-1,1), D(i,1));
end
for j = 2:m
    ca(1,j) = max(ca(1,j-1), D(1,j));
end
for i = 2:n
    for j = 2:m
        ca(i,j) = max(min([ca(i-1,j), ca(i-1,j-1), ca(i,j-1)]), D(i,j));
    end
end
d = ca(n,m);
end
