function xi_= pls(xi,L,Nj)
%PLS is the piecewise-linear strain interpolation used for 
xi = reshape(xi,6,[]);
N = size(xi,2);
ds = L/(N-1);
d = ds/(Nj-1);
xi_ = [];
for i = 1:N-1
    m = (xi(:,i+1) - xi(:,i))/ds;
    b = xi(:,i);
    for ii = 1:Nj-1
        x = (ii-1)*d;
        xi_ = [xi_,m * x + b];
    end
end
xi_ = [xi_,xi(:,end)];
end


