function [x_next] = dd_pinn_forward(weights,biases,x,xdim,n_funs)
%PINN_FORWARD Summary of this function goes here
%   Detailed explanation goes here
    x_in = x(:, 1:xdim);
    t_in = x(:,end)+1;
    x0 = x(:,1:end-1);
    x = x0;
    % Forward pass of NN
    n_layers=length(weights);
    for i = 1:n_layers
        % x=x*double(cell2mat(weights(i)))'+double(cell2mat(biases(i)))';
        x=x*weights{i}+biases{i};
        if i<n_layers
            x = geluForward(x);
        end
    end

    % extract Ansatz-function coefficients
    a = x(:,1:(n_funs*xdim));
    b = x(:,(n_funs*xdim+1):(2*n_funs*xdim));
    c = x(:,(2*n_funs*xdim+1):(3*n_funs*xdim)); 
    d = x(:,(3*n_funs*xdim+1):(4*n_funs*xdim));  

    % Calculation of DD-PINN g function
    g_out = a.*(sin(b.*t_in+c).*exp(-d.*t_in)-sin(c));

    % Calculate output of function g
    if isa(g_out,"casadi.MX")
        g_out_sum = sum(reshape(g_out,n_funs, xdim),1);
    else
        g_out_sum = sum(reshape(g_out,size(t_in,1),n_funs, xdim),2);
        if ndims(g_out_sum)==3
            g_out_sum = squeeze(g_out_sum);
            if size(g_out_sum,2)==1
                g_out_sum = g_out_sum';
            end
        end
    end
    x_next = x_in + g_out_sum;
end

function Z = geluForward(X)
% geluForward   Gaussian Error Linear Unit (GELU) activation
Z = (X./2) .* (1 + erf(X ./ sqrt(2)));
end