function out = distributed_actuation_pressure(x, w, C, xi0)
%DISTRIBUTED_ACTUATION This function is derived in the manuscript of the
%upcoming T-RO-Paper. 
%
% Author: Max Bartholdt
% Copyright(C)2024, Institut of Mechatronic Systems, Leibniz University
% Hannover
assert(length(x) == 12, 'Only works with 3D model.')


% gamma = C*x(7:end)+xi0.'; % gamma => local strains [v^T u^T]^T;
% op_gamma = operator_L1(gamma);
% out = -op_gamma*w;

out = -(operator_L0(xi0.')*w - operator_L2(w)*C*x(7:end));

end

