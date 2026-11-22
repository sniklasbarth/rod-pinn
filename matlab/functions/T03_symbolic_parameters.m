%% Define all parameters of rod as symbolic
% in later versions precurvature could be considered
syms radius % in m
A = pi*radius^2; % in m2
I_ax = pi*radius^4/4; % in m4
I_p = 2*I_ax; % in m4

% material
syms E% 240 GPa steel rod
syms nu
G = E/(2*(1+nu)); % poisson's ratio nu = 0.3 --> steel rod

% computation of matrices
% 
GA = G*A;
EI = E*I_ax;
GI = G*I_p;
EA = E*A;

% compliance matrix [check]
% in kg/m and kg*m
computeCompliance = inv(diag([GA;GA;EA;EI;EI;GI]));

syms rho % density in kg/m3

% mass-matrix [check]
% in kg/m and kg*m
computeMass = rho*diag([A,A,A,I_ax, I_ax, I_p]);


