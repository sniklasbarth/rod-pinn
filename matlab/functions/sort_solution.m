function resultStruct = sort_solution(y,modelcase)
%SORT_SOLUTION x1 are velocities and x2 all elastic internal forces
% You can use x2 to recalculate the absolute position and orientation of
% the beam. 
%
% Both quantities can best be ploted as "waves".
%
% Input: 
%   - y solution matrix containing results of all states
%   - modelcase: '1D', '2D', '3D' (different beam cases)
% Output:
%   - x1 velocities
%   - x2 internal elastic forces
%
% IMPORTANT: your matrix y has to contain all (!) states, even those
% constrained by boundary conditions. Otherwise, labeling will be completly
% wrong
%
% y =   [x_1_1(1) x_1_2(1) ... x_1_K(1) ... x_N_1(1) ... x_N_K(T);
%       ...                 ...             ...         ...
%       x_1_1(T) x_1_2(T) ... x_1_K(T) ... x_N_1(T) ... x_N_K(T);   ]
% with T samples, N discrete nodes and K states (fixed by modelcase)

% TODO: modelcase is outdated and cannot be different from '3D'
switch modelcase
    case '1D'
        K = 2;
    case '2D'
        K = 6;
    case '3D'
        K = 12;
    otherwise
        error('Enter suitable modelcase: 1D, 2D or 3D as string!')
end
[T,NK] = size(y); % number of samples and nodes
N = NK/K;
% N has to be an integer
disp(['Data with ', num2str(N), ' nodes and'])
disp([num2str(T), ' samples.'])

% Initialize a struct to store results
resultStruct = struct();

% Reshape the input matrix into cell arrays for each state
for k = 1:K
    % Extract every Kth state
    stateData = y(:, k:K:end);
    
    % Reshape the data into TxN matrix
    reshapedData = reshape(stateData, T, N);
    
    % Store the result in the struct
    resultStruct.(['x_', num2str(k)]) = reshapedData;
end
end
function out = exceptionN
out = ['Oops, something doesnot add up. Did you:',...
     '\n - chose the correct model case for the data?',...
    '\n- parsed all states including boundary values(see doc) ?']
end
