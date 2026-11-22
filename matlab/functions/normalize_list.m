function X_scal = normalize_list(X, props_names, m_x, b_x, derivative_flag, no_offset_flag, normalize_flag)
    % Normalizes an array of values where the column's states are defined in 'props_names'
    % derivative_flag - true if the state to normalize or rescale is a derivative towards t
    % no_offset_flag - true if normalization only based on m_x (e.g., when renormalizing ODE losses)
    % normalize_flag - decides weather to rescale (false) or normalize
    % (true)
    
    % Check input type and dimensions
    if isa(X, 'gpuArray') || isnumeric(X)
        if isvector(X)
            X = X(:).'; % Make X a row vector if it is 1-dimensional
        end
        X_scal = X; % Initialize X_scal as a clone of X
    end

    % Normalize or rescale each column
    for i = 1:length(props_names)
        prop_name = props_names{i};
        
        if derivative_flag
            if normalize_flag
                % Normalize with respect to time derivative
                X_scal(:, i) = X(:, i) / m_x.dt * m_x.(prop_name);
            else
                % Rescale with respect to time derivative
                X_scal(:, i) = X(:, i) * m_x.dt / m_x.(prop_name);
            end
        elseif no_offset_flag
            if normalize_flag
                % Normalize without offset
                X_scal(:, i) = X(:, i) * m_x.(prop_name);
            else
                % Rescale without offset
                X_scal(:, i) = X(:, i) / m_x.(prop_name);
            end
        else
            % Use normalizeVar function for normalization with offset
            X_scal(:, i, :) = normalizeVar(X(:, i,:), m_x.(prop_name), b_x.(prop_name), normalize_flag);
        end
    end
end