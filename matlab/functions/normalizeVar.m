function result = normalizeVar(var, m_var, b_var, flag)
    if flag
        % Normalize
        result = var * m_var + b_var;
    else
        % Renormalize/rescale
        result = (var - b_var) * (1 / m_var);
    end
end