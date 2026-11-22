function out = fhandleFromData(t, time, data)
    % Create a function handle
    out = interp1(time, data, t);
end