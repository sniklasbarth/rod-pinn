function waveplot(ax,s, t, waveMatrix)
    % Input:
    %   - s: Spatial domain (1D array)
    %   - t: Temporal domain (1D array)
    %   - waveMatrix: Matrix of wave values corresponding to (s, t) points
    
    % Create a heatmap plot of the wave
    [S,T] = meshgrid(s,t);
    if all(size(S) == size(waveMatrix')) 
        waveMatrix = waveMatrix';
    end
    surf(ax,S, T, waveMatrix,...
        'FaceAlpha',0.5,...
        'EdgeColor','none',...
        'EdgeAlpha',0.0);
    c  = colorbar(ax); % Add a colorbar for reference
    title(ax,'Wave Heatmap');
    xlabel(ax,'s in m');
    ylabel(ax,'t in s');
    view(ax,0,90)
    xlim(ax,[min(s),max(s)])
    ylim(ax,[min(t),max(t)])    
end


