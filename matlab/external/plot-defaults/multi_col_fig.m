function ax = multi_col_fig(fig, cols, rows, tile_size)
%MULTI_COL_FIG Figure is configured to
% 
t = tiledlayout(cols,rows);

for itile = 1:size(tile_size,1)
    try
       ax(itile) = nexttile(tile_size(itile,:));
    catch except
       throw(except)
    end
end
end
