function plot_blocks(block_idx)
% plot bias rate block transition points in figure
% Input: vector of base rate block indices
%
% written by Alex Skowron (2022)

    if ~isempty(block_idx) % plot block index for the case of multiple bias conditions
       hold on

       y_ax = ylim;

       for b = 1:length(block_idx)

           y_block = linspace(y_ax(1),y_ax(2),1000);
           x_block = repmat(block_idx(b),[1,length(y_block)]);

           plot(x_block,y_block,'-c','LineWidth',4)

           clear x_block y_block
       end

       hold off
       clear y_ax
    end
end