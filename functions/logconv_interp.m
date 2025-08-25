function [x_opt, cost] = logconv_interp(x_anchor_opt, corner_weights_selected, loss_matrix_max)
    % x_anchor_opt = max(x_anchor_opt, 0); 
    [nr_loc, nr_perturb] = size(loss_matrix_max); 
    for k = 1:1:nr_perturb
        for i = 1:1:nr_loc
            x_opt(i,k) = prod(x_anchor_opt(:, k)'.^corner_weights_selected(i, :)'); 
        end
    end
    for i = 1:1:nr_loc
        if isnan(sum(x_opt(i, :))) == 0
            x_opt(i, :) = x_opt(i, :)/sum(x_opt(i, :));
        else 
            x_opt(i, :) = 0;
        end
    end
    cost = sum(sum(x_opt.*loss_matrix_max))/size(loss_matrix_max, 1);
end