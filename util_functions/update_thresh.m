function acc_thresh_decay_post = update_thresh(coh_levels,kappa,nsim,acc_thresh_decay,acc_weight,acc_noBias,decision_time_noBias,t,K, prior_K, plot_process)
% update the slope of the (linear) time-decaying accuracy threshold function given a (new) bias belief
%
% Inputs:
% -coh_levels: stimulus coherence levels (motion strength)
% -kappa: subjective drift rate (signal-to-noise)
% -nsim: number of simulated evidence trajectories
% -acc_thresh_decay: slope of current the accuracy threshold function
% -acc_weight: weight on accuracy maximisation (determined S-A trade-off adjustment)
% -acc_noBias: accuracy level at each stimulus coherence level under no bias
% -decision_time_noBias: decision time at each stimulus coherence level under no bias
% -t: vector of discretised trial timepoints
% -K: vector of bias beliefs
% -prior_K: vector of probabilities for each bias belief
% -plot_process: toggle plotting of accuracy threshold update
%
% Output:
% -acc_thresh_decay_post: updated accuracy threshold slope
%
% written by Alex Skowron (2022)

nt = length(t); % number of timepoints
dt = t(2) - t(1);

decision_time_noBias_idx = decision_time_noBias./dt;

% % update slope of acc threshold decay function
% acc_thresh_pre = 1 - acc_thresh_decay_pre * ntr; % max acc at 0% coherence under previous acc threshold function

% % Learning rate definition of S-A trade-off
% % maybe definition of acc_weight as a learning rate not ideal. Because this
% % assumes subject will always maximise accuracy over time (Just more or less slowly).
% if (prior_K * K') >= 0.5
%     acc_thresh_post = acc_thresh_pre + acc_weight*(p_up_correct_Bias - acc_thresh_pre); % subject specific update towards the new max acc a subject's focus on accuracy over speed (acc_weight = learning rate)
% elseif (prior_K * K') < 0.5
%     acc_thresh_post = acc_thresh_pre + acc_weight*(p_down_correct_Bias - acc_thresh_pre); % subject specific update towards the new max acc a subject's focus on accuracy over speed (acc_weight = learning rate)
% end

% compute optimal threshold adjustment for each coherence level
acc_thresh_coh = zeros(1,length(acc_noBias));
acc_thresh_decay_post_coh = zeros(1,length(acc_noBias));
decision_time_Bias_idx = zeros(1,length(acc_noBias));
p_choice_bias = zeros(length(acc_noBias),nt); % evolving probability of a bias congruent choice over the course of a trial

for c = 1:length(acc_thresh_coh)
    
        % compute p(correct) bias case for each option at unbiased decision time assuming that a trial will be bias congruent
        if (prior_K * K') >= 0.5
            p_up_correct_noBias = 1 - acc_thresh_decay*decision_time_noBias_idx(c);
            p_down_correct_noBias = 1 - p_up_correct_noBias;
        elseif (prior_K * K') < 0.5
            p_down_correct_noBias = 1 - acc_thresh_decay*decision_time_noBias_idx(c);
            p_up_correct_noBias = 1 - p_down_correct_noBias;
        end
        
        p_up_correct_Bias = (prior_K * K') .* p_up_correct_noBias;
        p_down_correct_Bias = (prior_K * (1-K)') .* (1-p_up_correct_noBias);
        p_up_correct_Bias = p_up_correct_Bias./(p_up_correct_Bias+p_down_correct_Bias); % normalisation
        p_down_correct_Bias = 1 - p_up_correct_Bias;
        
    if (prior_K * K') >= 0.5
        
        % compute optimal threshold adjustment for a given coherence level
        acc_thresh_coh(c) = acc_noBias(c) + acc_weight*(p_up_correct_Bias - acc_noBias(c)); % subject specific update towards the new max acc
        
        drift = kappa * coh_levels(c);
        [decision_time_Bias_idx(c) p_choice_bias(c,:)] = sim_bias_dtb(t, drift, acc_thresh_coh(c), prior_K, K, nsim);
        
        if length(decision_time_Bias_idx(c)) ~= 1
           error('error computing bias decision time for threshold update') 
        end
        
        acc_thresh_decay_post_coh(c) = (1-acc_thresh_coh(c))/decision_time_Bias_idx(c);
        
        clear drift
        
    elseif (prior_K * K') < 0.5
        
        % compute optimal threshold adjustment for a given coherence level
        acc_thresh_coh(c) = acc_noBias(c) + acc_weight*(p_down_correct_Bias - acc_noBias(c)); % subject specific update towards the new max acc
        
        drift = kappa * coh_levels(c);
        [decision_time_Bias_idx(c)  p_choice_bias(c,:)] = sim_bias_dtb(t, -drift, acc_thresh_coh(c), prior_K, K, nsim);
        
        if length(decision_time_Bias_idx(c)) ~= 1
           error('error computing bias decision time for threshold update') 
        end
        
        acc_thresh_decay_post_coh(c) = (1-acc_thresh_coh(c))/decision_time_Bias_idx(c);
        
        clear drift
        
    end    
end

% sanity check that all thresholds have been computed correctly
if length(acc_thresh_decay_post_coh) ~= length(coh_levels)
   error('Error updating accuracy threshold function.') 
end

% Evaluate threshold proposals with respect to
% meeting the desired (minimum) accuracy level at each coherence level (accuracy level for a given coherence level may be 
% exceeded given constrains of the linear functional form of the accuracy treshold). 
% Pick proposal that meets (or exceeds) desired accuracy level across all coherence levels.

acc_thresh_decay_post = acc_thresh_decay; % initialise to baseline threshold (due to random variation sometimes no threshold proposal can be found. In this case do not update threshold from baseline.)
acc_coh_post = acc_noBias;

for prop = 1:length(acc_thresh_decay_post_coh) % cycle over threshold (slope) proposals. Should be lowest to highest coherence level!
    
    acc_coh_prop = zeros(1,length(coh_levels));
    
    for coh = 1:length(coh_levels) % cycle over coherence levels
        
        acc_thresh_prop = 1 - acc_thresh_decay_post_coh(prop) .* [1:length(t)]; % threshold proposal
        decision_time_prop_idx = find(p_choice_bias(coh,:) >= acc_thresh_prop,1);
        
        if isempty(decision_time_prop_idx)
            decision_time_prop_idx = nt; % In case accuracy level is never reached pick last timepoint of time window
        end
        
        acc_coh_prop(coh) = p_choice_bias(coh,decision_time_prop_idx);
        
    end
    
    if any(round(acc_coh_prop) < round(acc_thresh_coh)) % due to estimation noise of dtb-based probability trajectories sometimes minimum requirement not reach. Rounding should resolve problem.
        % do not accept proposals that do not meet the (minimum) desired accuracy level
        % for any given coherence level
        continue
        
    elseif sum(acc_coh_prop) < sum(acc_coh_post)
        % do not accept proposal if accuracy across all coherence levels is
        % worse than for the previously accepted proposal
        continue 
        
    else
        % accept proposal
        acc_thresh_decay_post = acc_thresh_decay_post_coh(prop);
    end
end

% if acc_thresh_decay_post == acc_thresh_decay
%    error('No accuracy threshold update accepted.') 
% end

% sanity check that new accuracy threshold has been found.
if isempty(acc_thresh_decay_post)
   error('No accuracy threshold update accepted.') 
end

if plot_process
    
    subplot(2,1,2)
    
    % accuracy threshold no bias (for reference)
    plot(decision_time_noBias_idx,acc_noBias,'g.','MarkerSize',28)
    hold on
    
    time_thresh_base = 1 - acc_thresh_decay .* [1:length(t)];
    plot([1:length(t)],time_thresh_base,'g-','LineWidth',4)
    
    % accuracy threshold for a given bias and S-A tradeoff
    time_thresh_post_coh = 1 - acc_thresh_decay_post_coh' * [1:length(t)];
    plot([1:length(t)]',time_thresh_post_coh','y-','LineWidth',1)
    
    plot(decision_time_Bias_idx,acc_thresh_coh,'r.','MarkerSize',28)
    
    time_thresh_post = 1 - acc_thresh_decay_post .* [1:length(t)];
    plot([1:length(t)],time_thresh_post,'r-','LineWidth',4)
    hold off
    
    ylim([0.5,1])
    xlim([0,length(t)])
    
    title('Accuracy threshold')
    xlabel('time')
    ylabel('accuracy')
    set(gca,'fontsize', 18);
    
    clear time_thresh_base time_thresh_post time_thresh_post_coh
    
end

end