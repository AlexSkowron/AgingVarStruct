function [bound_ev,conf,like,decision_time] =  calc_conf_bound(t, drift, acc_thresh_decay, prior_K, K, nsim)
% simulates the evolution of the decision variable (i.e. evidence) accounting for prior belief about the base rate on each
% trial and determines its absorption time based on a subjective accuracy threshold.
%
% Inputs: 
% t = vector of discretised timepoints (e.g. 0:0.0005:2)
% drift = drift rate
% acc_thresh_decay = slope of the time-dynamic accuracy threshold
% prior_K = vector of probabilities for each bias belief
% K = vector of (subjective) bias/ base rate beliefs
% nsim = number of simulated evidence trajectories
% trajectories
%
% Outputs: 
% bound_ev = subject trial boundary height (accumulated evidence) considering bias belief
% conf = subject confidence at decision time considering bias belief
% like = counterfactual confidence/likelihood at decision time assuming no bias belief
%
% written by Alex Skowron (2022)

% use threshold bound
acc_thresh_t = 1 - acc_thresh_decay .* [1:length(t)]; % accuracy threshold at each timepoint
    

nt = length(t); % number of timepoints
dt = t(2)-t(1); % time step size

ev = bsxfun(@plus,randn(nsim,nt)*sqrt(dt),drift*dt); % evolution of decision variable sample for each timepoint and trial

cev = cumsum(ev,2); % cumulated sampled evidence

% compute p(right & correct) unbias case to get acc threshold at a given coh level 
% - this could probably be done before once for each coherence level before running the
% simulation and then passed to this function rather than recomputing every
% time.

p_up_correct_noBias = sum(sign(cev) > 0, 1)./nsim; % probability of correct up, assuming an ideal observer taking into account noisy evidence accumulation (see Balsdon et al., 2018)
p_down_correct_noBias = 1- p_up_correct_noBias; % probability of correct down

% compute accuracy threshold for unbiased condition at a given coh level (baseline)

% if sign(drift) >= 0
%     decision_time_idx_noBias = find(p_up_correct_noBias >= acc_thresh_t,1); % decision time in discretised time
% elseif sign(drift) < 0
%     decision_time_idx_noBias = find(p_down_correct_noBias >= acc_thresh_t,1);
% end
% 
% %more elegant to use threshold decay function that always meets 50%
% %accuracy at response time limit
% if isempty(decision_time_idx_noBias)
%     decision_time_idx_noBias = nt; % set decision time to max response time if acc threshold is not reached
% end

% acc_thresh_noBias = acc_thresh_t(decision_time_idx_noBias); % accuracy threshold for correct choice option

% decision_var_noBias = mean(cev(:,decision_time_idx_noBias)); % cumulated sampled evidence

% compute p(correct) bias case for each option
p_up_correct_Bias = (prior_K * K') .* p_up_correct_noBias;
p_down_correct_Bias = (prior_K * (1-K)') .* (1-p_up_correct_noBias);
p_up_correct_Bias = p_up_correct_Bias./(p_up_correct_Bias+p_down_correct_Bias); % normalisation
p_down_correct_Bias = 1 - p_up_correct_Bias;

% get maximal accuracy bonus achievable on trials with bias consistent
% motion direction

if (prior_K * K') >= 0.5
    
    % hypothetical p(up = correct) belief if motion on this trial were bias congruent
    if sign(drift) < 0
        p_up_correct_Bias_belief = (prior_K * K') .* (1-p_up_correct_noBias);
        p_down_correct_Bias_belief = (prior_K * (1-K)') .* p_up_correct_noBias;
        p_up_correct_Bias_belief = p_up_correct_Bias_belief./(p_up_correct_Bias_belief+p_down_correct_Bias_belief); % normalisation
        p_down_correct_Bias_belief = 1 - p_up_correct_Bias_belief;
    else
        p_up_correct_Bias_belief = p_up_correct_Bias;
    end

elseif (prior_K * K') < 0.5
    
    % hypothetical p(down = correct) belief if motion on this trial were bias congruent
    if sign(drift) > 0
        p_down_correct_Bias_belief = (prior_K * (1-K)') .* (1-p_down_correct_noBias);
        p_up_correct_Bias_belief = (prior_K * K') .* p_down_correct_noBias;
        p_down_correct_Bias_belief = p_down_correct_Bias_belief./(p_down_correct_Bias_belief+p_up_correct_Bias_belief); % normalisation
        p_up_correct_Bias_belief = 1 - p_down_correct_Bias_belief;
    else
        p_down_correct_Bias_belief = p_down_correct_Bias;
    end
    
%     max_acc_Bias = p_down_correct_Bias_belief(decision_time_idx_noBias);
end
% 
% % sanity check
% if max_acc_Bias < acc_thresh_noBias
%     error('Error computing S-A trade-off.')
% end
% 
% % get evidence bound height (i.e. accumulated evidence) for a subject's speed-accuracy trade-off setting
% acc_thresh_Bias = acc_thresh_noBias + acc_weight*(max_acc_Bias - acc_thresh_noBias);

if (prior_K * K') >= 0.5
    sub_decision_time_idx = find(p_up_correct_Bias_belief >= acc_thresh_t,1);
elseif (prior_K * K') < 0.5
    sub_decision_time_idx = find(p_down_correct_Bias_belief >= acc_thresh_t,1);
end

if isempty(sub_decision_time_idx)
    sub_decision_time_idx = nt; % due to random noise it can sometimes happen that the accuracy threshold is not reached by the end of the response time window. Maybe there is a more elegant solution to this.
end

bound_ev = mean(cev(:,sub_decision_time_idx)); % bound height (accumulated evidence)
conf = p_up_correct_Bias(sub_decision_time_idx); % confidence at decision time(i.e. p(up = correct | bias))
like = p_up_correct_noBias(sub_decision_time_idx); % counterfactual confidence / likelihood at decision time (i.e. p(up = correct | bias = 0.5))
decision_time = sub_decision_time_idx*dt;

% sanity check
if isempty(bound_ev) || isempty(conf) || isempty(like) || isempty(decision_time)
    error('Error evaluating decision model.')
end

end