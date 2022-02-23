function [decision_time_Bias_idx,p_choice_Bias] =  sim_bias_dtb(t, drift, acc_thresh, prior_K, K, nsim)
% simulate dtb and return decision time for a given accuracy threshold setting and
% bias belief
%
% Input:
% -t: vector of discretised trial timepoints
% -drift: drift rate
% -acc_thresh: accuracy threshold for choice (at a given coherence level)
% -prior_K: vector of probabilities over bias beliefs
% -K: vector of bias beliefs
% -nsim: number of simulated evidence trajectories
%
% Output:
% - decision_time_Bias_idx: decision timepoint in discretised time
% - p_choice_Bias: choice confidence at decision time given bias belief
%
% written by Alex Skowron (2022)

nt = length(t); % number of timepoints
dt = t(2)-t(1); % time step size

ev = bsxfun(@plus,randn(nsim,nt)*sqrt(dt),drift*dt); % evolution of decision variable sample for each timepoint and trial

cev = cumsum(ev,2); % cumulated sampled evidence

% compute p(correct) unbias case to get acc threshold at a given coh level
p_up_correct_noBias = sum(sign(cev) > 0, 1)./nsim; % probability of correct up, assuming an ideal observer taking into account noisy evidence accumulation (see Balsdon et al., 2018)
p_down_correct_noBias = 1 - p_up_correct_noBias; % probability of correct down

% compute p(correct) bias case for each option
p_up_correct_Bias = (prior_K * K') .* p_up_correct_noBias;
p_down_correct_Bias = (prior_K * (1-K)') .* (1-p_up_correct_noBias);
p_up_correct_Bias = p_up_correct_Bias./(p_up_correct_Bias+p_down_correct_Bias); % normalisation
p_down_correct_Bias = 1 - p_up_correct_Bias;

% get decision time at accuracy threshold
if (prior_K * K') >= 0.5
    p_choice_Bias = p_up_correct_Bias;
elseif (prior_K * K') < 0.5
    p_choice_Bias = p_down_correct_Bias;
end

decision_time_Bias_idx = find(p_choice_Bias >= acc_thresh,1); % decision time in discretised time

if isempty(decision_time_Bias_idx)
    decision_time_Bias_idx = nt; % due to random noise it can sometimes happen that the accuracy threshold is not reached by the end of the response time window. Maybe there is a more elegant solution to this.
end

end