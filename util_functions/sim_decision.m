function [decision_time_idx_noBias,decision_var_noBias,decision_time_idx_Bias,decision_var_Bias,max_acc_Bias,max_speed_acc,sub_decision_time_idx,sub_decision_var,sub_acc,p_up_correct_noBias,p_up_correct_Bias,acc_thresh] =  sim_decision(t, drift, acc_thresh_decay, acc_weight,pBias, nsim, ev_bound)
% simulates the evolution of the decision variable (i.e. evidence) accounting for prior belief about the base rate on each
% trial and determines its absorption time based on a subjective accuracy threshold (defined for each coherence level).
%
% Inputs: t = vector of discretised timepoints (e.g. 0:0.0005:2), acc_thresh = subjective accuracy target before responding (speed-accuracy trade-off for a given coherence level), acc_weight = subjective speed-accuracy trade-off (0 = speed maximised, 1 = acc maximised), pBias = probability of up bias, drift = drift rate, nsim = number of simulated evidence
% trajectories
%
% adapted from Zylberberg et al. 2018
%
% written by Alex Skowron (2022)

if isempty(ev_bound)
    % use threshold bound
    acc_thresh_t = 1 - [1:length(t)] .* acc_thresh_decay; % accuracy threshold at each timepoint
end
    

nt = length(t); % number of timepoints
dt = t(2)-t(1); % time step size

ev = bsxfun(@plus,randn(nsim,nt)*sqrt(dt),drift*dt); % evolution of decision variable sample for each timepoint and trial

cev = cumsum(ev,2); % cumulated sampled evidence

% compute p(right & correct) unbias case at acc threshold
p_up_correct_noBias = sum(sign(cev) > 0, 1)./nsim; % probability of correct up bound crossing, assuming an ideal observer taking into account noisy evidence accumulation (see Balsdon et al., 2018)

if isempty(ev_bound)
    decision_time_idx_noBias = find(p_up_correct_noBias >= acc_thresh_t,1); % decision time in discretised time
    
    % better to use threshold decay function that meets 50% accuracy at the
    % end of the response time window.
    if isempty(decision_time_idx_noBias)
        decision_time_idx_noBias = nt; % set decision time to max response time if evidence bound is not reached
    end
    
    acc_thresh = acc_thresh_t(decision_time_idx_noBias);
else
    decision_time_idx_noBias = find(mean(cev,1) >= ev_bound,1);
    
    if isempty(decision_time_idx_noBias)
        decision_time_idx_noBias = nt; % set decision time to max response time if evidence bound is not reached
    end
    
    acc_thresh = p_up_correct_noBias(decision_time_idx_noBias);
end

decision_var_noBias = mean(cev(:,decision_time_idx_noBias)); % cumulated sampled evidence

% compute p(right & correct) bias case at acc threshold (max speed)
p_up_correct_Bias = pBias .* p_up_correct_noBias;
p_down_correct_Bias = (1-pBias) .* (1-p_up_correct_noBias);
p_up_correct_Bias = p_up_correct_Bias./(p_up_correct_Bias+p_down_correct_Bias); % normalisation

decision_time_idx_Bias = find(p_up_correct_Bias >= acc_thresh,1); % decision time in discretised time
decision_var_Bias = mean(cev(:,decision_time_idx_Bias)); % cumulated sampled evidence at a given bias belief keeping p(up=correct) constant
max_speed_acc = p_up_correct_Bias(decision_time_idx_Bias);

% get maximal accuracy bonus achievable based on bias prior
max_acc_Bias = p_up_correct_Bias(decision_time_idx_noBias);

% get decision-time and decision variable for a given subjective speed-accuracy trade-off setting
sub_decision_time_idx = find(p_up_correct_Bias >= acc_thresh + acc_weight*(max_acc_Bias - acc_thresh),1);
sub_decision_var = mean(cev(:,sub_decision_time_idx));
sub_acc = p_up_correct_Bias(sub_decision_time_idx);

% This does not consider knowledge of noisy evidence accumulation by the agent
%     p_up_nobias = normpdf(cev(tpoint),drift*tpoint,sqrt(tpoint)); % probability of an up-bound correct response at time point t
%     p_up_nobias = p_up_nobias/(p_up_nobias+normpdf(cev(tpoint),-drift*tpoint,sqrt(tpoint)))

end