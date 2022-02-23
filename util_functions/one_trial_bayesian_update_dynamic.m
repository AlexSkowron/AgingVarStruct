function [Ph] = one_trial_bayesian_update_dynamic(K,prior,p_up,HR_belief,prior_K_ini)
% Update bias belief based on trial observation and assumed environmental
% volatility
%
% Input:
% -K: vector of bias/ base rate beliefs (e.g.: [0:0.1:1])
% -prior: probability of each bias
% -p_up: probability that upward motion is correct; used for belief updating (depends on updating model setting)
% -HR_belief: subjective belief about the change rate of bias blocks (hazard rate)
% -prior_K_ini: initial prior over bias beliefs
%
% Output:
% - Ph: Updated prior over bias beliefs (aka posterior)
%
% written by Alex Skowron (2022)

% update bias belief based on trial evidence
pev_h = K.*p_up+(1-K).*(1-p_up); % update for each bias belief level (see Figure 3c in paper)
ph_ev = pev_h.*prior; % apply update to the prior at each bias level
ph_ev = ph_ev/sum(ph_ev); % normalisation so probabilities over belief levels sum to 1

% adjust bias belief given knowledge about environmental volatility
Ph = (1-HR_belief) .* ph_ev + HR_belief .* prior_K_ini;
Ph = Ph/sum(Ph); % normalisation

end

