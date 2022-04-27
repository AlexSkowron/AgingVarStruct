% simulate Bayesian observer model for struct_RDM task
%
% This script is based on the model reported in Zylberberg et al (Neuron, 2018)
%
% written by Alex Skowron (2022)
clear all

model_names = {'sim_taskV1_tinyHR_maxAcc'};

for mod = 1:length(model_names)

addpath(genpath('./util_functions'))

exp_sim_info.seed=90422;
rng(exp_sim_info.seed)

% nWorkers = 4; % number of workers available for parallel processing
% 
% p = gcp('nocreate'); % If no pool, do not create new one.
% if isempty(p)
%     parpool(nWorkers) % start parallel pool
% else
%     delete(gcp('nocreate'))
%     parpool(nWorkers) % start new parallel pool
% end

%% Parameters of simulated subject
% note that the model simulates average subject behaviour

% discrete trial time for dtb
dt = 0.01; % time steps for discrete time points
max_t = 2; % maximum response time
t = 0:dt:(max_t+dt); % discrete trial time points

sub_par.K = 0:0.05:1; % base rate belief (assuming subjects dont know the ground truth)
sub_par.sigma = 10; % parameters shaping prior for extreme values (i.e. extreme base rates perceived less probable a priori); In this implementation parameter reflects the SD of a truncated normal

if strcmp(model_names{mod},'sim_taskV1_smallSigma_maxAcc')
    sub_par.sigma = 0.15;
elseif strcmp(model_names{mod},'sim_taskV1_tinySigma_maxAcc')
    sub_par.sigma = 0.05;
end

prior_K_ini = arrayfun(@(x) normpdf(x, 0.5, sub_par.sigma), sub_par.K);
prior_K_ini = prior_K_ini/sum(prior_K_ini); % normalised prior over bias rates

% prior_K_ini = ones(1,length(sub_par.K));
% prior_K_ini([1,end])  = deal(sub_par.w1);
% prior_K_ini([2,end-1])  = deal(sub_par.w2);

sub_par.wtype = 1;
% sub_par.wtype: belief updating model
%   1: use counterfactual confidence (optimal) 
%   2: use confidence to update prior (suboptimal)
%   3: use fixed factor to update prior (suboptimal)

if strcmp(model_names{mod},'sim_taskV1_confUpdate_maxAcc')
    sub_par.wtype = 2;
end

if strcmp(model_names{mod},'sim_taskV1_confUpdateWeighted_maxAcc')
    sub_par.wtype = 3;
    sub_par.confWeight = 0.3; % optimal update influenced by (suboptimal) confidence
end

if sub_par.wtype == 4
    sub_par.fix_like = 0.5; % fixed posterior update for model ignoring (counterfactual-)confidence
end

sub_par.kappa = 2.5; % subject-specific drift rate (signal-to-noise)
sub_par.acc_thresh_decay = 0.5/length(t); % linear rate of decay of the acc threshold over time (based on empirical estimates from no bias baseline condition)
sub_par.acc_weight_target = 1; % speed-accuracy trade-off. 0 = maximise speed, 1 = maximise accuracy given prior on bias. This parameter determines how much a subject aims to adjust their accuracy threshold towards the maximal achievable accuracy (using their bias belief) at baseline (no prior) decision time of each respective coherence level. Note: In the current implementation the desired level may be exceeded given the linear functional form of the assumed accuracy threshold.
sub_par.use_belief_weight = 0; % use bias belief confidence (probability) to dynamically change S-A trade-off during learning -- not yet implemented
sub_par.nsim = 50000; % number of simulated decision variable trajectories used to estimate p(up & correct); should be a multiple of nWorkers for parallel processing

if strcmp(model_names{mod},'sim_taskV1_matchHR_maxRT')
    sub_par.acc_weight_target = 0;
end

% --Replaced by dynamic accuracy threshold
%
% acc_thresh = 0.7; % subjective response threshold. Subjects commits to a decision when this p(up & correct) threshold is crossed.
% time_thresh = 100; % timepoint when subjective acc threshold is crossed.

% --Replaced by new algorithm to update threshold
%
% if any(coh_levels == 0) % do not consider 0 motion coherence for accuracy threshold updating (S-A trade-off breaks down because of 0 drift).
%     coh_weights = ones(1,length(coh_levels));
%     coh_weights(coh_levels == 0) = 0;
%     coh_weights = coh_weights./(length(coh_weights)-1);
% else
%     coh_weights = ones(1,length(coh_levels))./length(coh_levels); % possible weight of different coh levels on S-A adjustment (e.g. because of unequal trial count)
% end

sub_par.HR_belief = 0.03; %0.03 % belief about the change frequency in base rates (assumed to be subjective and fixed)

if strcmp(model_names{mod},'sim_taskV1_noHR_maxAcc')
    sub_par.HR_belief = 0;
elseif strcmp(model_names{mod},'sim_taskV1_tinyHR_maxAcc')
    sub_par.HR_belief = 0.01;
elseif strcmp(model_names{mod},'sim_taskV1_highHR_maxAcc')
    sub_par.HR_belief = 0.09;
elseif strcmp(model_names{mod},'sim_taskV1_bigHR_maxAcc')
    sub_par.HR_belief = 0.2;
end

%% determine custom coherence levels based on specified difficulty levels given simulated subject parameters

fprintf('determining coherence levels...\n')

% custom difficulty levels for simulated subject
diff_levels = [0.5 0.6 0.7 0.8 0.9];
% coh_levels = [0 0.05 0.1 0.2 0.4 0.6]; % note: 0 coherence condition must make up a small proportion of trials relative to all other levels. Otherwise the observer cannot learn the bias (or only very slowly)
% coh_levels = sort([-coh_levels 0 coh_levels]);

% sample p(up = correct | no bias) across coherence levels
coh_sam = 0:0.01:1;
acc_noBias_sam = zeros(1,length(coh_sam));

for cs = 1:length(coh_sam)
   [~, ~, acc_noBias_sam(cs), ~] = calc_conf_bound(t, [sub_par.kappa * coh_sam(cs)], sub_par.acc_thresh_decay, prior_K_ini, sub_par.K, sub_par.nsim); 
end

% find discretised coherence level closest to desired difficulty levels
coh_levels = zeros(1,length(diff_levels));

for d = 1:length(diff_levels)
    [~,closeIdx] = min(abs(acc_noBias_sam - diff_levels(d)));
    coh_levels(d) = coh_sam(closeIdx);
end

% ensure that 0.5 difficulty level corresponds to 0 coherence (may not hold true simply due to sampling limits of dtb)
if any(diff_levels == 0.5)
    coh_levels(diff_levels == 0.5) = 0;
end

fprintf('coherence levels found.\n')

%% simulate experimental run

fprintf('simulating experimental run...\n')

use_exp_info = 1; % simulate using conditions from actual experiment or generate freely
onload_exp_info = 1; % load simulated experimental run

if use_exp_info
    
    if onload_exp_info
        
        load('sim_results/sim_taskV1_matchHR_maxAcc.mat','exp_sim_info')
        save('temp_exp_info.mat','-struct','exp_sim_info')
        
        load('temp_exp_info.mat')
        delete('temp_exp_info.mat')
        
        rng(exp_sim_info.seed)
           
    else
    
        exp_dir = '/Users/skowron/Documents/struct_RDM2'; % path to experiment repo

        % load condition info
        bias_conds25_info = readtable(fullfile(exp_dir,'bias_cond25_info.csv'));
        bias_conds50_info = readtable(fullfile(exp_dir,'bias_cond50_info.csv'));
        bias_conds75_info = readtable(fullfile(exp_dir,'bias_cond75_info.csv'));

        bias_order_info = readtable(fullfile(exp_dir,'bias_order_info.csv'));

        % pick bias condition order
        bias_order_idx = randi(size(bias_order_info,1),1); % pick or randomise
        K_true = str2num(bias_order_info.bias_order{bias_order_idx});
        K_levels = unique(K_true);

        % save stuff
        exp_sim_info.K_true = K_true;
        exp_sim_info.K_levels = K_levels;
        exp_sim_info.bias_order_idx = bias_order_idx;

        % randomly assign blocks of each bias condition
        bias_block_idx = nan(1,length(K_true));

        for bb = 1:length(K_levels) % cycle over bias conditions

            % randomise bias condition block order
            cond_block_idx = 1:size(eval(['bias_conds' num2str(K_levels(bb) * 100) '_info']),1);
            cond_block_idx = cond_block_idx(randperm(length(cond_block_idx)));

            % assign bias block order
            bias_block_idx(K_true == K_levels(bb)) = cond_block_idx;

        end

        % save stuff
        exp_sim_info.bias_block_idx = bias_block_idx;

        % generate trial-wise coherence levels
        coh_tr = [];
        block_lengths = nan(1,length(K_true));

        for block = 1:length(K_true)

            if K_true(block) == 0.25

                block_coh = arrayfun(@(x) coh_levels(x+1), str2num(bias_conds25_info.trial_coh_idx{bias_block_idx(block)})) .* (str2num(bias_conds25_info.trial_motion_dir{bias_block_idx(block)}) .* -1); % get trial-wise coherence; note reformatting of indices and motion directions due to js implementation
                block_lengths(block) = length(block_coh);

                coh_tr = [coh_tr block_coh]; % concatenate trial-wise coherence vector

                clear block_coh

            elseif K_true(block) == 0.5

                block_coh = arrayfun(@(x) coh_levels(x+1), str2num(bias_conds50_info.trial_coh_idx{bias_block_idx(block)})) .* (str2num(bias_conds50_info.trial_motion_dir{bias_block_idx(block)}) .* -1); % get trial-wise coherence; note reformatting of indices and motion directions due to js implementation
                block_lengths(block) = length(block_coh);

                coh_tr = [coh_tr block_coh]; % concatenate trial-wise coherence vector

                clear block_coh

            elseif K_true(block) == 0.75

                block_coh = arrayfun(@(x) coh_levels(x+1), str2num(bias_conds75_info.trial_coh_idx{bias_block_idx(block)})) .* (str2num(bias_conds75_info.trial_motion_dir{bias_block_idx(block)}) .* -1); % get trial-wise coherence; note reformatting of indices and motion directions due to js implementation
                block_lengths(block) = length(block_coh);

                coh_tr = [coh_tr block_coh]; % concatenate trial-wise coherence vector

                clear block_coh

            else

                error('unknown bias condition.')

            end

        end

        % total number of trials
        ntr = sum(block_lengths);

        % save stuff
        exp_sim_info.coh_tr = coh_tr;
        exp_sim_info.block_lengths = block_lengths;
        exp_sim_info.ntr = ntr;
    
    end
    
else

    ntr = 50; % number of simulated trials per condition

    K_true = [0.2 0.8]; % ground truth base rates of each block

    % simulate bias conditions
    K_tr = [];

    for k = 1:length(K_true)
        K_tr = [K_tr binornd(1,K_true(k),1,ntr)]; % add bias condition;
    end

    K_tr(K_tr == 0) = deal(-1);
    
    coh_tr = coh_levels(randi([1,length(coh_levels)],1,length(K_tr)));
    coh_tr = coh_tr .* K_tr;

end

% save stuff
exp_sim_info.diff_levels = diff_levels;
exp_sim_info.coh_levels = coh_levels;

% compute subject-specific drift rates for each trial
drifts = sub_par.kappa * coh_tr; % drift rate for up choice

fprintf('experimental run simulated.\n')

% Simulating infrequent 0% coherence catch trials
%
% coh0_prop = 0.1; % proportion of trials with 0% coherence (catch trials)
%
% if any(coh_levels == 0) % treat 0 coherence as a special case. since no bias learning can occur on these trials they should only make up a small number of (catch) trials
%     
%     coh_levels_not0 = coh_levels(coh_levels ~= 0);
%     
%     coh_tr = coh_levels_not0(randi([1,length(coh_levels_not0)],1,length(K_tr)));
%     coh_tr = coh_tr .* K_tr;
%     
%     coh0_idx = randi([1,length(coh_tr),1, round(length(coh_tr)*coh0_prop)]);
%     coh_tr(coh0_idx) = deal(0);
%     
% else
%     
%     coh_tr = coh_levels(randi([1,length(coh_levels)],1,length(K_tr)));
%     coh_tr = coh_tr .* K_tr;
%     
% end

%% saving options

save_results = 1;
SAVE_name = model_names{mod};
SAVE_path = '/Users/skowron/Documents/structRDM_modelling/structRDM_model/sim_results';

%% plotting options

plot_process = 1; % toggle to plot prior and threshold evolution
plot_results = 1;
save_animated_plots = 1;

savefig = 0; % toggle result figures saving

%% Simulate (average) subject behaviour

fprintf('Simulating agent model...\n')

if plot_process
   plts = figure;
   set(plts, 'Position',  [400, 200, 600, 800])
   axis tight manual
end

% initialise
model_allsim.like_right   = nan(1,ntr); % updating term based on counterfactual confidence (see Fig 3C in paper)
model_allsim.conf_right   = nan(1,ntr); % choice confidence (i.e. probability of choosing rightward based on prior and likelihood/counterfactual confidence)
model_allsim.decision_time = nan(1,ntr);
model_allsim.posterior_belief = nan(ntr,length(sub_par.K)); % belief distribution over base rates after trial update
model_allsim.prior_belief = nan(ntr,length(sub_par.K)); % % belief distribution over base rates before trial update (redundant)
model_allsim.post_thresh_decay = nan(1,ntr); % accuracy threshold function slope after bias belief update
model_allsim.pre_thresh_decay = nan(1,ntr); % accuracy threshold function slope before bias belief update
model_allsim.acc_weight = nan(1,ntr); % accuracy weight for S-A trade-off. Changes over time if belief uncertainty scaling is assumed

% get baseline accuracy levels for each coherence level (necessary to compute S-A threshold adjustment)
acc_noBias = zeros(1,length(coh_levels));
decision_time_noBias = zeros(1,length(coh_levels));

for ci = 1:length(coh_levels)
   [~, ~, acc_noBias(ci), decision_time_noBias(ci)] = calc_conf_bound(t, [sub_par.kappa * coh_levels(ci)], sub_par.acc_thresh_decay, prior_K_ini, sub_par.K, sub_par.nsim);
end

for i = 1:length(coh_tr)  % cycle over trials
    
    if i==1 % use initial prior at the start
        prior_K = prior_K_ini;
        acc_thresh_decay_bias = sub_par.acc_thresh_decay;
    end
    
    % Compute confidence (p(up = correct | bias)), counterfactual
    % confidence (p(up = correct | unbiased)) and decision time under a
    % trial=specific bias belief and accuracy threshold setting
    
    [~,right_conf_bias,right_conf_nobias,decision_time] = calc_conf_bound(t, drifts(i), acc_thresh_decay_bias, prior_K, sub_par.K, sub_par.nsim);
    
    % use average choice and RT as model prediction; alternatively
    % could simulate diffusion to bound model for each trial with
    % trial-wise change in bound height

    % updating term of the prior depending on updating model
    if sub_par.wtype == 1
        update = right_conf_nobias; % use counterfactual confidence (optimal) -- computed above (p(up = correct | unbiased))
    elseif sub_par.wtype == 2
        update = right_conf_bias; % scale update by confidence (suboptimal) -- use confidence computed above (p(up = correct | bias))
    elseif sub_par.wtype == 3
        update = (1-sub_par.confWeight)*right_conf_nobias + sub_par.confWeight*right_conf_bias; % optimal counterfactual confidence update corrupted by suboptimal confidence
    elseif sub_par.wtype == 4
        update = fix_like; % scale update by a fixed factor (suboptimal) -- free parameter
    end
    
    % update bias belief
    prior_K_prev = prior_K;
    [prior_K] = one_trial_bayesian_update_dynamic(sub_par.K,prior_K,update,sub_par.HR_belief,prior_K_ini); % update the prior for the next trial
    
   % plot bias belief distribution
    if plot_process
        
        subplot(2,1,1)
        
        plot(sub_par.K,prior_K);
        y_ax = ylim;
        ylim([0,y_ax(2)]);
        title(['Trial #' num2str(i) newline 'Bias belief'])
        xlabel('base rate up','LineWidth',4)
        ylabel('probability')
        set(gca,'fontsize', 22);
        
        clear y_ax
    end
    
    % model S-A trade-off change over time because of learning
    if sub_par.use_belief_weight % not yet implemented
        belief_unc = sum((sub_par.K - (prior_K * sub_par.K')).^2 .* prior_K); % uncertainty in bias belief = prior variance
        acc_weight = belief_unc * (1-sub_par.acc_weight_target) + sub_par.acc_weight_target; % acc_weight scales with bias belief confidence
        %acc_weight = (1 - abs(prior_K(2)-prior_K(1)))*(1-sub_par.acc_weight_target) + sub_par.acc_weight_target; % acc_weight scales with bias belief confidence
    else
        acc_weight = sub_par.acc_weight_target;
    end
    
    % update accuracy threshold (slope)
    acc_thresh_decay_bias_prev=acc_thresh_decay_bias;
    
    acc_thresh_decay_bias = update_thresh(coh_levels,sub_par.kappa,sub_par.nsim,sub_par.acc_thresh_decay,acc_weight,acc_noBias,decision_time_noBias,t,sub_par.K, prior_K, plot_process);
    
    % save stuff
    %model_allsim.choice(i) = m_choice; % simulated choice
    model_allsim.like_right(i)   = right_conf_nobias; % updating term based on counterfactual confidence (see Fig 3C in paper)
    model_allsim.conf_right(i)   = right_conf_bias; % simulated choice confidence (including prior expectation)
    model_allsim.decision_time(i) = decision_time;
    model_allsim.posterior_belief(i,:) = prior_K; %save the full posterior of the updated prior
    model_allsim.prior_belief(i,:) = prior_K_prev;%overkill
    model_allsim.post_thresh_decay(i) = acc_thresh_decay_bias; % accuracy threshold function slope after bias belief update
    model_allsim.pre_thresh_decay(i) = acc_thresh_decay_bias_prev; % accuracy threshold function slope before bias belief update
    model_allsim.acc_weight(i) = acc_weight;
    
    if plot_process    
       drawnow % update all plots
       
       if save_animated_plots
          
            % Capture the plot as an image 
            frame = getframe(plts); 
            im = frame2im(frame); 
            [imind,cm] = rgb2ind(im,256);
            
            % Write to the GIF File 
            if i == 1 
                imwrite(imind,cm,fullfile('sim_animations',[SAVE_name '_process.gif']),'gif', 'Loopcount',inf); 
            else 
                imwrite(imind,cm,fullfile('sim_animations',[SAVE_name '_process.gif']),'gif','WriteMode','append'); 
            end 
           
       end
    end
    
end

fprintf('Agent model simulated.\n')

%% plot simulation results
if plot_results

figure

trial_num=1:length(coh_tr);

bias_block_str = regexprep(num2str(K_true),'\s+',', ');

leg = {};

for c = 1:length(coh_levels)
    leg = [leg ['motion strength = ' num2str(coh_levels(c))]];
end

if use_exp_info
    block_idx = cumsum(block_lengths)+1; % block condition start points
else
    block_idx = find(mod(trial_num,ntr+1) == 0); % block condition start points
end

% p(choice = correct | unbiased belief)
like_choice = model_allsim.like_right;
like_choice(sign(coh_tr) < 0) = 1-like_choice(sign(coh_tr) < 0);

for c = 1:length(coh_levels)
    plot(trial_num(abs(coh_tr)==coh_levels(c)),like_choice(abs(coh_tr)==coh_levels(c)),'o--','LineWidth',4)
    hold on
end
hold off

plot_blocks(block_idx); % plot block markers

title(['Simulated bias blocks: ' bias_block_str])
xlabel('trial')
ylabel('p(choice = correct | unbiased belief)')
legend(leg);
set(gca,'fontsize', 22);

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,['figures_dyn/' SAVE_name '_acc_nobias.png'])
end

pause

% p(choice = correct | bias belief) for non-zero coherence trials

conf_choice = model_allsim.conf_right;

conf_choice(coh_tr < 0) = 1-conf_choice(coh_tr < 0); % p(choice = correct)

plot(trial_num(coh_tr > 0),conf_choice(coh_tr > 0),'bo--','LineWidth',4)
hold on

plot(trial_num(coh_tr < 0),conf_choice(coh_tr < 0),'ro--','LineWidth',4)
hold off

plot_blocks(block_idx); % plot block markers

title(['Simulated bias blocks: ' bias_block_str newline 'motion strength > 0'])
xlabel('trial')
legend({'up motion trial' 'down motion trial'})
ylabel('p(choice = correct | bias belief)')

set(gca,'fontsize', 22);

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,['figures_dyn/' SAVE_name '_acc_bias.png'])
end

pause

% p(choice = block bias congruent | bias belief) zero-coherence level

true_K_tr = [];
for k = 1:length(K_true)
    
    if use_exp_info
        true_K_tr = [true_K_tr repmat(K_true(k),[1,block_lengths(k)])];
    else
        true_K_tr = [true_K_tr repmat(K_true(k),[1,ntr])];
    end
end

conf_choice((true_K_tr < 0.5) & (coh_tr == 0)) = 1-conf_choice((true_K_tr < 0.5) & (coh_tr == 0)); % p(choice = block bias congruent)

plot(trial_num(coh_tr == 0),conf_choice(coh_tr == 0),'bo--','LineWidth',4)

plot_blocks(block_idx); % plot block markers

title(['Simulated bias blocks: ' bias_block_str newline 'motion strength = 0'])
xlabel('trial')
ylabel('p(choice = block bias congruent | bias belief)')

set(gca,'fontsize', 22);

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,['figures_dyn/' SAVE_name '_acc_mot0.png'])
end

pause

% decision time

for c = 1:length(coh_levels)
    plot(trial_num(abs(coh_tr)==coh_levels(c)),model_allsim.decision_time(abs(coh_tr)==coh_levels(c)),'o--','LineWidth',4)
    hold on
end

plot_blocks(block_idx); % plot block markers

hold off

title(['Simulated bias blocks: ' bias_block_str])
xlabel('trial')
ylabel('decision time')
legend(leg);
set(gca,'fontsize', 22);

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,['figures_dyn/' SAVE_name '_dt.png'])
end

pause

% accuracy threshold decay rate

plot(trial_num,model_allsim.post_thresh_decay,'r-','LineWidth',4)

plot_blocks(block_idx); % plot block markers

title(['Simulated bias blocks: ' bias_block_str])
xlabel('trial')
ylabel('accuracy threshold decay')
set(gca,'fontsize', 22);

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,['figures_dyn/' SAVE_name '_accThreshDecay.png'])
end

pause

% bias belief
bias_belief_mean = model_allsim.prior_belief * sub_par.K';
bias_belief_var = sum((repmat(sub_par.K,[length(trial_num),1]) - bias_belief_mean).^2 .* model_allsim.prior_belief,2);

plot(bias_belief_mean,'LineWidth',4)

plot_blocks(block_idx); % plot block markers

title(['Simulated bias blocks: ' bias_block_str])
xlabel('trial')
ylabel('bias belief')
set(gca,'fontsize', 22);

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,['figures_dyn/' SAVE_name '_biasBelief.png'])
end

pause

plot(bias_belief_var,'LineWidth',4)

plot_blocks(block_idx); % plot block markers

title(['Simulated bias blocks: ' bias_block_str])
xlabel('trial')
ylabel('bias belief uncertainty')
set(gca,'fontsize', 22);

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,['figures_dyn/' SAVE_name '_biasBelief_unc.png'])
end

end

%% saving

if save_results
   
    save(fullfile(SAVE_path,[SAVE_name '.mat']),'exp_sim_info','sub_par','model_allsim')
    
end

close all

end