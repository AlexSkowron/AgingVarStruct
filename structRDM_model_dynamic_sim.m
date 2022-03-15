% simulate Bayesian observer model for struct_RDM task
%
% This script is based on the model reported in Zylberberg et al (Neuron, 2018)
%
% written by Alex Skowron (2022)

clear all

addpath(genpath('./util_functions'))

%% simulate experimental run

ntr = 50; % number of simulated trials per condition
coh_levels = [0 0.05 0.1 0.2 0.4 0.6]; % note: 0 coherence condition must make up a small proportion of trials relative to all other levels. Otherwise the observer cannot learn the bias (or only very slowly)
% coh_levels = sort([-coh_levels 0 coh_levels]);

K_true = [0.2 0.8]; % ground truth base rates of each block

% simulate experimental run
K_tr = [];

for k = 1:length(K_true)
    K_tr = [K_tr binornd(1,K_true(k),1,ntr)]; % add bias condition;
end

K_tr(K_tr == 0) = deal(-1);

coh_tr = coh_levels(randi([1,length(coh_levels)],1,length(K_tr)));
coh_tr = coh_tr .* K_tr;

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

% discrete trial time for dtb
dt = 0.01; % time steps for discrete time points
max_t = 2; % maximum response time
t = 0:dt:(max_t+dt); % discrete trial time points

%% Parameters of simulated subject
% note that the model simulates average subject behaviour

sub_par.K = 0:0.1:1; % base rate belief (assuming subjects dont know the ground truth)
sub_par.w1 = 1; % parameters shaping prior for extreme values (i.e. extreme base rates perceived less probable a priori)
sub_par.w2 = 1;
prior_K_ini = ones(1,length(sub_par.K));
prior_K_ini([1,end])  = deal(sub_par.w1);
prior_K_ini([2,end-1])  = deal(sub_par.w2);
prior_K_ini = prior_K_ini/sum(prior_K_ini); % normalised prior over bias rates

wtype = 1;
% wtype: belief updating model
%   1: use counterfactual confidence (optimal) 
%   2: use confidence to update prior (suboptimal)
%   3: use fixed factor to update prior (suboptimal)

if isequal(wtype,'ignore_conf_nonparW_shrink') || wtype == 3
    fix_like = 0.5; % fixed posterior update for model ignoring (counterfactual-)confidence
end

sub_par.kappa = 2.5; % subject-specific drift rate (signal-to-noise)
drifts = sub_par.kappa * coh_tr; % drift rate for up choice
sub_par.acc_thresh_decay = 0.5/length(t); % linear rate of decay of the acc threshold over time (based on empirical estimates from no bias baseline condition)
sub_par.acc_weight_target = 1; % speed-accuracy trade-off. 0 = maximise speed, 1 = maximise accuracy given prior on bias. This parameter determines how much a subject aims to adjust their accuracy threshold towards the maximal achievable accuracy (using their bias belief) at baseline (no prior) decision time of each respective coherence level. Note: In the current implementation the desired level may be exceeded given the linear functional form of the assumed accuracy threshold.
sub_par.use_belief_weight = 0; % use bias belief confidence (probability) to dynamically change S-A trade-off during learning -- not yet implemented
nsim = 100000; % number of simulated decision variable trajectories used to estimate p(up & correct)

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

sub_par.HR_belief = 0.01; %0.01 % belief about the change frequency in base rates (assumed to be subjective and fixed)

%% saving options

save_results = 0;
SAVE_name = 'sim_switch_matchHR_maxAcc';
SAVE_path = '/Users/skowron/Documents/structRDM_modelling/structRDM_model/sim_results';

%% plotting options

plot_process = 1; % toggle to plot prior and threshold evolution
save_animated_plots = 0;

savefig = 0; % toggle result figures saving

%% Simulate (average) subject behaviour

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
   [~, ~, acc_noBias(ci), decision_time_noBias(ci)] = calc_conf_bound(t, [sub_par.kappa * coh_levels(ci)], sub_par.acc_thresh_decay, prior_K_ini, sub_par.K, nsim);
end

for i = 1:length(coh_tr)  % cycle over trials   
    
    if i==1 % use initial prior at the start
        prior_K = prior_K_ini;
        acc_thresh_decay_bias = sub_par.acc_thresh_decay;
    end
    
    % Compute confidence (p(up = correct | bias)), counterfactual
    % confidence (p(up = correct | unbiased)) and decision time under a
    % trial=specific bias belief and accuracy threshold setting
    
    [~,right_conf_bias,right_conf_nobias,decision_time] = calc_conf_bound(t, drifts(i), acc_thresh_decay_bias, prior_K, sub_par.K, nsim);
    
    % use average choice and RT as model prediction; alternatively
    % could simulate diffusion to bound model for each trial with
    % trial-wise change in bound height

    % updating term of the prior depending on updating model
    if wtype == 1
        update = right_conf_nobias; % use counterfactual confidence (optimal) -- computed above (p(up = correct | unbiased))
    elseif wtype == 2 || isequal(wtype,'use_conf_nonparW_shrink')
        update = right_conf_bias; % scale update by confidence (suboptimal) -- use confidence computed above (p(up = correct | bias))
    elseif wtype == 3 || isequal(wtype,'ignore_conf_nonparW_shrink')
        update = fix_like; % scale update by a fixed factor (suboptimal) -- free parameter
    end
    
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
    
    % update bias belief
    prior_K_prev = prior_K;
    [prior_K] = one_trial_bayesian_update_dynamic(sub_par.K,prior_K,update,sub_par.HR_belief,prior_K_ini); % update the prior for the next trial
    
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
    
    acc_thresh_decay_bias = update_thresh(coh_levels,sub_par.kappa,nsim,sub_par.acc_thresh_decay,acc_weight,acc_noBias,decision_time_noBias,t,sub_par.K, prior_K, plot_process);
    
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

%% plot simulation results
figure

trial_num=1:length(coh_tr);

bias_block_str = regexprep(num2str(K_true),'\s+',', ');

leg = {};

for c = 1:length(coh_levels)
    leg = [leg ['motion strength = ' num2str(coh_levels(c))]];
end

block_idx = find(mod(trial_num,ntr+1) == 0); % condition start points

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
    true_K_tr = [true_K_tr repmat(K_true(k),[1,ntr])];
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

% saving

if save_results
   
    save(fullfile(SAVE_path,[SAVE_name '.mat']),'coh_tr','ntr','K_true','coh_levels','sub_par','model_allsim')
    
end