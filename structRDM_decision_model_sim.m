% simulate decision process for a given bias prior, motion coherence level (i.e. drift rate), and subjective
% speed-accuracy trade-off

addpath(genpath('./util_functions'));

%% set parameters for simulation

dt = 0.01;
max_t = 2;
t = 0:dt:(max_t+dt); % discrete trial time points
kappa = 4; % subject-specific drift rate (signal-to-noise)
coh = 0.2; % stimulus-specific drift rate
drift = kappa * coh; % drift rate for up choice
% acc_thresh = 0.7; % subjective response threshold. Subjects commits to a decision when this p(up & correct) threshold is crossed.
% time_thresh = 100; % timepoint when subjective acc threshold is crossed.
acc_thresh_decay = 0.5/length(t); % linear rate of decay of the acc threshold over time (when fitting the model this would be a free parameter that determines acc thresh and decision time)
acc_weight = 0.5; % speed-accuracy trade-off. 0 = maximise speed, 1 = maximise accuracy given prior on bias.
pBias = 0.6; % probability of an up motion direction bias (belief). 0.5 = no bias
nsim = 10000; % number of simulated decision variable trajectories used to estimate p(up & correct)

savefig = 1;

%% simulate example decision process

[decision_time_idx_noBias,...
    decision_var_noBias,...
    decision_time_idx_Bias,...
    decision_var_Bias,...
    max_acc_Bias,...
    max_speed_acc,...
    sub_decision_time_idx,...
    sub_decision_var,...
    sub_acc,...
    p_up_correct_noBias,...
    p_up_correct_Bias,...
    acc_thresh] = sim_decision(t, drift, acc_thresh_decay, acc_weight,pBias, nsim,[]);

% plot results
figure
hold on

plot(1:length(t),repmat(acc_thresh,[1,length(t)]),'k--')
plot(1:length(t),repmat(max_acc_Bias,[1,length(t)]),'k--')

p(1)=plot(p_up_correct_noBias,'r-','LineWidth',4);
xlabel('time bin')
ylabel('p(up = correct)')
p(2)=plot(p_up_correct_Bias,'b-','LineWidth',4);

plot(repmat(decision_time_idx_noBias,[1,51]),[0.5:0.01:1],'g-','LineWidth',4)
plot(repmat(decision_time_idx_Bias,[1,51]),[0.5:0.01:1],'g-','LineWidth',4)

plot(repmat(sub_decision_time_idx,[1,51]),[0.5:0.01:1],'c-','LineWidth',4)

xlim([1,length(t)])
title('Speed-Accuracy adjustment')
legend([p(1) p(2)],{'no bias' 'bias'})
set(gca,'fontsize', 22);

hold off

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,'figures_decision/acc_thresh_example_1Level.png')
end

pause

close all

%% visualise decaying acc threshold for 2 cohernce levels
coh_sim = [0.3 0.1];

linetype = {'-','--'}; % linetype for plotting each coh level

% plot
figure
hold on

acc_thresh_decay_maxAcc = zeros(1,2);

for c = 1:length(coh_sim)
    
    drift = kappa * coh_sim(c);
    
    [decision_time_idx_noBias,...
        decision_var_noBias,...
        decision_time_idx_Bias,...
        decision_var_Bias,...
        max_acc_Bias,...
        max_speed_acc,...
        sub_decision_time_idx,...
        sub_decision_var,...
        sub_acc,...
        p_up_correct_noBias,...
        p_up_correct_Bias,...
        acc_thresh] = sim_decision(t, drift, acc_thresh_decay, acc_weight,pBias, nsim,[]);
    
    acc_thresh_decay_maxAcc(c) = (1-max_acc_Bias)/decision_time_idx_noBias; % threshold proposal for accuracy maximising agent
    

%     plot(1:length(t),repmat(acc_thresh,[1,length(t)]),'k--')
%     plot(1:length(t),repmat(max_acc_Bias,[1,length(t)]),'k--')

    plot(p_up_correct_noBias,['r' linetype{c}],'LineWidth',2);
    plot(p_up_correct_Bias,['b' linetype{c}],'LineWidth',2);

    plot(repmat(decision_time_idx_noBias,[1,51]),[0.5:0.01:1],'k--','LineWidth',1)
%     plot(repmat(decision_time_idx_Bias,[1,51]),[0.5:0.01:1],'g-','LineWidth',4)
% 
%      plot(repmat(sub_decision_time_idx,[1,51]),[0.5:0.01:1],'c-','LineWidth',4)

end

% plot baseline acc threshold
acc_thresh_t = 1 - [1:length(t)] .* acc_thresh_decay; % accuracy threshold at each timepoint
plot(1:length(t),acc_thresh_t,'g-','LineWidth',4)

% plot possible acc threshold adjustment
acc_thresh_t_adj = 1 - [1:length(t)] .* acc_thresh_decay_maxAcc(1);
plot(1:length(t),acc_thresh_t_adj,'y-','LineWidth',4)

xlim([1,length(t)])
xlabel('time bin')
ylabel('p(up = correct)')
title('Decaying accuracy threshold')
%legend({'no bias + high motion strength' 'bias + high motion strength' 'no bias + low motion strength' 'bias + low motion strength' 'no bias Acc threshold' 'bias Acc threshold'})
set(gca,'fontsize', 22);

hold off

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,'figures_decision/acc_thresh_example_2Level.png')
end

pause


%% simulate decision process across bias levels

bias_levels = 0.5:0.001:1;

decision_var_noBias_all = zeros(1,length(bias_levels));
decision_var_Bias_all = zeros(1,length(bias_levels));
sub_decision_var_all = zeros(1,length(bias_levels));
max_acc_Bias_all = zeros(1,length(bias_levels));
sub_acc_all = zeros(1,length(bias_levels));
max_speed_acc_all = zeros(1,length(bias_levels));
decision_time_idx_noBias_all = zeros(1,length(bias_levels));
decision_time_idx_Bias_all = zeros(1,length(bias_levels));
sub_decision_time_idx_all = zeros(1,length(bias_levels));

for b = 1:length(bias_levels)

[decision_time_idx_noBias,...
    decision_var_noBias,...
    decision_time_idx_Bias,...
    decision_var_Bias,...
    max_acc_Bias,...
    max_speed_acc,...
    sub_decision_time_idx,...
    sub_decision_var,...
    sub_acc,...
    p_up_correct_noBias,...
    p_up_correct_Bias,...
    acc_thresh] = sim_decision(t, drift, acc_thresh_decay, acc_weight,bias_levels(b), nsim,[]);
    
    % collect output
    sub_decision_var_all(b) = sub_decision_var;
    decision_var_noBias_all(b) = decision_var_noBias;
    decision_var_Bias_all(b) = decision_var_Bias;
    max_acc_Bias_all(b) = max_acc_Bias;
    sub_acc_all(b) = sub_acc;
    max_speed_acc_all(b) = max_speed_acc;
    decision_time_idx_noBias_all(b) = decision_time_idx_noBias;
    decision_time_idx_Bias_all(b) = decision_time_idx_Bias;
    sub_decision_time_idx_all(b) = sub_decision_time_idx;
end

% look at change in boundary adjustment on the accumulated evidence for a
% given speed-accuracy trade-off
plot(bias_levels,decision_var_noBias_all,'r-','LineWidth',4);
hold on
plot(bias_levels,decision_var_Bias_all,'b-','LineWidth',4);
plot(bias_levels,sub_decision_var_all,'c-','LineWidth',4);

lims=ylim;
line=linspace(lims(1),lims(2),100);
plot(repmat(acc_thresh,[1,length(line)]),line,'m-','LineWidth',4);

xlabel('bias')
ylabel('decision variable')
legend({'max accuracy' 'max speed' 'simulated trade-off' 'Acc threshold'})
set(gca,'fontsize', 22);
hold off

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,'figures_decision/bias_dv_example_1Level.png')
end

pause

% look at change in accuracy for a given speed-accuracy trade-off
plot(bias_levels,max_acc_Bias_all,'r-','LineWidth',4);
hold on
plot(bias_levels,max_speed_acc_all,'b-','LineWidth',4);
plot(bias_levels,sub_acc_all,'c-','LineWidth',4);

lims=ylim;
line=linspace(lims(1),lims(2),100);
plot(repmat(acc_thresh,[1,length(line)]),line,'m-','LineWidth',4);

xlabel('bias')
ylabel('p(up = correct)')
legend({'max accuracy' 'max speed' 'simulated trade-off' 'Acc threshold'})
set(gca,'fontsize', 22);
hold off

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,'figures_decision/bias_acc_example_1Level.png')
end

pause

% look at change in speed for a given speed-accuracy trade-off
plot(bias_levels,decision_time_idx_noBias_all,'r-','LineWidth',4);
hold on
plot(bias_levels,decision_time_idx_Bias_all,'b-','LineWidth',4);
plot(bias_levels,sub_decision_time_idx_all,'c-','LineWidth',4);

lims=ylim;
line=linspace(lims(1),lims(2),100);
plot(repmat(acc_thresh,[1,length(line)]),line,'m-','LineWidth',4);

xlabel('bias')
ylabel('decision time')
legend({'max accuracy' 'max speed' 'simulated trade-off' 'Acc threshold'})
set(gca,'fontsize', 22);
hold off

set(gcf, 'Position',  [0, 0, 1440, 900])

if savefig
    saveas(gcf,'figures_decision/bias_dt_example_1Level.png')
end

pause

%% simulate decision process across coherence levels
bias_fix = 0.8; % choose bias level where speed-accuracy adjustemt effect is large
fix_bound = 0.8; % use fixed bound on evidence across coherence levels

coh_levels = 0.01:0.01:1; % min coherence level must be where acc_thresh can be reached within the maximal trial time.
drifts_sim = kappa.*coh_levels;

% % model accuracy threshold for each coherence level (should be determined empirically in study)
% threshold = 0.72;
% slope = 3.5;
% guess = 0.5;
% lapse = 0.01;
% acc_thresh_sim = zeros(1,length(coh_levels));
% 
% for c = 1:length(coh_levels)
%     acc_thresh_sim(c) = weibull(coh_levels(c), threshold, slope, guess, lapse);
% end

% simulate decision process
for thresh_sim = 1:2 % cycle over fix bound (=1) and decaying acc thresh (=2)

    % initialise output for plotting
    speed_acc_adjustment = zeros(1,length(coh_levels));
    speed_adjustment_all = zeros(1,length(coh_levels));
    acc_adjustment_all = zeros(1,length(coh_levels));
    decision_time_idx_noBias_all = zeros(1,length(coh_levels));
    acc_thresh_all = zeros(1,length(coh_levels));
    decision_var_noBias_all = zeros(1,length(coh_levels));

    for c = 1:length(coh_levels)
    
    if thresh_sim == 1 % use fixed evidence threshold
        [decision_time_idx_noBias,...
            decision_var_noBias,...
            decision_time_idx_Bias,...
            decision_var_Bias,...
            max_acc_Bias,...
            max_speed_acc,...
            sub_decision_time_idx,...
            sub_decision_var,...
            sub_acc,...
            p_up_correct_noBias,...
            p_up_correct_Bias,...
            acc_thresh] = sim_decision(t, drifts_sim(c), [], acc_weight,bias_fix, nsim, fix_bound);
    elseif thresh_sim == 2 % use acc threshold
        [decision_time_idx_noBias,...
            decision_var_noBias,...
            decision_time_idx_Bias,...
            decision_var_Bias,...
            max_acc_Bias,...
            max_speed_acc,...
            sub_decision_time_idx,...
            sub_decision_var,...
            sub_acc,...
            p_up_correct_noBias,...
            p_up_correct_Bias,...
            acc_thresh] = sim_decision(t, drifts_sim(c), acc_thresh_decay, acc_weight,bias_fix, nsim, []);
    end

        % collect ouputs
        decision_time_idx_noBias_all(c) = decision_time_idx_noBias;
        acc_thresh_all(c) = acc_thresh;
        decision_var_noBias_all(c) = decision_var_noBias;

        % compute speed-accuracy adjustment effect size for the simulated subject
        acc_adjustment = (sub_acc - acc_thresh)/(max_acc_Bias - acc_thresh); % accuracy adjustment normalised by max possible adjustment
        speed_adjustment = (decision_time_idx_noBias - sub_decision_time_idx)/(decision_time_idx_noBias - decision_time_idx_Bias); % speed adjustment divided by maximally possible adjustment

        speed_adjustment_all(c) = speed_adjustment;
        acc_adjustment_all(c) = acc_adjustment;

        % prevent division by zero when no speed adjustment is possible
        if isnan(speed_adjustment)
            speed_adjustment = 1; % maximum adjustment
        end

        speed_acc_adjustment(c) = (acc_adjustment + speed_adjustment)/2; % aggregate normalised speed-accuracy effect


    end

    %plot decision variable for each coherence level for the unbiased condition
    plot(coh_levels, decision_var_noBias_all,'b-','LineWidth',4);
    
    if thresh_sim == 1
        title('Fixed decision variable bound')
    elseif thresh_sim == 2
        title('Decaying accuracy threshold')
    end
    
    xlabel('coherence levels')
    ylabel('decision variable')
    set(gca,'fontsize', 22);

    set(gcf, 'Position',  [0, 0, 1440, 900])

    if (savefig && thresh_sim == 1)
        saveas(gcf,'figures_decision/coh_dv_example_1bias_fixDV.png')
    elseif (savefig && thresh_sim == 2)
        saveas(gcf,'figures_decision/coh_dv_example_1bias_accThreshDecay.png')
    end

    pause

    %plot decision times for each coherence level for the unbiased condition
    plot(coh_levels, decision_time_idx_noBias_all,'b-','LineWidth',4);
    
    if thresh_sim == 1
        title('Fixed decision variable bound')
    elseif thresh_sim == 2
        title('Decaying accuracy threshold')
    end

    xlabel('coherence levels')
    ylabel('decision time')
    set(gca,'fontsize', 22);

    set(gcf, 'Position',  [0, 0, 1440, 900])

    if (savefig && thresh_sim == 1)
        saveas(gcf,'figures_decision/coh_dt_example_1bias_fixDV.png')
    elseif (savefig && thresh_sim == 2)
        saveas(gcf,'figures_decision/coh_dt_example_1bias_accThreshDecay.png')
    end

    pause

    %plot accuracy for each coherence level for the unbiased condition
    plot(coh_levels, acc_thresh_all,'b-','LineWidth',4);

    if thresh_sim == 1
        title('Fixed decision variable bound')
    elseif thresh_sim == 2
        title('Decaying accuracy threshold')
    end
    
    xlabel('coherence levels')
    ylabel('p(up = correct)')
    set(gca,'fontsize', 22);

    set(gcf, 'Position',  [0, 0, 1440, 900])

    if (savefig && thresh_sim == 1)
        saveas(gcf,'figures_decision/coh_acc_example_1bias_fixDV.png')
    elseif (savefig && thresh_sim == 2)
        saveas(gcf,'figures_decision/coh_acc_example_1bias_accThreshDecay.png')
    end

    pause

    %plotting
    plot(coh_levels,speed_adjustment_all,'r-','LineWidth',4);
    hold on
    plot(coh_levels,acc_adjustment_all,'g-','LineWidth',4);
    plot(coh_levels,speed_acc_adjustment,'b-','LineWidth',4);
    ylim([0,1])

    if thresh_sim == 1
        title('Fixed decision variable bound')
    elseif thresh_sim == 2
        title('Decaying accuracy threshold')
    end
    
    xlabel('coherence levels')
    ylabel('speed-accuracy adjustment')
    legend({'speed adjustment' 'Acc adjustment' 'Total adjustment'})
    set(gca,'fontsize', 22);
    
    hold off

    set(gcf, 'Position',  [0, 0, 1440, 900])

    if (savefig && thresh_sim == 1)
        saveas(gcf,'figures_decision/coh_SAadj_example_1bias_fixDV.png')
    elseif (savefig && thresh_sim == 2)
        saveas(gcf,'figures_decision/coh_SAadj_example_1bias_accThreshDecay.png')
    end

    pause

end

% % psychometric function model
% function y = weibull(stim, threshold, slope, guess, lapse)
% % weibull function as used in QUEST. Models the relationship between
% % coherence level and accuracy
% 
% tmp=slope*(stim-threshold);
% y = lapse*guess+(1-lapse)*(1-(1-guess)*exp(-10^tmp));
% 
% end