% analyse simulated behaviour of the Bayesian agent model
clear all

% setup
addpath(genpath('/Users/skowron/Documents/structRDM_modelling/structRDM_model/sim_analysis/tools'))

sim_dir = '/Users/skowron/Documents/structRDM_modelling/structRDM_model/sim_results';
sim_fname = 'sim_taskV1_tinyHR_maxAcc';

% load sim results
load(fullfile(sim_dir,[sim_fname '.mat']));

coh_levels_signed = unique(exp_sim_info.coh_tr);
bias_levels = exp_sim_info.K_levels;

% initialise output
p_choice_right = cell(length(bias_levels),length(coh_levels_signed));
p_choice_right_unbiased = cell(length(bias_levels),length(coh_levels_signed));
RTs = cell(length(bias_levels),length(coh_levels_signed));
mean_RT = zeros(length(bias_levels),length(coh_levels_signed));
sd_RT = zeros(length(bias_levels),length(coh_levels_signed));
struct_belief_cond = cell(1,length(bias_levels));
struct_belief = cell(length(bias_levels),length(coh_levels_signed));
mean_struct_belief_cond = zeros(1,length(bias_levels));
mean_struct_belief = zeros(length(bias_levels),length(coh_levels_signed));
sd_struct_belief_cond = zeros(1,length(bias_levels));
sd_struct_belief = zeros(length(bias_levels),length(coh_levels_signed));

RT_block = cell(length(bias_levels),1);
coh_block = cell(length(bias_levels),1); % unsigned coherence level of trial

% add starting index of blocks
exp_sim_info.block_start_idx = cumsum(exp_sim_info.block_lengths);
exp_sim_info.block_start_idx = [1 exp_sim_info.block_start_idx(1:end-1)+1];

for b = 1:length(bias_levels)
    
    % bias condition vector
    bias_cond_idx = [];
    for k = 1:length(exp_sim_info.K_true)
        bias_cond_idx = [bias_cond_idx repmat(exp_sim_info.K_true(k) == bias_levels(b),[1,exp_sim_info.block_lengths(k)])];
    end
    
    bias_cond_idx = logical(bias_cond_idx);
    
    prior_belief_tr = model_allsim.prior_belief * sub_par.K';
    struct_belief_cond{b} = prior_belief_tr(bias_cond_idx);
    mean_struct_belief_cond(b) = mean(prior_belief_tr(bias_cond_idx));
    sd_struct_belief_cond(b) = std(prior_belief_tr(bias_cond_idx));
    
    % get temporal RT evolution throughout block
    bias_block_idx = find(exp_sim_info.K_true == bias_levels(b));
    RT_block{b} = nan(length(bias_block_idx),max(exp_sim_info.block_lengths));
    coh_block{b} = nan(length(bias_block_idx),max(exp_sim_info.block_lengths));
    
    for bl = 1:length(bias_block_idx)
        block_end_idx = exp_sim_info.block_start_idx(bias_block_idx(bl)) + exp_sim_info.block_lengths(bias_block_idx(bl)) - 1;
        RT_thisBlock = model_allsim.decision_time(exp_sim_info.block_start_idx(bias_block_idx(bl)) : block_end_idx);
        coh_thisBlock = abs(exp_sim_info.coh_tr(exp_sim_info.block_start_idx(bias_block_idx(bl)) : block_end_idx)); % unsigned trial coherence
        
        RT_block{b}(bl,1:length(RT_thisBlock)) = RT_thisBlock;
        coh_block{b}(bl,1:length(coh_thisBlock)) = coh_thisBlock;
        
        clear RT_thisBlock coh_thisBlock block_end_idx
    end
    
    for c = 1:length(coh_levels_signed)
        
        % compute mean probability of choosing up for a given bias
        % condition and coherence level
        tr_idx = (exp_sim_info.coh_tr == coh_levels_signed(c)) & logical(bias_cond_idx);
        
        p_choice_right{b,c} = model_allsim.conf_right(tr_idx);
        p_choice_right_unbiased{b,c} = model_allsim.like_right(tr_idx);
        
        RTs{b,c} = model_allsim.decision_time(tr_idx);
        mean_RT(b,c) = mean(model_allsim.decision_time(tr_idx));
        sd_RT(b,c) = std(model_allsim.decision_time(tr_idx));
        
        prior_belief_tr = model_allsim.prior_belief * sub_par.K';
        struct_belief{b,c} = prior_belief_tr(tr_idx)';
        mean_struct_belief(b,c) = mean(prior_belief_tr(tr_idx));
        sd_struct_belief(b,c) = std(prior_belief_tr(tr_idx));
        
    end
end

p_choice_rightM = cellfun(@(x) median(x), p_choice_right); % mean p_choice right
p_choice_rightSD = cellfun(@(x) std(x), p_choice_right); % SD of p_choice right

p_choice_rightN = cellfun(@(x) length(x), p_choice_right); % N datapoints p_choice right

RT_block_M = cellfun(@(x) nanmean(x,1), RT_block, 'UniformOutput', false); % mean RT per timepoint

%% plotting

% settings
cond_names = {'unstructured' 'structured'};
col={'b' 'r'};
p_offset=[-0.01 0.01];

% p(choice = bias congruent) - collapsed across structured and unstructured
% conditions
figure
hold on

%collapse structure conditions
p_choice_right(end+1,:) = flip(p_choice_right(find(bias_levels < 0.5),:));
p_choice_right(end,:) = cellfun(@(x) 1-x, p_choice_right(end,:), 'UniformOutput', false);

for i = 1:size(p_choice_right,2)
    p_choice_right{end,i} = [p_choice_right{end,i} p_choice_right{find(bias_levels > 0.5),i}];
end

p_choice_rightN(end+1,:) = cellfun(@(x) length(x), p_choice_right(end,:)); % N datapoints p_choice right
p_choice_rightM(end+1,:) = cellfun(@(x) median(x), p_choice_right(end,:));
p_choice_rightSD(end+1,:) = cellfun(@(x) std(x), p_choice_right(end,:));

for cond = 1:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(p_choice_right,1);
    end
    
    for c = 1:length(coh_levels_signed)
       plc(cond,c)=plot(repmat(coh_levels_signed(c),[1,p_choice_rightN(p_choice_idx,c)]) + p_offset(cond),p_choice_right{p_choice_idx,c},[col{cond} 'o']); 
    end
    
    pl(cond)=plot(coh_levels_signed,p_choice_rightM(p_choice_idx,:),['--' col{cond}]);
    
    clear p_choice_idx
end

legend(pl,cond_names);

hold off

xlim([-1,1])

xlabel('coherence level (bias congruent)')
ylabel('p(choice = bias congruent)')

pause

% plot mean condition differences
p_choice_rightM_diff = p_choice_rightM(size(p_choice_right,1),:) - p_choice_rightM(find(bias_levels == 0.5),:);

figure
plot(coh_levels_signed,p_choice_rightM_diff,'g-','LineWidth',8)

xlabel('coherence level (bias congruent)')
ylabel('p(choice = bias congruent) condition diff')

pause

% fit psychometric function for p(choice = bias congruent) as a function of
% signed coherence levels

ffits=cell(1,length(cond_names));

figure
hold on

for cond = 1:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(p_choice_right,1);
    end
    
    % set parameter bounds
    ul=[0.2 0.2 Inf Inf];
    sp=[0.1 0.1 0 0];
    ll=[0 0 -Inf -Inf];

    SP=[ul;sp;ll];

    [ffit, curve] = fitPsycheCurveBraun(coh_levels_signed,p_choice_rightM(p_choice_idx,:),SP);
    
    %collect output
    ffits{cond} = ffit;
    
    % plot
    plDAT(cond)=plot(coh_levels_signed,p_choice_rightM(p_choice_idx,:),[col{cond} 'o'],'MarkerSize',15); % plot empirical data
    plFIT(cond)=plot(curve(:,1),curve(:,2),[col{cond} '-'],'LineWidth',3); % plot fitted curve
    xlim([-1,1])
    xlabel('coherence level (bias congruent)')
    
end

legend(plFIT,cond_names);

hold off

pause

% p(choice = bias congruent | unbiased belief) - collapsed across structured and unstructured
% conditions
figure
hold on

%collapse structure conditions
p_choice_right_unbiased(end+1,:) = flip(p_choice_right_unbiased(find(bias_levels < 0.5),:));
p_choice_right_unbiased(end,:) = cellfun(@(x) 1-x, p_choice_right_unbiased(end,:), 'UniformOutput', false);

for i = 1:size(p_choice_right_unbiased,2)
    p_choice_right_unbiased{end,i} = [p_choice_right_unbiased{end,i} p_choice_right_unbiased{find(bias_levels > 0.5),i}];
end

p_choice_rightN_unbiased = cellfun(@(x) length(x), p_choice_right_unbiased); % N datapoints p_choice right
p_choice_rightM_unbiased = cellfun(@(x) median(x), p_choice_right_unbiased);
p_choice_rightSD_unbiased = cellfun(@(x) std(x), p_choice_right_unbiased);

for cond = 1:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(p_choice_right_unbiased,1);
    end
    
    for c = 1:length(coh_levels_signed)
       plc(cond,c)=plot(repmat(coh_levels_signed(c),[1,p_choice_rightN_unbiased(p_choice_idx,c)]) + p_offset(cond),p_choice_right_unbiased{p_choice_idx,c},[col{cond} 'o']); 
    end
    
    pl(cond)=plot(coh_levels_signed,p_choice_rightM_unbiased(p_choice_idx,:),['--' col{cond}]);
    
    clear p_choice_idx
end

legend(pl,cond_names);

hold off

xlabel('coherence level (bias congruent)')
ylabel('p(choice = bias congruent | unbiased belief)')

pause

% plot probability correct choice
p_choice_correct_temp = p_choice_right(1:length(bias_levels),:);
p_choice_correct_temp(:,coh_levels_signed < 0) = cellfun(@(x) 1-x, p_choice_correct_temp(:,coh_levels_signed < 0), 'UniformOutput', false);
p_choice_correct_temp(:,coh_levels_signed == 0) = []; % remove 0 coherence level condition as there is no correct choice

% collapse to get p(choice = correct) by motion strength
coh_levels = unique(abs(coh_levels_signed));
coh_levels(coh_levels == 0) = [];

coh_levels_signed_noZero = coh_levels_signed;
coh_levels_signed_noZero(coh_levels_signed_noZero == 0) = [];

p_choice_correct = cell(length(bias_levels)+1,length(coh_levels));

for c = 1:length(coh_levels)
    for b = 1:length(bias_levels)
    
        p_choice_correct{b,c} = [p_choice_correct_temp{b,find(coh_levels_signed_noZero == -coh_levels(c))} p_choice_correct_temp{b,find(coh_levels_signed_noZero == coh_levels(c))}];
    
    end
end

% collapse across bias conditions

for i = 1:size(p_choice_correct,2)
    p_choice_correct{end,i} = [p_choice_correct{find(bias_levels < 0.5),i} p_choice_correct{find(bias_levels > 0.5),i}];
end

p_choice_correctN = cellfun(@(x) length(x), p_choice_correct); % N datapoints p_choice right
p_choice_correctM = cellfun(@(x) mean(x), p_choice_correct);
p_choice_correctSD = cellfun(@(x) std(x), p_choice_correct);

clear p_choice_correct_temp

figure
hold on

for cond = 1:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(p_choice_correct,1);
    end
    
    for c = 1:length(coh_levels)
       plc(cond,c)=plot(repmat(coh_levels(c),[1,p_choice_correctN(p_choice_idx,c)]) + p_offset(cond),p_choice_correct{p_choice_idx,c},[col{cond} 'o']); 
    end
    
    pl(cond)=plot(coh_levels,p_choice_correctM(p_choice_idx,:),['--' col{cond}]);
    
    clear p_choice_idx
end

legend(pl,cond_names);

hold off

xlabel('motion strength')
ylabel('p(choice = correct)')

pause

% confidence (p(choice = bias conguent)) variability
figure
hold on 

for cond = 1:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(p_choice_right,1);
    end
    
    pl(cond)=plot(coh_levels_signed,p_choice_rightSD(p_choice_idx,:),['--' col{cond}]);
    
    clear p_choice_idx
end

legend(pl,cond_names);

hold off

xlabel('coherence level (bias congruent)')
ylabel('confidence variability')

pause

% RT
figure
hold on

%collapse across bias conds
RTs(end+1,:) = cell(1,size(RTs,2));
for c = 1:length(coh_levels_signed)
    RTs{end,c} = [RTs{find(bias_levels < 0.5),c} RTs{find(bias_levels > 0.5),c}];
end

RTs_N = cellfun(@(x) length(x), RTs);

mean_RT = cellfun(@(x) nanmean(x), RTs);

for cond = 1:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(RTs,1);
    end
    
    for c = 1:length(coh_levels_signed)
       plc(cond,c)=plot(repmat(coh_levels_signed(c),[1,RTs_N(p_choice_idx,c)]) + p_offset(cond),RTs{p_choice_idx,c},[col{cond} 'o']); 
    end
    
    pl(cond)=plot(coh_levels_signed,mean_RT(p_choice_idx,:),['--' col{cond}]);
    
    clear p_choice_idx
end

legend(pl,cond_names);

hold off

xlabel('coherence level (bias congruent)')
ylabel('RT')

pause

% sRT(1)=plot(coh_levels_signed,sd_RT(1,:),'-or');
% hold on
% sRT(2)=plot(coh_levels_signed,sd_RT(2,:),'-ob');
% sRT(3)=plot(coh_levels_signed,sd_RT(3,:),'-og');
% 
% if plot_comp
%     sRT(4)=plot(coh_levels_signed,comp_mod.sd_RT(1,:),'--or');
%     sRT(5)=plot(coh_levels_signed,comp_mod.sd_RT(2,:),'--ob');
%     sRT(6)=plot(coh_levels_signed,comp_mod.sd_RT(3,:),'--og');
% end
% 
% sRT(1:3)=legend({num2str(bias_levels(1)),num2str(bias_levels(2)),num2str(bias_levels(3))});
% 
% hold off
% 
% xlabel('coherence level')
% ylabel(('SD RT'))

% RT evolution
RT_block{end+1} = [RT_block{find(bias_levels < 0.5)}; RT_block{find(bias_levels > 0.5)}];
RT_block_M{end+1} = nanmedian(RT_block{end}, 1);

% % mean RT for each coh level for struct condition
% coh_levels = unique(abs(exp_sim_info.coh_tr);
% RT_block_M_coh = nan(length(coh_levels)),max(exp_sim_info.block_lengths));
% 
% for coh = 1:length(coh_levels)
%    RT_block_M_coh() 
% end

coh_block{end+1} = [coh_block{find(bias_levels < 0.5)}; coh_block{find(bias_levels > 0.5)}];

figure

pl=[];
plc=[];

hold on

for cond = 2:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(RT_block,1);
    end
    
    for tr = 1:max(exp_sim_info.block_lengths)
       plc(length(plc)+1,tr)=scatter(repmat(tr,[1,size(RT_block{p_choice_idx},1)]) + p_offset(cond), RT_block{p_choice_idx}(:,tr),[],coh_block{p_choice_idx}(:,tr)); 
    end
    
    pl(length(pl)+1)=plot(1:max(exp_sim_info.block_lengths),RT_block_M{p_choice_idx},['--' col{cond}]);
    
    clear p_choice_idx
end

legend(pl,cond_names{2});

colormap('autumn');
cm=colorbar

cm.Title.String = "motion strength";

xlabel('trial in block')
ylabel('RT')

hold off

pause

% belief
figure
hold on

for b = 1:length(bias_levels)

    plc(b)=plot(repmat(bias_levels(b),[1,length(struct_belief_cond{b})]),struct_belief_cond{b},'ro'); 
    
end

pl = plot(bias_levels,mean_struct_belief_cond,'r--');

hold off

xlim([0,1])

xlabel('bias level')
ylabel('struct belief')

pause

figure
hold on

plot(bias_levels,sd_struct_belief_cond,'-or');

hold off

xlim([0,1])

xlabel('bias level')
ylabel('SD struct belief')

% plot struct belief by trial coherence level

figure
hold on

%collapse structure conditions
struct_belief(end+1,:) = flip(struct_belief(find(bias_levels < 0.5),:));
struct_belief(end,:) = cellfun(@(x) 1-x, struct_belief(end,:), 'UniformOutput', false); % for the structured condition base rate for left/right motion direction respectively.

for i = 1:size(struct_belief,2)
    struct_belief{end,i} = [struct_belief{end,i} struct_belief{find(bias_levels > 0.5),i}];
end

struct_beliefN = cellfun(@(x) length(x), struct_belief); % N datapoints struct_belief
mean_struct_belief(end+1,:) = cellfun(@(x) mean(x), struct_belief(end,:));
sd_struct_belief(end+1,:) = cellfun(@(x) std(x), struct_belief(end,:));

for cond = 1:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(struct_belief,1);
    end
    
    for c = 1:length(coh_levels_signed)
       plc(cond,c)=plot(repmat(coh_levels_signed(c),[1,struct_beliefN(p_choice_idx,c)]) + p_offset(cond),struct_belief{p_choice_idx,c},[col{cond} 'o']); 
    end
    
    pl(cond)=plot(coh_levels_signed,mean_struct_belief(p_choice_idx,:),['--' col{cond}]);
    
    clear p_choice_idx
end

legend(pl,cond_names);

hold off

xlabel('coherence level (bias congruent)')
ylabel('struct belief (bias congruent)')

% overlay p(choice = bias congruent | unbiased belief) and struct belief
% (which together influence p(choice = bias congruent)) for structured condition

figure
hold on

custom_offset = [-0.03 0 0.03];

plc=[];
pl=[];

for cond = 2:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(p_choice_right_unbiased,1);
    end
    
    for c = 1:length(coh_levels_signed)
       plc(length(plc)+1,c)=plot(repmat(coh_levels_signed(c),[1,p_choice_rightN_unbiased(p_choice_idx,c)]) + custom_offset(1),p_choice_right_unbiased{p_choice_idx,c},[col{cond} 'o']); 
    end
    
    pl(length(pl)+1)=plot(coh_levels_signed,p_choice_rightM_unbiased(p_choice_idx,:),['--' col{cond}]);
    
    clear p_choice_idx
end

col_struct={'c','m'};

for cond = 2:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(struct_belief,1);
    end
    
    for c = 1:length(coh_levels_signed)
       plc(length(plc)+1,c)=plot(repmat(coh_levels_signed(c),[1,struct_beliefN(p_choice_idx,c)]) + custom_offset(2),struct_belief{p_choice_idx,c},[col_struct{cond} 'o']); 
    end
    
    pl(length(pl)+1)=plot(coh_levels_signed,mean_struct_belief(p_choice_idx,:),['--' col_struct{cond}]);
    
    clear p_choice_idx
end

% legend(pl,{'struct belief (struct cond)','','sensory evidence (struct cond)',''});

% % sanity check to see if p_choice_right is computed correctly
% p_right_check = cell(1,length(coh_levels_signed));
% 
% for coh = 1:length(coh_levels_signed)
%     
%     p_choice_right_check{coh} = p_choice_right_unbiased{end,coh} .* struct_belief{end,coh};
%     p_choice_right_check{coh} = p_choice_right_check{coh} ./ (p_choice_right_check{coh} + ((1-struct_belief{end,coh}) .* (1-p_choice_right_unbiased{end,coh}))); %normalisation
% 
% end
% 
% p_choice_right_checkM = cellfun(@(x) median(x), p_choice_right_check);
% p_choice_right_checkN = cellfun(@(x) length(x), p_choice_right_check);

for cond = 2:length(cond_names)
    
    if cond == 1
        p_choice_idx = find(bias_levels == 0.5);
    elseif cond == 2
        p_choice_idx = size(p_choice_right,1);
    end
    
    for c = 1:length(coh_levels_signed)
       plc(length(plc)+1,c)=plot(repmat(coh_levels_signed(c),[1,p_choice_rightN(p_choice_idx,c)]) + custom_offset(3),p_choice_right{p_choice_idx,c},['ko']); 
    end
    
    pl(length(pl)+1)=plot(coh_levels_signed,p_choice_rightM(p_choice_idx,:),['k--']);
    
    clear p_choice_idx
end

title('structured conditions')
legend(pl,{'sensory evidence (likelihood)','struct belief (prior)','p(choice = bias congruent) (posterior)'})

hold off

xlabel('coherence level (bias congruent)')
ylabel('choice probability')

pause


%% saving

save([sim_fname '_beh.mat'],'RT_block','coh_levels_signed','bias_levels','ffits','p_choice_right','p_choice_rightM','p_choice_rightN','mean_RT','sd_RT','mean_struct_belief_cond','sd_struct_belief_cond','struct_belief_cond')
