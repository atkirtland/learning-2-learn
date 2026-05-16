clear
close all

useSaved = true;

% dir = '../data/2023-08-01T17:43:58_plain';
% dir = '../data/2023-08-01T17:44:08_proj';

dirs_sets = {
    {'2023-11-19T00:18:02_plains0', '2023-10-07T23:34:02_plains1', '2023-10-07T23:34:37_plains2'};
    {'2023-10-11T23:36:53_lcp7.5s0', '2023-10-11T23:37:04_lcp7.5s1', '2023-10-11T23:37:14_lcp7.5s2'};
    {'2023-10-12T21:24:49_lcp6.0s0', '2023-10-12T21:25:04_lcp6.0s1', '2023-10-12T21:25:13_lcp6.0s2'};
    {'2023-10-23T11:31:41_lcp5.5s0', '2023-10-23T11:32:16_lcp5.5s1', '2023-10-23T11:32:45_lcp5.5s2'};
};
names = {'plain', 'lcp7-5', 'lcp6-0', 'lcp5-5'};

dirs_idx = 1;
dirs = dirs_sets{dirs_idx};
name = names{dirs_idx};

dir = '2023-11-19T00:18:02_plains0';

numNet = 3;
numTasks = 31; 

dt = 1.0;
tau = 100.0;
alphax = dt/tau;
N = 100;
T = 2;

probsPerGroup = 2; % default 50
numEp = 15; % default 20
halfway = (numTasks-1)/2; % default 500

if useSaved == false
    
    corrs = []; % to save correlation coefficients / R^2 between weight change magnitudes and learning performance
    for net = 1:numNet
        
        VFC = zeros(numTasks-1,9); % to save vector field change magnitude measurements
        % to save vector field change dimensionality measurements
        dims_SingProb = zeros(numTasks-1,2,2);
        dims_SingEp = zeros(numEp,2,2);
        dimsT_SingProb = zeros(numTasks-1,1);
        dimsT_SingEp = zeros(numEp,1);
        
        tc=0;
        % loop through problems in groups of 50 at a time
        for ep = 1:numEp

            disp(strcat('ep: ', int2str(ep)))
            
            tasks = ((ep-1)*probsPerGroup+2):(ep*probsPerGroup +1);

            comps = zeros(2, 2, N, 50, 2, int32(2000/dt));
            compsT = zeros(N, 50, 2, int32(2000/dt));
            t_ec = 0;
            
            % loop through each task in the current 50-problem epoch
            for task = tasks
                tc = tc+1;
                t_ec = t_ec+1;
                [task tc t_ec]
                
                file_path = sprintf('../data/%s/saved/%d.mat', dirs{net}, task-1);
                load(file_path);
                wts_RNNin_weights = wts_leakyRNN_kernel(1:11, :);
                wts_leakyRNN_weights = wts_leakyRNN_kernel(12:end, :);

                IS0 = double(wts_leakyRNN_init_state);
                IW0 = double(wts_RNNin_weights);
                RW0 = double(wts_leakyRNN_weights);
                RB0 = double(wts_leakyRNN_biases);
                
                file_path = sprintf('../data/%s/saved/%d.mat', dirs{net}, task);
                load(file_path);
                wts_RNNin_weights = wts_leakyRNN_kernel(1:11, :);
                wts_leakyRNN_weights = wts_leakyRNN_kernel(12:end, :);

                IS = double(wts_leakyRNN_init_state);
                IW = double(wts_RNNin_weights);
                RW = double(wts_leakyRNN_weights);
                RB = double(wts_leakyRNN_biases);
                images = double(images);
                
                R = repmat(IS,T,1);
                R0x = repmat(IS,T,1);
                
                st = zeros(T,11);
                st(:,11) = 1.0/sqrt(10.0);
                st(1:2,1:10) = images;
                
                mWC = 0;
                
                mW = 0;
                mW_par = 0;
                mW_perp = 0;
                
                mS = 0;
                mS_par = 0;
                mS_perp = 0;
                
                mT = 0;
                
                % loop through time to measure vector field quantity
                % at each time step
                for t = 1:int32(2000/dt)
                    
                    if t == int32(500/dt)+1
                        st = zeros(T,11);
                        st(:,11) = 1.0/sqrt(10.0);
                    elseif t == int32(1500/dt)+1
                        st = zeros(T,11);
                    end
                    
                    Dx = R-R0x;
                    
                    % change in post-syn current magnitudes
                    vfcW_C = st*(IW-IW0) + R*(RW-RW0) + (RB-RB0);
                    mWC = mWC + vecnorm(vfcW_C, 2, 2);
                    
                    % weight-driven vector field change magnitudes
                    vfcW = alphax*(log(1+exp(st*IW + R*RW + RB))-log(1+exp(st*IW0 + R*RW0 + RB0)));
                    mW = mW + vecnorm(vfcW, 2, 2);
                    
                    % state-driven vector field change magnitudes
                    vfcS = alphax*(-R + R0x + log(1+exp(st*IW0 + R*RW0 + RB0))-log(1+exp(st*IW0 + R0x*RW0 + RB0)));
                    mS = mS + vecnorm(vfcS, 2, 2);
                    
                    dZ = vfcW + vfcS; % delta Z
                    mdZ = vecnorm(dZ, 2, 2);
                    mT = mT + mdZ;
                    
                    % normalize dZ vector magnitude
                    dZ = dZ./repmat(mdZ, 1, N);
                    
                    % compute parallel and perp. components of
                    % weight- and state-driven VFCs relative to dZ
                    vfcW_par = repmat(diag(vfcW*dZ'), 1, N).*dZ;
                    vfcW_perp = vfcW - vfcW_par;
                    mW_par = mW_par + vecnorm(vfcW_par, 2, 2);
                    mW_perp = mW_perp - vecnorm(vfcW_perp, 2, 2);
                    comps(1, 1, :, t_ec, : , t) = vfcW_par';
                    comps(1, 2, :, t_ec, : , t) = vfcW_perp';
                    
                    vfcS_par = repmat(diag(vfcS*dZ'), 1, N).*dZ;
                    vfcS_perp = vfcS - vfcS_par;
                    mS_par = mS_par + vecnorm(vfcS_par, 2, 2);
                    mS_perp = mS_perp + vecnorm(vfcS_perp, 2, 2);
                    comps(2, 1, :, t_ec, : , t) = vfcS_par';
                    comps(2, 2, :, t_ec, : , t) = vfcS_perp';
                    
                    compsT(:, t_ec, : , t) = dZ';
                    
                    % simulate forward in time
                    R = (1.0-alphax)*R + alphax*log(1+exp(st*IW + R*RW + RB));
                    R0x = (1.0-alphax)*R0x + alphax*log(1+exp(st*IW0 + R0x*RW0 + RB0));
                    
                    
                end
                % temporal mean of magnitudes
                mWC = mWC./(2000/dt);
                
                mW = mW./(2000/dt);
                mW_par = mW_par./(2000/dt);
                mW_perp = mW_perp./(2000/dt);
                
                mS = mS./(2000/dt);
                mS_par = mS_par./(2000/dt);
                mS_perp = mS_perp./(2000/dt);
                
                mT = mT./(2000/dt);
                
                % dimensionality of VFC quantities within single problems
                for i = 1:2
                    for j = 1:2
                        [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(comps(i, j, :, t_ec, :))', 'Centered', false);
                        dims_SingProb(tc, i, j) = (sum(LATENT).^2)/sum(LATENT.^2);
                    end
                end
                [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(compsT(:, t_ec, :))', 'Centered', false);
                dimsT_SingProb(tc) = (sum(LATENT).^2)/sum(LATENT.^2);
                
                
                VFC(tc,1) = mean(mWC);
                
                VFC(tc,2) = mean(mW);
                VFC(tc,3) = mean(mS);
                
                VFC(tc,4) = vecnorm(RW(:)-RW0(:));
                
                VFC(tc,5) = mean(mT);
                
                VFC(tc,6) = mean(mW_par);
                VFC(tc,7) = mean(mW_perp);
                VFC(tc,8) = mean(mS_par);
                VFC(tc,9) = mean(mS_perp);
                
            end
            
            % dimensionality of VFC quantities across 50-problem group
            for i = 1:2
                for j = 1:2
                    [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(comps(i,j, :, :))', 'Centered', false);
                    dims_SingEp(ep, i, j) = (sum(LATENT).^2)/sum(LATENT.^2);
                end
            end
            [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(squeeze(compsT( :, :))', 'Centered', false);
            dimsT_SingEp(ep) = (sum(LATENT).^2)/sum(LATENT.^2);
            
        end
        % Correlate weight change magnitude with learning performance
        C = load(sprintf('../data/%s/conv.txt',dirs{net}));
        C = C(2:end, 1);
        corrs = [corrs; corr(VFC(:,2),C) corr(VFC(:,3),C) corr(VFC(:,4),C)];
        
        %Summarize quantities by problem group for plotting
        VFC_sum = zeros(numEp, 9);
        dims_SingProb_sum = zeros(numEp,2,2);
        dimsT_SingProb_sum = zeros(numEp,1);
        
        for ep = 1:numEp
            tasks = ((ep-1)*probsPerGroup+1):(ep*probsPerGroup);
            VFC_sum(ep, :) = mean(VFC(tasks,:),1);
            dims_SingProb_sum(ep, :, :) = mean(dims_SingProb(tasks, :, :),1);
            dimsT_SingProb_sum(ep) = mean(dimsT_SingProb(tasks));
        end
        
        % Raw plots
        figure;
        subplot(2, 6, 1)
        plot(dims_SingProb_sum(:, 1, 1)); hold on
        plot(dims_SingEp(:, 1, 1))
        subplot(2, 6, 2)
        plot(dims_SingProb_sum(:, 1, 2)); hold on
        plot(dims_SingEp(:, 1, 2))
        subplot(2, 6, 3)
        plot(dims_SingProb_sum(:, 2, 1)); hold on
        plot(dims_SingEp(:, 2, 1))
        subplot(2, 6, 4)
        plot(dims_SingProb_sum(:, 2, 2)); hold on
        plot(dims_SingEp(:, 2, 2))
        subplot(2, 6, 5)
        plot(dimsT_SingProb_sum); hold on
        plot(dimsT_SingEp)
        
        subplot(2, 6, 6)
        plot(corrs(end,:).^2)
        
        
        subplot(2, 6, 7)
        plot(VFC_sum(:, 1))
        subplot(2, 6, 8)
        plot(VFC_sum(:, [2 3 5]))
        subplot(2, 6, 9)
        plot(VFC_sum(:, 4))
        subplot(2, 6, 10)
        plot(VFC_sum(:, [2 6 7]))
        subplot(2, 6, 11)
        plot(VFC_sum(:, [3 8 9]))
        save(sprintf('../results/f5And6_seed_%d_%s.mat', net-1, name), 'C', 'VFC', 'VFC_sum', 'dims_SingEp','dimsT_SingEp', 'dims_SingProb_sum', 'dimsT_SingProb_sum');

        saveas(gca, sprintf('../results/f5and6_seed_%d_%s.pdf', net-1, name), 'pdf')
    end
    save(sprintf('../results/f5And6_All_%s.mat', name), 'corrs');
end

% Summarize data for plots
dZDecMag = zeros(10, 2, 3);
dZDecDim = zeros(10, 2, 2);
dW_VFC = zeros(10, 1);
dMag = zeros(10,numEp,3);
for net = 1:numNet
    load(sprintf('../results/f5And6_seed_%d_%s.mat', net-1, name));
    dZDecMag(net, 1, 1) = VFC_sum(1, 5);
    dZDecMag(net, :, 2) = VFC_sum(1, 6:7)';
    dZDecMag(net, :, 3) = VFC_sum(1, 8:9)';
    
    dZDecDim(net, 1, 1:2) = dims_SingProb_sum(1,1,1:2);
    dZDecDim(net, 2, 1:2) = dims_SingEp(1,1,1:2);

    dW_VFC(net) = corr(VFC(1:halfway,4), C(1:halfway))^2;
    
    X = VFC_sum(:,[4, 1, 2]);
    X = X./repmat(X(1,:), numEp, 1);
    dMag(net,:,:) = X;

end
load(sprintf('../results/f5And6_All_%s.mat', name));
% dW_VFC = corrs(:,3);

figure;
subplot(1,6,1)
h = boxplot(squeeze(dZDecMag(:,:,1)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',1)
set(gca,'xticklabel',{'Total deltaz'})
ylabel('Magnitude (a.u.)')
ylim([-0.015 0.04])
subplot(1,6,2)
h = boxplot(squeeze(dZDecMag(:,:,2)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'WdVFC(par)', 'WdVFC(perp)'})
ylim([-0.015 0.04])
subplot(1,6,3)
h = boxplot(squeeze(dZDecMag(:,:,3)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'SdVFC(par)', 'SdVFC(perp)'})
ylim([-0.015 0.04])


subplot(1,6,5)
h = boxplot(squeeze(dZDecDim(:,:,1)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'Single(par)', 'Group(par)'})
ylabel('Dimensionality')
ylim([0 14])
subplot(1,6,6)
h = boxplot(squeeze(dZDecDim(:,:,2)));
set(h,{'linew'},{2});
box off
set(gca,'fontsize',15)
set(gca,'linewidth',2)
set(gca,'xtick',[1 2])
set(gca,'xticklabel',{'Single(perp)', 'Group(perp)'})
ylim([0 14])

fig1 = figure(1);
fig1.WindowState = 'maximized';
orient(fig1,'landscape')
saveas(fig1, sprintf('../results/f5and6_all_1_%s.pdf', name), 'pdf')

figure;
subplot(1,3,1)
% load(sprintf('../results/f5And6_seed_%d', 3));
load(sprintf('../results/f5And6_seed_%d', 0));
plot(VFC(1:halfway,4), C(1:halfway), 'o')
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
xlabel('Trials to Criterion')
ylabel('Change in W_{rec}')
ylim([0 450])
xlim([0 0.43])

subplot(1,3,2)
h = boxplot(dW_VFC.*100);
set(h,{'linew'},{2});
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
ylabel('Explained variance')
ylim([40 100])
set(gca,'xtick',[])

subplot(1,3,3)
% errorbar(repmat([25:50:1000]',1,3), squeeze(mean(dMag,1)), squeeze(std(dMag,0,1))./sqrt(10), 'linewidth',2)    
errorbar(repmat([1:probsPerGroup:numTasks-1]',1,3), squeeze(mean(dMag,1)), squeeze(std(dMag,0,1))./sqrt(10), 'linewidth',2)
box off
set(gca,'fontsize',20)
set(gca,'linewidth',2)
xlabel('Problems')
ylabel('Normalized magnitude')
legend('delta(W_{rec})','delta(Curr.)','W-d VFC')
legend boxoff

fig1 = figure(1);
fig1.WindowState = 'maximized';
orient(fig1,'landscape')
saveas(fig1, sprintf('../results/f5and6_all_2_%s.pdf', name), 'pdf')