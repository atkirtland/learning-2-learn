clear
close all

% dir = '../data/2023-08-01T17:43:58_plain';
% dir = '../data/2023-08-01T17:44:08_proj';


% dirs_sets = [
%     ['2023-11-19T00:18:02_plains0', '2023-10-07T23:34:02_plains1', '2023-10-07T23:34:37_plains2']
%     ['2023-10-12T21:24:49_lcp6.0s0', '2023-10-12T21:25:04_lcp6.0s1', '2023-10-12T21:25:13_lcp6.0s2']
%     ['2023-10-23T11:31:41_lcp5.5s0', '2023-10-23T11:32:16_lcp5.5s1', '2023-10-23T11:32:45_lcp5.5s2']
%     ['2023-10-11T23:36:53_lcp7.5s0', '2023-10-11T23:37:04_lcp7.5s1', '2023-10-11T23:37:14_lcp7.5s2']
% ];

% dirs_sets = [
%     ['2023-11-19T00:18:02_plains0', '2023-10-07T23:34:02_plains1', '2023-10-07T23:34:37_plains2'],
%     ['2023-10-12T21:24:49_lcp6.0s0', '2023-10-12T21:25:04_lcp6.0s1', '2023-10-12T21:25:13_lcp6.0s2'],
%     ['2023-10-23T11:31:41_lcp5.5s0', '2023-10-23T11:32:16_lcp5.5s1', '2023-10-23T11:32:45_lcp5.5s2'],
%     ['2023-10-11T23:36:53_lcp7.5s0', '2023-10-11T23:37:04_lcp7.5s1', '2023-10-11T23:37:14_lcp7.5s2']
% ];


dirs_sets = {
    {'2023-11-19T00:18:02_plains0', '2023-10-07T23:34:02_plains1', '2023-10-07T23:34:37_plains2'};
    {'2023-10-11T23:36:53_lcp7.5s0', '2023-10-11T23:37:04_lcp7.5s1', '2023-10-11T23:37:14_lcp7.5s2'};
    {'2023-10-12T21:24:49_lcp6.0s0', '2023-10-12T21:25:04_lcp6.0s1', '2023-10-12T21:25:13_lcp6.0s2'};
    {'2023-10-23T11:31:41_lcp5.5s0', '2023-10-23T11:32:16_lcp5.5s1', '2023-10-23T11:32:45_lcp5.5s2'};
};



dt = 1.0;
tau = 100.0;
alphax = dt/tau;
N = 100;
T = 2;

for dirs_idx = 1:length(dirs_sets)
    dirs = dirs_sets{dirs_idx};
    for net = 1:1
        for ep = 1:1
            figure;
            % tasks = ((ep-1)*50+2):(ep*50 +1);
            tasks = 1:30;
            tCnt = 0;
            rHist = zeros(int32(2000/dt), 100, N); % To save learned trajectories
            for task = tasks
                tCnt = tCnt+1;

                file_path = sprintf('../data/%s/saved/%d.mat', dirs{net}, task);
                disp(file_path);
                load(file_path);
                wts_RNNin_weights = wts_leakyRNN_kernel(1:11, :);
                wts_leakyRNN_weights = wts_leakyRNN_kernel(12:end, :);

                IS = double(wts_leakyRNN_init_state);
                IW = double(wts_RNNin_weights);
                RW = double(wts_leakyRNN_weights);
                RB = double(wts_leakyRNN_biases);
                images = double(images);

                R = repmat(IS,T,1); % network initial condition

                % setup input stimuli
                st = zeros(T,11);
                st(:,11) = 1.0/sqrt(10.0);
                st(1:2,1:10) = images;            
                for t = 1:int32(2000/dt)

                    if t == int32(500/dt)+1
                        st = zeros(T,11);
                        st(:,11) = 1.0/sqrt(10.0);
                    elseif t == int32(1500/dt)+1
                        st = zeros(T,11);
                    end

                    % Simulate network and save trajectory
                    R = (1.0-alphax)*R + alphax*log(1+exp(st*IW + R*RW + RB));
                    rHist(t,[tCnt tCnt+50],:) = R;                
                end            
            end

            % Compute decision subspace loading vectors and projection matrix
            decTrajectories = cat(2, mean(rHist(:,1:50,:), 2), mean(rHist(:,51:100,:), 2));
            decTrajectories = permute(decTrajectories, [3,2,1]);
            [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(decTrajectories(:,:)', 'Centered', false);
            LV_Dec = COEFF(:,1:4);
            P = LV_Dec*LV_Dec';

            % Compute stimulus subspace loading vectors and projection matrix
            Q = eye(N) - P;
            rHist = permute(rHist, [3,2,1]);
            rHistD = P*rHist(:,:);
            rHistS = rHist(:,:) - rHistD;
            [COEFF, SCORE, LATENT, TSQUARED, EXPLAINED, MU] = pca(rHistS', 'Centered', false);
            LV_Stim = COEFF(:,1:3);

            % reshape for output current calculation and plotting
            rHistD = reshape(rHistD, [N, 100, int32(2000/dt)]);
            rHistS = reshape(rHistS, [N, 100, int32(2000/dt)]);        

            % loop to calculate net post-synaptic currents at output units per problem
            % tasks = ((ep-1)*50+2):(ep*50 +1);
            tasks = 1:30;
            tCnt = 0;
            outProj = zeros(3, 3, 100, int32(2000/dt));
            for task = tasks
                tCnt = tCnt+1;
                load(sprintf('../data/%s/saved/%d.mat', dirs{net}, task));

                OW = double(wts_out_RNN_weights);

                RtaskD = rHistD(:,[tCnt tCnt+50],:);
                RtaskS = rHistS(:,[tCnt tCnt+50],:);

                outProj(1,:,tCnt,:) = OW'*squeeze(mean(RtaskD, 2)); % mean decision -> output current
                % mean decision representation in first 3 modes of decision subspace
                RD_mean = LV_Dec'*squeeze(mean(RtaskD, 2));
                subplot(3,3,1)
                hold on
                plot3(RD_mean(1,:), RD_mean(2,:), RD_mean(3,:), 'k')

                % residual decision representation in first 3 modes of decision subspace
                subplot(3,3,2)
                hold on
                RD_resid = LV_Dec'*squeeze(RtaskD(:,1,:) - mean(RtaskD, 2)); % resid. decision -> output current
                outProj(2,:,tCnt,:) = OW'*squeeze(RtaskD(:,1,:) - mean(RtaskD, 2));
                hold on;
                plot3(RD_resid(1,:), RD_resid(2,:), RD_resid(3,:), 'b')
                RD_resid = LV_Dec'*squeeze(RtaskD(:,2,:) - mean(RtaskD, 2)); % resid. decision -> output current
                outProj(2,:,tCnt+50,:) = OW'*squeeze(RtaskD(:,2,:) - mean(RtaskD, 2));
                hold on
                plot3(RD_resid(1,:), RD_resid(2,:), RD_resid(3,:), 'r')

                % stimulus representation in first 3 modes of stimulus subspace
                subplot(3,3,3)
                RS = LV_Stim'*squeeze(RtaskS(:,1,:));
                outProj(3,:,tCnt,:) = OW'*squeeze(RtaskS(:,1,:));  % stim -> output current
                hold on
                plot3(RS(1,:), RS(2,:), RS(3,:), 'b')
                RS = LV_Stim'*squeeze(RtaskS(:,2,:));
                outProj(3,:,tCnt+50,:) = OW'*squeeze(RtaskS(:,2,:));  % stim -> output current
                hold on
                plot3(RS(1,:), RS(2,:), RS(3,:), 'r')            
            end
            subplot(3,3,1)
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            title('Mean Decision')
            set(gca,'xticklabel',[])
            set(gca,'yticklabel',[])
            set(gca,'zticklabel',[])
            grid on
            subplot(3,3,2)
            legend('Mapping 1','Mapping 2')
            legend boxoff        
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            title('Residual Decision')
            set(gca,'xticklabel',[])
            set(gca,'yticklabel',[])
            set(gca,'zticklabel',[])
            grid on
            subplot(3,3,3)
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            title('Stimulus')
            set(gca,'xticklabel',[])
            set(gca,'yticklabel',[])
            set(gca,'zticklabel',[])
            grid on

            subplot(3,3,4)
            for i = 1:2
                errorbar(squeeze(mean(outProj(1,i,1:50,:),3)), squeeze(std(outProj(1,i,1:50,:),0,3))./sqrt(50))
                hold on
            end
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            legend('Response 1','Response 2')
            box off
            legend boxoff
            ylim([-18 8])
            set(gca,'xtick',[0 500 1500 2000])
            set(gca,'xticklabel',[0 0.5 1.5 2.0])
            xlabel('Time (s)')
            ylabel('Post-synaptic Currents')
            subplot(3,3,5)
            for i = 1:2
                errorbar(squeeze(mean(outProj(2,i,1:50,:),3)), squeeze(std(outProj(2,i,1:50,:),0,3))./sqrt(50))
                hold on
            end
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            box off
            ylim([-8 8])        
            set(gca,'xtick',[0 500 1500 2000])
            set(gca,'xticklabel',[0 0.5 1.5 2.0])
            xlabel('Time (s)')
            subplot(3,3,6)
            for i = 1:2
                errorbar(squeeze(mean(outProj(3,i,1:50,:),3)), squeeze(std(outProj(3,i,1:50,:),0,3))./sqrt(50))
                hold on
            end        
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            box off
            ylim([-3 3])        
            set(gca,'xtick',[0 500 1500 2000])
            set(gca,'xticklabel',[0 0.5 1.5 2.0])
            xlabel('Time (s)')
            subplot(3,3,8)
            for i = 1:2
                errorbar(squeeze(mean(outProj(2,i,51:100,:),3)), squeeze(std(outProj(2,i,51:100,:),0,3))./sqrt(50))
                hold on
            end        
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            box off
            ylim([-8 8])        
            set(gca,'xtick',[0 500 1500 2000])
            set(gca,'xticklabel',[0 0.5 1.5 2.0])
            xlabel('Time (s)')
            subplot(3,3,9)
            for i = 1:2
                errorbar(squeeze(mean(outProj(3,i,51:100,:),3)), squeeze(std(outProj(3,i,51:100,:),0,3))./sqrt(50))
                hold on
            end        
            set(gca,'fontsize',18)
            set(gca,'linewidth',2)
            box off
            ylim([-3 3])        
            set(gca,'xtick',[0 500 1500 2000])
            set(gca,'xticklabel',[0 0.5 1.5 2.0])
            xlabel('Time (s)')

            saveas(gca, "fig2_"+dirs{net}+".pdf", "pdf")
        end
    end
end
