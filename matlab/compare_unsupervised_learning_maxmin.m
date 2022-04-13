% close all;
clear;

load('CSI_CF_training.mat');
load('output-minibatch-newloss.mat');
load('output-minibatch-newloss-others.mat');

SE_cvx = zeros(K,nbrOfSetups);
SE_un_le = zeros(K,nbrOfSetups);
SE_maxmin_other = zeros(K,nbrOfSetups);

% maxmin
for i = 1:nbrOfSetups
    [SE_cvx(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),p_maxmin_cell(:,i),0.9,K);
    [SE_un_le(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),pBest(:,i),0.9,K);
    [SE_maxmin_other(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),pBest_compare(:,i),0.9,K);
end
    
figure;
hold on; box on;

plot(sort(reshape(SE_cvx,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'b-','LineWidth',2);
plot(sort(reshape(SE_un_le,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'r-.','LineWidth',2);
plot(sort(reshape(SE_maxmin_other,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'k--','LineWidth',2);

xlabel('Spectral efficiency of Max-min[bit/s/Hz]','Interpreter','Latex');
ylabel('CDF','Interpreter','Latex');
legend({'Max-min-optimization','Maxmin-dl','Maxmin-dl'},'Interpreter','Latex','Location','SouthEast');
xlim([0,2]);



%% predict_set
 clear;

load('CSI_CF_predict_8UE.mat');
load('output-minibatch-newloss.mat');
load('output-minibatch-newloss-others.mat');

SE_cvx = zeros(K,nbrOfSetups);
SE_un_le = zeros(K,nbrOfSetups);
SE_maxmin_other = zeros(K,nbrOfSetups);

for i = 1:nbrOfSetups
    SE_cvx(:,i) = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),p_maxmin_cell(:,i),0.9,K);
    SE_un_le(:,i) = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),pBest_test(:,i),0.9,K);
    SE_maxmin_other(:,i) = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),pBest_test_compare(:,i),0.9,K);
end

figure;
hold on; box on;
xlim([0 2]);
plot(sort(reshape(SE_cvx, [K*nbrOfSetups 1])), linspace(0,1,K*nbrOfSetups), 'b-', 'LineWidth', 2);
plot(sort(reshape(SE_un_le, [K*nbrOfSetups 1])), linspace(0,1,K*nbrOfSetups), 'r-.', 'LineWidth', 2);
plot(sort(reshape(SE_maxmin_other, [K*nbrOfSetups 1])), linspace(0,1,K*nbrOfSetups), 'k--', 'LineWidth', 2);
xlim([0 1.8]);
xlabel('Spectral efficiency of Max-min[bit/s/Hz]','Interpreter','Latex','fontsize',14);
ylabel('CDF','Interpreter','Latex','fontsize',14);
legend({'Max-min-optimization','Max-min-dl','Max-min-dl'},'Interpreter','Latex','Location','SouthEast','fontsize',14);
