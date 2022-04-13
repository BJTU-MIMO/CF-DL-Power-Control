% maxprod
load('CSI_CF_training_20UE_10000.mat');
load('output-minibatch-newloss-prod.mat');
 
SE_cvx_prod = zeros(K,nbrOfSetups);
SE_un_le_prod = zeros(K,nbrOfSetups);
SE_equal = zeros(K,nbrOfSetups);

for i = 1:nbrOfSetups
    [SE_cvx_prod(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),p_maxprod_cell(:,i),0.9,K);
    [SE_un_le_prod(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),pBest(:,i),0.9,K);
    [SE_equal(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),100*ones(K,1),0.9,K);
end


figure;
hold on; box on;


plot(sort(reshape(SE_cvx_prod,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'b-','LineWidth',2);
plot(sort(reshape(SE_un_le_prod,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'r-.','LineWidth',2);

xlabel('Spectral efficiency [bit/s/Hz]','Interpreter','Latex');
ylabel('CDF','Interpreter','Latex');
legend({'Max-prod-optimization','Max-prod-dl','Equal'},'Interpreter','Latex','Location','SouthEast');


    
%% predict_set
% maxprod
load('CSI_CF_predict_20UE_200.mat');
load('output-minibatch-newloss-prod.mat');
 
SE_cvx_prod = zeros(K,nbrOfSetups);
SE_un_le_prod = zeros(K,nbrOfSetups);
SE_equal = zeros(K,nbrOfSetups);

for i = 1:nbrOfSetups
    [SE_cvx_prod(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),p_maxprod_cell(:,i),0.9,K);
    [SE_un_le_prod(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),pBest_test(:,i),0.9,K);
    [SE_equal(:,i)] = computeSE(signal_cell(:,i),interference_cell(:,:,i),betaVal_cell(:,:,i),G_cell(:,:,i),100*ones(K,1),0.9,K);

end


figure;
hold on; box on;

plot(sort(reshape(SE_cvx_prod,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'b-','LineWidth',2);
plot(sort(reshape(SE_un_le_prod,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'r-.','LineWidth',2);
plot(sort(reshape(SE_equal,[K*nbrOfSetups 1])),linspace(0,1,K*nbrOfSetups),'k--','LineWidth',2);

xlabel('Spectral efficiency [bit/s/Hz]','Interpreter','Latex','fontsize',14);
ylabel('CDF','Interpreter','Latex','fontsize',14);
legend({'Max-prod-optimization','Proposed Max-prod-DL','Equal'},'Interpreter','Latex','Location','SouthEast','fontsize',14);
