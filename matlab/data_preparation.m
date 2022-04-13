%% training_set
close all;
clear;
tic;

%% Define simulation setup

%Number of Monte Carlo setups
nbrOfSetups = 1000; %200;

%Number of channel realizations per setup
nbrOfRealizations = 100;

%Number of APs in the cell-free network
L = 20;

%Number of UEs
K = 8;

%Number of antennas per AP
N = 1;

%Length of the coherence block
tau_c = 200;

%Number of pilots per coherence block
tau_p = 20;

%Uplink transmit power per UE (mW)
p = 100;


%Prepare to save simulation results

betaVal_cell = zeros(L,K,nbrOfSetups);
pilotIndex_cell = zeros(K,nbrOfSetups);
signal_cell = zeros(K,nbrOfSetups);
interference_cell = zeros(K,K,nbrOfSetups);
G_cell = zeros(K,K,nbrOfSetups);

p_maxmin_cell = zeros(K,nbrOfSetups);
p_maxprod_cell = zeros(K,nbrOfSetups);

%% Go through all setups
for n = 1:nbrOfSetups
    
    %Display simulation progress
    disp(['Setup ' num2str(n) ' out of ' num2str(nbrOfSetups)]);
    
    %Generate one setup with UEs at random locations
    [gainOverNoisedB,R,pilotIndexCF,pilotIndexSC] = generateSetup_threeslope(L,K,N,tau_p,1,p);
    betaVal = db2pow(gainOverNoisedB);
    
    
    %Generate channel realizations, channel estimates, and estimation
    %error correlation matrices for all UEs to the APs 
    [Hhat_AP,H_AP,B_AP] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndexCF,p);
    
    
    %Extract terms in the numerator and denominator of the SINRs
    [signalCF,interferenceCF,signalSC,interferenceSC,signalSC2,interferenceSC2] = functionSINRterms_uplink_ngo(p,L,K,tau_p,pilotIndexCF,pilotIndexSC,betaVal);
    signal_cell(:,n) = signalCF;
    interference_cell(:,:,n) = interferenceCF;
    
    % maxmin power 
    [~,p_maxmin] = functionPowerOptimization_maxmin(signalCF,interferenceCF,p,1);
    p_maxmin_cell(:,n) = p_maxmin;
    
    % maxprod power
    [~,p_maxprod] = functionPowerOptimization_prodSINR(signalCF,interferenceCF,p,1);
    p_maxprod_cell(:,n) = p_maxprod;
    
    %betaVal_re = reshape(betaVal,[],1);
    %pBest_CF_re = reshape(pBest_CF_sumrate,[],1);
    
    betaVal_cell(:,:,n) = betaVal;
    %pBest_CF_cell(:,n) = pBest_CF_re;
    pilotIndex_cell(:,n) = pilotIndexCF;
    for row = 1:K
        for col = 1:K
            if row == col
                continue
            end
            G_cell(row,col,n) = sum(betaVal(:,row).*betaVal(:,col));
        end
    end
end

%define input of NN
sumVal = zeros(K,nbrOfSetups);

for t = 1:nbrOfSetups
    
    a = (sum(betaVal_cell(:,:,t),1))';
    
    for n = 1:K
      sumVal(n,t) = a(n,1);
    end
end

nome_file='CSI_CF_training';
save(nome_file,'-v7.3');
   
toc;


%% Predict_set

%Number of Monte Carlo setups
nbrOfSetups = 200; %200;

%Number of channel realizations per setup
nbrOfRealizations = 100;

%Number of APs in the cell-free network
L = 20;

%Number of UEs
K = 8;

%Number of antennas per AP
N = 1;

%Length of the coherence block
tau_c = 200;

%Number of pilots per coherence block
tau_p = 20;

%Uplink transmit power per UE (mW)
p = 100;


%Prepare to save simulation results

betaVal_cell = zeros(L,K,nbrOfSetups);
pilotIndex_cell = zeros(K,nbrOfSetups);
signal_cell = zeros(K,nbrOfSetups);
interference_cell = zeros(K,K,nbrOfSetups);
G_cell = zeros(K,K,nbrOfSetups);

p_maxmin_cell = zeros(K,nbrOfSetups);
p_maxprod_cell = zeros(K,nbrOfSetups);

%% Go through all setups
for n = 1:nbrOfSetups
    
    %Display simulation progress
    disp(['Setup ' num2str(n) ' out of ' num2str(nbrOfSetups)]);
    
    %Generate one setup with UEs at random locations
    [gainOverNoisedB,R,pilotIndexCF,pilotIndexSC] = generateSetup_threeslope(L,K,N,tau_p,1,p);
    betaVal = db2pow(gainOverNoisedB);
    
    
    %Generate channel realizations, channel estimates, and estimation
    %error correlation matrices for all UEs to the APs 
    [Hhat_AP,H_AP,B_AP] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndexCF,p);
    
    
    %Extract terms in the numerator and denominator of the SINRs
    [signalCF,interferenceCF,signalSC,interferenceSC,signalSC2,interferenceSC2] = functionSINRterms_uplink_ngo(p,L,K,tau_p,pilotIndexCF,pilotIndexSC,betaVal);
    signal_cell(:,n) = signalCF;
    interference_cell(:,:,n) = interferenceCF;
    
    % max-min power control
    [~,p_maxmin] = functionPowerOptimization_maxmin(signalCF,interferenceCF,p,1);
    p_maxmin_cell(:,n) = p_maxmin;
    
    % maxprod power
    [~,p_maxprod] = functionPowerOptimization_prodSINR(signalCF,interferenceCF,p,1);
    p_maxprod_cell(:,n) = p_maxprod;
    
    %betaVal_re = reshape(betaVal,[],1);
    %pBest_CF_re = reshape(pBest_CF_sumrate,[],1);
    
    betaVal_cell(:,:,n) = betaVal;
    %pBest_CF_cell(:,n) = pBest_CF_re;
    pilotIndex_cell(:,n) = pilotIndexCF;
    for row = 1:K
        for col = 1:K
            if row == col
                continue
            end
            G_cell(row,col,n) = sum(betaVal(:,row).*betaVal(:,col));
        end
    end
   
end

sumVal_pr = zeros(K,nbrOfSetups);

for t = 1:nbrOfSetups
    
    a = (sum(betaVal_cell(:,:,t),1))';
    
    for n = 1:K
      sumVal_pr(n,t) = a(n,1);
    end
end
    
nome_file='CSI_CF_predict';
save(nome_file,'-v7.3');
