function [signalCF,interferenceCF,signalSC,interferenceSC,signalSC2,interferenceSC2] = functionSINRterms_uplink_ngo(Pmax,L,K,tau_p,pilotIndexCF,pilotIndexSC,betaVal)
%% Computations for Cell-free mMIMO

%Compute the gamma parameters for all APs and UE according to (8) in [15]
gammaVal = zeros(L,K);

for l = 1:L
    for k = 1:K
        gammaVal(l,k) = Pmax*tau_p*(betaVal(l,k))^2 ./ (Pmax*tau_p*sum(betaVal(l,pilotIndexCF(k) == pilotIndexCF)) + 1);
    end
end

%Compute the numerator of (27) in [15] for all UEs, divided by the noise
%term and neglecting the transmit power 
signalCF = sum(gammaVal,1)';

%Compute the interference terms in the denominator of (27) in [15] for all
%UEs, neglecting the transmit powers
interferenceCF = zeros(K,K);

for k = 1:K
    
    noiseTerm = sum(gammaVal(:,k));
    
    %Add the last two terms of the denominator
    interferenceCF(:,k) = interferenceCF(:,k) + sum(repmat(gammaVal(:,k),[1 K]).*betaVal,1)'/noiseTerm;
    
    %Compute the first term with interference due to pilot contamination
    coPilot = (pilotIndexCF(k) == pilotIndexCF); %Extract UEs that use the same pilot
    coPilot(k) = false; %Remove the UE of interest
    samePilot = find(coPilot);
    
    for ind = 1:length(samePilot)
        interferenceCF(samePilot(ind),k) = interferenceCF(samePilot(ind),k) + sum(gammaVal(:,k).*betaVal(:,samePilot(ind))./betaVal(:,k))^2 /noiseTerm;
    end
    
end



%% Computations for Small cells - largest large-scale fading AP association

%Compute the numerator of (48) in [15] for all UEs, divided by the noise
%term and neglecting the transmit power 
signalSC = zeros(K,1);

%Compute the interference terms in the denominator of (48) in [15] for all
%UEs, neglecting the transmit powers
interferenceSC = zeros(K,K);

%All APs are available to serve UEs in the beginning
availableAPs = true(L,1);

%Prepare to store which AP that serves which UE
bestAP = zeros(K,1);

%Go through the UEs in "random" order
for k = 1:K
    
    %Select the AP to serve UE k using (38) in [15]
    activeAPs = find(availableAPs);
    [~,ind] = max(betaVal(activeAPs,k));
    m_k = activeAPs(ind);
    bestAP(k) = m_k;
    
    %Set the AP as unavailable
    availableAPs(m_k) = false;
    
    %Compute the signal and interference in (48), using (49)
    Psi_tkl = (Pmax*tau_p*sum(betaVal(m_k,pilotIndexSC(k) == pilotIndexSC)) + 1);
    omega_mkk = Pmax*tau_p*(betaVal(m_k,k))^2 ./ Psi_tkl;
    
    signalSC(k) = omega_mkk;
    interferenceSC(k,k) = interferenceSC(k,k) - omega_mkk;
    interferenceSC(:,k) = interferenceSC(:,k) + betaVal(m_k,:)';

end


%% Computations for Small cells - largest SINR AP association

%Compute the numerator of (48) in [15] for all UEs, divided by the noise
%term and neglecting the transmit power 
signalSC2 = zeros(K,1);

%Compute the interference terms in the denominator of (48) in [15] for all
%UEs, neglecting the transmit powers
interferenceSC2 = zeros(K,K);

%Prepare to store which AP that serves which UE
bestAP2 = zeros(K,1);

%Go through the UEs
for k = 1:K
    
    signal = zeros(L,1);
    interf = zeros(K,L);
    
    %Go through all APs
    for m_k = 1:L
        
        %Compute the signal and interference in (48), using (49)
        Psi_tkl = (Pmax*tau_p*sum(betaVal(m_k,pilotIndexSC(k) == pilotIndexSC)) + 1);
        omega_mkk = Pmax*tau_p*(betaVal(m_k,k))^2 ./ Psi_tkl;
        
        signal(m_k) = omega_mkk;
        
        interf(k,m_k) = interf(k,m_k) - omega_mkk;
        interf(:,m_k) = interf(:,m_k) + betaVal(m_k,:)';
        
    end
    
    
    %Find the AP that gives the highest SINR with full power transmission
    SINRs = signal./(sum(interf,1)'+1/Pmax);
    [~,selectAP] = max(SINRs);
    
    %Assign the UE to that AP and store signal and interference terms
    bestAP2(k) = selectAP;
    signalSC2(k) = signal(selectAP);
    interferenceSC2(:,k) = interf(:,selectAP);
    
end

