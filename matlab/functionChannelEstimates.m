function [Hhat,H,B] = functionChannelEstimates(R,nbrOfRealizations,L,K,N,tau_p,pilotIndex,p)
%% Generate channel realizations

%Generate uncorrelated Rayleigh fading channel realizations
H = (randn(L*N,nbrOfRealizations,K)+1i*randn(L*N,nbrOfRealizations,K));


%Go through all channels and apply the spatial correlation matrices
for l = 1:L
    
    for k = 1:K
        
        %Apply correlation to the uncorrelated channel realizations
        Rsqrt = sqrtm(R(:,:,l,k));
        H((l-1)*N+1:l*N,:,k) = sqrt(0.5)*Rsqrt*H((l-1)*N+1:l*N,:,k);
        
    end
    
end


%% Perform channel estimation

%Store identity matrix of size N x N
eyeN = eye(N);

%Generate realizations of normalized noise
Np = sqrt(0.5)*(randn(N,nbrOfRealizations,L,tau_p) + 1i*randn(N,nbrOfRealizations,L,tau_p));


%Prepare to store results
Hhat = zeros(L*N,nbrOfRealizations,K);

if nargout>2
    B = zeros(size(R));
end


%Go through all APs
for l = 1:L
    
    %Go through all pilots
    for t = 1:tau_p
        
        %Compute processed pilot signal for all UEs that use pilot t
        yp = sqrt(p)*tau_p*sum(H((l-1)*N+1:l*N,:,t==pilotIndex),3) + sqrt(tau_p)*Np(:,:,l,t);
        
        %Compute the matrix that is inverted in the MMSE estimator
        PsiInv = (p*tau_p*sum(R(:,:,l,t==pilotIndex),4) + eyeN);
        
        %Go through all UEs that use pilot t
        for k = find(t==pilotIndex)'
            
            %Compute the MMSE estimate
            RPsi = R(:,:,l,k) / PsiInv;
            Hhat((l-1)*N+1:l*N,:,k) = sqrt(p)*RPsi*yp;
            
            %Compute the spatial correlation matrix of the estimate
            if nargout>2
                B(:,:,l,k) = p*tau_p*RPsi*R(:,:,l,k);
            end
            
        end
        
    end
    
end
