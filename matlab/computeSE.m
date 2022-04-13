%% compute SE
function [SE ] = computeSE(signal,interference,betaVal,G_cell,rho,prelogFactor,K)

SINR = signal.*rho ./ (interference'*rho + 1);

SE = prelogFactor*log2(1+SINR);


end