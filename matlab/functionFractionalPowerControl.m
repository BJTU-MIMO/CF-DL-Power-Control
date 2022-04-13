function [p] = functionFractionalPowerControl(beta,MatA,pmax,theta)

[L,K] = size(MatA);
%Large-scale fading coefficients that count
beta_count = beta.*MatA;

%Sum of the large-scale fading coefficients that count for all UEs
SumOfBetaPerUE = (sum(beta_count, 1)).^theta;

scaling = min(SumOfBetaPerUE);

p = (pmax*scaling./SumOfBetaPerUE)';
end





