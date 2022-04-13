function [SE, rhoBest ] = functionPowerOptimization_sumrate(signal,interference,Pmax,prelogFactor)


%Extract number of UEs
K = size(signal,1);


%% Solve geometric program in (7.8) using CVX
cvx_begin gp
cvx_quiet(true); % This suppresses screen output from the solver

variable rho(K);
variable c(K);

minimize sum(c)

subject to

for k = 1:K

    if signal(k)>0
        %SINR constraints of UE k in cell j
        c(k)*(sum(rho.*interference(:,k)) + 1) <= (rho(k)*signal(k));

        rho(k) >= 0;

    else
        %This applies if UE k in cell j is inactive
        c(k) == 1;
        rho(k) >= 0;

    end
    
rho(k) <= Pmax;

end

cvx_end

%% Analyze the CVX output and prepare the output variables

if ~strfind(cvx_status,'Solved') %The problem was not solved by CVX, for some reason, and we then consider equal power allocation
    rhoSolution = Pmax*ones(K);
else %The problem was solved by CVX
    rhoSolution = rho;
end

%Refine the solution obtained from CVX using the Matlab command fmincon.
options = optimoptions('fmincon','Display','off','Algorithm','interior-point','MaxFunEvals',50000,'MaxIter',5000);
xend = fmincon(@(x) -sum(computeSE(signal,interference,x,prelogFactor)),rhoSolution(:),[],[],[],[],zeros(K,1),100*ones(K,1),[],options);
rhoBest = reshape(xend,[K 1]);

%Compute the SEs using Theorem 4.6
SE = computeSE(signal,interference,rho,prelogFactor);


function SE = computeSE(signal,interference,rho,prelogFactor)

SINR = signal.*rho ./ (interference'*rho + 1);

SE = prelogFactor*log2(1+SINR);

