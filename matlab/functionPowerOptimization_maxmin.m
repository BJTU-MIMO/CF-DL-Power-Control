function [SE,pBest] = functionPowerOptimization_maxmin(signal,interference,Pmax,prelogFactor)


%Extract the number of UEs
K = size(signal,1);


%Initalize the lower and upper rate limits
rateLower = 0;
rateUpper = log2(1+Pmax*min(signal));

%Set the accuracy of the bisection
delta = 0.01;

%Prepare to save the power solution
pBest = zeros(K,1);

%Solve the max-min problem by bisection - iterate until the difference
%between the lower and upper points in the interval is smaller than delta
while norm(rateUpper - rateLower) > delta
    
    %Compute the midpoint of the line
    rateCandidate = (rateLower+rateUpper)/2;
    
    %Transform the midpoints into SINR requirements
    SINRcandidate = 2.^(rateCandidate)-1; 
    
    %Solve the feasibility problem using CVX
    [feasible,pSolution] = functionFeasibilityProblem_cvx(signal,interference,Pmax,K,SINRcandidate);
    
    
    %If the problem was feasible, then replace rateLower with
    %gammaCandidate and store pSolution as the new best solution.
    if feasible
        rateLower = rateCandidate;
        pBest = pSolution;
    else
        %If the problem was not feasible, then replace ratePoint with
        %gammaCandidate
        rateUpper = rateCandidate;
    end
    
end

%Compute the SEs with the max-min solution
SE = computeSE(signal,interference,pBest,prelogFactor);



function [feasible,rhoSolution] = functionFeasibilityProblem_cvx(signal,interference,Pmax,K,SINRconstraint)


cvx_begin
cvx_quiet(true); %This suppresses screen output from the solver

variable rho(K);  %Power allocation of the users
variable scaling; %Scaling parameter for power constraints

minimize scaling %Minimize the power indirectly by scaling the power constraints

subject to

for k = 1:K
    
    %SINR constraints
    SINRconstraint*(sum(rho.*interference(:,k)) + 1) - (rho(k)*signal(k)) <= 0
    
    rho(k)>=0
    
    rho(k)<=scaling*Pmax
    
end


scaling >= 0; %Power constraints must be positive

cvx_end


%% Analyze the CVX output and prepare the output variables
if isempty(strfind(cvx_status,'Solved')) %Both the power minimization problem and the feasibility problem are infeasible
    feasible = false;
    rhoSolution = [];
elseif scaling>1 %Only the power minimization problem is feasible
    feasible = false;
    rhoSolution = rho;
else %Both the power minimization problem and feasibility problem are feasible
    feasible = true;
    rhoSolution = rho;
end


function SE = computeSE(signal,interference,rho,prelogFactor)

SINR = signal.*rho ./ (interference'*rho + 1);

SE = prelogFactor*log2(1+SINR);
