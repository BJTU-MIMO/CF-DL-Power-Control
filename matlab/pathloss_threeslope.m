function PL = pathloss_threeslope(dvec)


%Set distances in the three-slope model
d0 = 10; %meter
d1 = 50; %meter

%Constant term in the model from (53) in [15]
L = 140.7151;

%Compute the pathloss using the three-slope model in (52) in [15]
PL = zeros(length(dvec),1);

for ind = 1:length(dvec)
    
    d = dvec(ind);
    
    if d<=d0
        PL(ind) = -L -15*log10(d1/1000) -20*log10(d0/1000);
    elseif d<=d1
        PL(ind) = -L -15*log10(d1/1000) -20*log10(d/1000);
    else
        PL(ind) = -L -35*log10(d/1000);
    end
    
end
