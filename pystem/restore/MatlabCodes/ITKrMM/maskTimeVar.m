function mask = maskTimeVar(d, N, paramMask)

% syntax: mask=maskTimeVar(d, N, paramMask)
%
% Creation of the mask with time-varying damage inside the signal
%
% Input:
% d...  signal dimension
% N ... number of signals
% paramMask.. struct with parameters:
%   p0..Probability of clean signal
%   p1..Probability of length ld damagel
%   p2..Probability of length 2*ld damage
%   ld..Length of damage
%   q..Probability of a signal damage in the upper half
% Output:
% mask....d x N mask

% Safe Checking
if ~isfield(paramMask,'p0') 
    paramMask.p0=0.3;
end

if ~isfield(paramMask,'p1') 
    paramMask.p1=0.3;
end

if ~isfield(paramMask,'p2') 
    paramMask.p2 = 1 - paramMask.p1 - paramMask.p0;
end

if (d < paramMask.ld) 
    disp('Error: length of the damage should be smaller than dimension');
    mask=[];		
    return;
end

if (d<2*paramMask.ld) 
    disp('Warning: length of the damage would be uniform');
    paramMask.p1 = paramMask.p1 + paramMask.p2;		
end

if (paramMask.p1 == 1) 
   disp('Warning: All signals will be damaged');
   paramMask.p2 = 0;
   paramMask.p0 = 0;
end

if (paramMask.p0 == 1) 
   disp('Warning: All signals will be clean');
   paramMask.p1 = 0;
   paramMask.p2 = 0;
end

if (paramMask.p2 == 1) 
   disp('Warning: All signals will be damaged by signal with length 2');
   paramMask.p0 = 0;
   paramMask.p1 = 0;
end

if (paramMask.p1 + paramMask.p2 + paramMask.p0 ~=1)
    paramMask.p2 = 1-paramMask.p0-paramMask.p1;
end

%%
mask = ones(d,N);

% Probability Across the signal:
% 0.. not damaged signal
% 1.. damaged signal with length t
% 2.. damaged signal by length 2t
qProb = rand(1, N);
qProb(qProb<=paramMask.p0) = 0;
qProb(qProb > (paramMask.p1+paramMask.p0)) = 2;
qProb(qProb<(paramMask.p1+paramMask.p0) & qProb>paramMask.p0) = 1;

% Start Location of the damage, randomly generated
qProbPos = rand(1, N);
qProbPos(qProbPos>paramMask.q) = 0;
qProbPos = ceil(qProbPos);

qPos = ones(1,N);
qPos(qProbPos==1) = randi([1,d/2], 1, nnz(qProbPos));
qPos(qProbPos==0) = randi([d/2+1,d], 1, N-nnz(qProbPos));

qPos(qProb==0) = 0;  % Remove damage at clean signals

% Insert damage of length t
tt = qPos(qProb==1);
%ind = zeros(paramMask.ld, numel(tt));
  col = find(qProb==1);
  
for i =1:numel(tt)
    if tt(i)+paramMask.ld <= d 
        ind = tt(i):(tt(i)+paramMask.ld-1);
    else
        temp = d-tt(i);
        ind = [1:(paramMask.ld-temp-1) tt(i):d];
    end
  
    mask(ind, col(i)) = zeros(paramMask.ld,1);    
end


% Insert damage of length 2t

t2 = qPos(qProb==2);
% ind2 = zeros(2*paramMask.ld, numel(t2));
    col2 = find(qProb==2);
    
for i =1:numel(t2)
    if t2(i)+2*paramMask.ld <= d 
        ind2 = t2(i):(t2(i)+2*paramMask.ld-1);
    else
        temp = d-t2(i);
        ind2 = [1:(2*paramMask.ld-temp-1) t2(i):d];
    end
    mask(ind2, col2(i)) = zeros(2*paramMask.ld,1);  
end


end
