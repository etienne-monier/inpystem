function mask = RandMask(d, N, paramMask)

% syntax: mask=RandMask(d, N, paramMask)
%
% Creation of the random mask with probabilities of a pixel / signal
% to be corrupted
%
% Input:
% d...  signal dimension
% N ... number of signals
% paramMask.. struct with parameters:
%   p1.. Probability pixel-wise corruption in the upper half of the signal
%   p2.. Probability pixel-wise corruption in the lower half of the signal
%   q1, q2..Probability of a signal corruption (P(q1) = P(q2) = 1/2)
% Output:
% mask....d x N mask 

% Safe Checking
if nargin < 3
    disp('Error: Check your input');
    mask=[];		
    return;
end

if ~isfield(paramMask,'p1') 
    paramMask.p1=1;
end

if ~isfield(paramMask,'p2') 
    paramMask.p2=1;
end

if ~isfield(paramMask,'q1') 
    paramMask.q1=1;
end

if ~isfield(paramMask,'q2') 
    paramMask.q2=1;
end

% Probability Across the signal
qProb = rand(1, N);
temp = qProb<=1/2;
qProb(temp) = paramMask.q1;
qProb(~temp) = paramMask.q2;

pProb = zeros(d,1);
pProb(1:ceil(d/2)) = paramMask.p1;
pProb(ceil(d/2)+1:end) = paramMask.p2;

mask = rand(d,N);
masksProb = pProb*qProb;
mask = ceil(masksProb-mask);

end
