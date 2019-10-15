function sigN=makesparselowsig(dico, N, paramSig, paramLR,rho,bfix)

% syntax: sigN = makesparsesig(dico,N,paramSig,paramLR,rho,bfix)
%
%
% input:
% dico... generating dictionary
% N... number of sparse signals to create
% ParamSig.. struct containing the following values:
%   S... effective sparsity - number of strong coefficients - default 1
%   T... number of non-zero coefficients T>=S - default T=S
%   b... decay parameter for coefficients - default b=0
%   s.. scaling parameter, scale \in [1,s]
% ParamLR.. struct containing the following values:
%   L.. # Low rank component
%   bL... decay parameter for coefficients - default b=0
%   rad..the sphere radius on which x_0 for low rank component is generated
% 
% rho... noiselevel - default - rho=0
%
% output:
% sigN... d x N matrix with N sparse signals as columns
%
% Karin Schnass 24.01.14
% modified by VN 21.02.16

if (nargin<4)	
    disp('syntax: sigN=makesparsesig(dico,N,paramSig,paramLR,rho,bfix)');
    sigN=[];		
    return;
end

inputflaw=false;

if ~isfield(paramSig,'S')
   paramSig.S=1;
end

if ~isfield(paramSig,'T')
    paramSig.T=paramSig.S;
end

if ~isfield(paramSig,'s')
    paramSig.s=2;
end	

if ~isfield(paramSig,'b')
    paramSig.b=0;
end	

if ~isfield(paramLR,'L')
   paramLR.L = 0;
   paramLR.rad = 0;
end

if ~isfield(paramLR,'bL')
   paramLR.bL=0;
end

if ~isfield(paramLR,'rad')
   paramLR.rad=1/8;
end

if nargin < 5
   rho=0;
end

if (nargin<6)
    bfix=1;
end

if round(paramSig.S)<1
    paramSig.S=max([round(abs(paramSig.S)),1]);
    inputflaw=true;
end

if paramSig.T<paramSig.S
    inputflaw=true;
    paramSig.T=paramSig.S;
end

if (paramSig.b<0) || (paramSig.b>1)
    inputflaw=true;
    paramSig.b=0;
end

S = paramSig.S;
T = paramSig.T;
b = paramSig.b;
s = paramSig.s;

L = paramLR.L;
bL = paramLR.bL;
rad = paramLR.rad;


[d, K]=size(dico);

if S>d-1
   inputflaw=true;
   S=d-1;
end

if T>K;
   inputflaw=true;
   T=K;
end


if inputflaw==true
   disp('warning, strange input parameters, used: [d,S,T,b,rho]=');
   [d,S,T,b,rho]
end

sigN=[];

for n=1:N
    if bfix == 1
        beta=1-b;
        betaL = 1-bL;
    else
        beta=1-b *rand(1,1);
        betaL = 1-bL*rand(1,1);
    end
    %% Generate Signal for Low rank component
    if L > 0
        
     if betaL < 1
          x1toL = sqrt(1./L)*betaL.^[1:L]';
      x1toLsign = 2*round(rand(L,1))-1;
          x1toL = x1toL.*x1toLsign;
     else
          x1toL = randn(L,1);  
     end
     
        x1toL = rad*x1toL/norm(x1toL);
          sig = dico(:,1:L)*x1toL;     
    
    else
        sig=zeros(d,1);
    end
    
    x1toS=sqrt(1./S)*beta.^[1:S]';
    x1toSsign=2*round(rand(S,1))-1;
    x1toS=x1toS.*x1toSsign;
    
    %norm(x1toS)
    if T > S
        xSp1toT=randn(T-S,1);
        xSp1toT=xSp1toT* sqrt(1-norm(x1toS)^2)/norm(xSp1toT);
        x1toT=sqrt(1-rad^2)*[x1toS; xSp1toT];
    else
        x1toT= sqrt(1-rad^2)*x1toS/sqrt(norm(x1toS)^2);
    end
    p=randperm(K-L);
  
    sig= sig + dico(:,L+p(1:T))*x1toT;
   
  
    if (nargin == 7)
        noise = rho*randn(d,1);
        sig=(sig+noise)/sqrt(1+noise'*noise);
    end     
    
    scale = 1+rand(1,1)*(s-1);
    sig=sig*scale;
    
    sigN=[sigN,sig];  
    
end

