function dico=itkrm(data,K,S,maxit,dinit)

% syntax: dico=itkrm(data,K,S,maxit,dinit)
%
% Iterative Thresholding& K residual Means 
% dictionary learning algorithm as described in
% 'Convergence radius and sample complexity 
%  of ITKM algorithms for dictionary learning
% arXiv: 1503.07027
%
% input:
% data... d x N matrix containing the training signals as its columns
% K... number of dictionary atoms/dictionary size - default d
% S... desired/estimated sparsity level of the signals - default 1
% maxit... number of iterations - default 100
% dinit... initialisation, d x K unit norm column matrix - default random 
%
% output:
% dico... d x K dictionary 
%
% last modified 09.12.16 
% Karin Schnass


%%%% preparations
if(nargin < 1)
    disp('syntax: dico=itkrm(data,K,S,maxit,dinit)');
    dico=[];
    return;
end

[d,N]=size(data);

if(nargin < 2)
    K=d;
end

if (N < K+1)
    disp('less training signals than atoms => trivial solution');
    dico=data;
    return;
end

if(nargin < 5) 
    dinit = randn(d,K); 
    scale = sum(dinit.*dinit);
    dinit=dinit*diag(1./sqrt(scale));  	
end

if(nargin < 4)
    maxit = 100;
end

if(nargin < 3)
    S=1;
end

if size(dinit)~=[d,K]
    disp('initialisation does not match dictionary size - random initialisation used');	
    dinit = randn(d,K); 
    scale = sum(dinit.*dinit);
    dinit=dinit*diag(1./sqrt(scale));
end
	

%%%% algorithm

dold=dinit;

for it=1: maxit
    
    ip=dold'*data;
    absip=abs(ip);
    signip=sign(ip);
    [sortip,I] = sort(absip,1,'descend');
    gram=dold'*dold;
    dnew=zeros(d,K);
    for n=1:N
        res=(data(:,n)-dold(:,I(1:S,n))*pinv(gram(I(1:S,n),I(1:S,n)))*ip(I(1:S,n),n));
        dnew(:,I(1:S,n))=dnew(:,I(1:S,n)) + real(res*signip(I(1:S,n),n)');
        dnew(:,I(1:S,n))=dnew(:,I(1:S,n)) + dold(:,I(1:S,n))*diag(absip(I(1:S,n),n));
    end
    scale=sum(dnew.*dnew); 
    %%% redraw atoms that are not used
    iszero=find(scale < 0.00001);
    dnew(:,iszero)=randn(d,length(iszero));
    scale(iszero)=sum(dnew(:,iszero).*dnew(:,iszero));
    %%% normalise
    dnew = dnew*diag(1./sqrt(scale));
    dold = dnew;
      
end

dico=dold;
