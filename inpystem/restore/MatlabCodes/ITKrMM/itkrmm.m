function [dico, E]=itkrmm(data,masks,K,S,dicoL,maxit,dinit,verbose, Xref)

% syntax: dico=itkrmm(data,masks,K,S,dicoL,maxit,dinit)
%
% Iterative Thresholding& K residual Means masked
% as described in the new paper 
%
% input:
% data... d x N matrix containing the (corrupted)
%                                     training signals as its columns
% masks ... d x N matrix containing the masks as its columns
%                    - masks(.,.) in {0,1} - default masks = 1                              
% K... number of dictionary atoms/dictionary size - default d
% S... desired/estimated sparsity level of the signals - default 1
% dicoL...orthobasis for low rank component - default []
% maxit... number of iterations - default 50
% dinit... initialisation, d x K unit norm column matrix - default random 
% verbose ... 1 to send output information. 0 else. (Default is 1).
%
% output:
% dico....d x K dictionary 
%        
%
% last modified 09.08.16
% Karin Schnass 

%%%%%% preparations

if(nargin < 1)
    disp('syntax: dico=itkrmm(data,masks,K,S,dicoL,maxit,dinit)');
    dico=[];
    return;
end

[d,N]=size(data);

if (nargin < 2)
    masks=ones(d,N);
end

data=data.*masks; % safeguard against the massimo effect

if(nargin < 3)
    K=d;
end

if(nargin < 4)
    S=1;
end

if(nargin < 5)
    dicoL=[];
end

[~,L]=size(dicoL);

if (N < K+1)
    disp('less training signals than atoms => trivial solution');
    dico=data;
    return;
end

if(nargin < 6)
    maxit = 50;
end

if (nargin < 7) 
    dinit = randn(d,K); 
    scale = sum(dinit.*dinit);
    dinit=dinit*diag(1./sqrt(scale)); 
end

if (nargin < 8) 
    verbose = 1;
end

if (nargin < 9) 
    Xref = 0;
end

if size(dinit)~=[d,K]
    disp('initialisation does not match dictionary size - random initialisation used');	
    dinit = randn(d,K); 
    scale = sum(dinit.*dinit);
    dinit=dinit*diag(1./sqrt(scale));
end

if L > 0
    dinit=dinit-dicoL*dicoL'*dinit;
    scale = sum(dinit.*dinit);
    dinit=dinit*diag(1./sqrt(scale));   
end

%%% subtract lr comp from data
if L > 0
    for n=1:N
        dicoLMn = dicoL.*(masks(:,n)*ones(1,L));
        data(:,n) = data(:,n) - dicoLMn*(pinv(dicoLMn)*data(:,n));
    end
end

%%% learn dictionary %%%%%
dold=dinit;

rng(1)

for it=1:maxit
    
    if verbose == 1
        if it == 1
            disp(['Iteration #' num2str(it) ' over ' num2str(maxit)])
        else
            disp(['Iteration #' num2str(it) ' over ' num2str(maxit) ' (estimated remaining time: ' sec2str(time_t*(maxit-it+1)) ')'])
        end
    end
    
    start_t = tic;

    dnew=zeros(d,K);
    maskweight = zeros(d,K);
    
    %ip=dold'*data;
    
    for n=1:N
            
        %%%% inner products with renormalised corrupted dico
        msupp=find(masks(:,n)>0);
        doldm_2 = sqrt(sum(dold(msupp,:).*dold(msupp,:)));
        ipn=dold'*data(:,n)./doldm_2';
        
        %%%% find support
        absipn=abs(ipn);
        signipn=sign(ipn);
        [~,In] = sort(absipn,1,'descend');
        
        %%%% renormalised corrupted dico on support
        dInm=(dold(:,In(1:S)).*(masks(:,n)*ones(1,S)))*diag(1./doldm_2(In(1:S))); 
        
        %%%% construct residual
        if (L>0)
            dicoLMn = dicoL.*(masks(:,n)*ones(1,L));
            dILnm=[dicoLMn,dInm];
            resn=real(data(:,n)-pinv(dILnm)'*[zeros(L,1); ipn(In(1:S))]);
        else    
            resn=real(data(:,n)-pinv(dInm)'*ipn(In(1:S)));
        end    
        
        %%%% update new dictionary and maskweight
        dnew(:,In(1:S))=dnew(:,In(1:S))+ resn*(signipn(In(1:S))'.*ones(1,S));
        dnew(:,In(1:S))=dnew(:,In(1:S))+ dInm*diag(absipn(In(1:S)));
        maskweight(:,In(1:S))=maskweight(:,In(1:S))+ masks(:,n)*ones(1,S);
         
    end
        
    if min(min(maskweight)) > 0
        dnew=dnew./maskweight*N;
    else
        dnew=dnew./(maskweight + 0.001)*N;
    end
    
    if L>0 
        dnew=dnew-dicoL*dicoL'*dnew;
    end
    scale=sum(dnew.*dnew); 
    %%% redraw atoms that are not used
    iszero=find(scale < 0.00001);
    dnew(:,iszero)=randn(d,length(iszero));
    scale(iszero)=sum(dnew(:,iszero).*dnew(:,iszero));
    %%% normalise
    dnew = dnew*diag(1./sqrt(scale));
    dold = dnew;

    %% Compute error

    if numel(Xref) ~= 1
        lrcdico = [dicoL, dold];
        coeff = OMPm(lrcdico, data, masks, S);
        inppatches = lrcdico*coeff;
        
        E(it) = sqrt(sum(sum(masks.*(Xref - inppatches).^2))/prod(size(data)));
    end

    time_t = toc(start_t);
   
end

dico=dold;
