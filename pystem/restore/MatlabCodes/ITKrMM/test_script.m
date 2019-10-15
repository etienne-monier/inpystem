%%%% itkrmm test script %%%%

rng(0);                  % reset random generator for reproducible results
savefile = 'mysave.mat'; % save to savefile if defined
display = 0;             % to display progress (best reduce N to 10000) 
cp2itkrm = 1;            % compare to svd + itkrm (not using mask info)

%%%%%%%%% parameter settings %%%%%%%%%%
runs=10;            % number of runs: 10 for paper
d = 256;            % signal dimension
maxitLR = 10;       % number of iterations for low rank component
NLR = 30000;        % number of signals for low rank component per it.
maxit = 100;        % number of iterations for DL (100 for rand init, 10 for 1:1 init)
N = 100000;         % number of signals for DL per it.
rho = 0.25/sqrt(d);      % noise level
             

%%%%% dictionary and low rank component:
%dicotype = 'dct';   % dictionary = dct, low rank = first dct atoms
dicotype = 'rnd';   % lrc = rand subspace, dico = K rand vectors in ortho complement
                    % default setting 'rnd'
K = 1.5*d;          % if 'rnd' and K+L=d, [lrc,dico] = random orthonormal basis
suppsize = 256;     % support size parameter for random low rank and dico

%%%%% low rank component size and coefficients:
lf1 = 'L';          lv1 =   2;     % low rank atoms: 0,1,2,4
lf2 = 'bL';         lv2 = 0.15;    % decay parameter
lf3 = 'rad';        lv3 = 1/3;     % radius of the sphere where signals are generated

paramLR = struct(lf1,lv1,lf2,lv2,lf3,lv3);

%%% sparse coefficients and scaling:
f1 = 'S';         v1 = 6;      % sparsity level
f2 = 'T';         v2 = v1;     % number of non zero coefficients
f3 = 'b';         v3 = 0.1;    % decay parameter
f4 = 's';         v4 = 4;      % scaling parameter for signals

paramSig = struct(f1,v1,f2,v2,f3,v3,f4,v4);

%%%% mask parameters
masktype = 'e';   %%% random erasures (default)
%masktype = 't';   %%% time-varying/burst errors

if masktype == 't'
    %%% time varying (burst error) parameters
    m1 = 'p0';         mv1 = .0;   % Probability of clean signal
    m2 = 'p1';         mv2 = .7;   % Probability of length ld damage
    m3 = 'p2';         mv3 = .3;   % Probability of length 2*ld damage
    m4 = 'ld';         mv4 = 64;   % Length of damage
    m5 = 'q';          mv5 = .7;   % Probability of a signal damage in the upper half
    
    paramMask = struct(m1,mv1,m2,mv2,m3,mv3, m4, mv4, m5, mv5);
else
    
    masktype = 'e'; %%% set to default
    %%% erasure parameters:
    m1 = 'p1';         mv1 = 0.7;   % Probability to be pixel-wise corrupted upper half
    m2 = 'p2';         mv2 = 0.5;   % Probability to be pixel-wise corrupted lower half
    m3 = 'q1';         mv3 = 0.7;   % Probability of a signal to be corrupted 
    m4 = 'q2';         mv4 = 0.5;   % Probability of a signal to be corrupted
     
    paramMask = struct(m1,mv1,m2,mv2,m3,mv3,m4,mv4);
end

%%%% initialisation parameters
inittype = 'r';  %%%% 'r' random (default),  'd' data
alpha = 1;       %%%% alpha initialisation + beta generating dictionary
beta = 0;

%%%%% end of parameter settings

%%% create dictionary and low rank component
%%% 'dico' contains lowrank component on first L columns and 
%%% sparsifying atoms on the remaining columns

if dicotype == 'dct'
    dico = idct(eye(d));  %%% first L entries low rank comp, rest dictionary
    K=d-paramLR.L; 
else
    if suppsize == d
        dico = randn(d,K+paramLR.L);
    else
        dico=zeros(d,K+paramLR.L);
        for iat=1:K+paramLR.L
            supp=randperm(d,suppsize);
            dico(supp,iat)=randn(suppsize,1);
        end
    end
       
    if K+paramLR.L==d
        [Ud,~,Vd]=svd(dico,'econ');
        dico=Ud*Vd';
    else
        lrc=dico(:,1:paramLR.L);    
        scale=sqrt(sum(lrc.*lrc));
        lrc = lrc*diag(1./scale);
        [Ulrc,~,Vlrc]=svd(lrc,'econ');
        lrconb=Ulrc*Vlrc';

        dico=dico-lrconb*(lrconb'*dico);
        dico(:,1:paramLR.L)=lrconb;
        scale=sqrt(sum(dico.*dico));
        dico=dico*diag(1./scale); 
    end
end
%%%% end of dictionary creation


%%%% generate signals for initialisations
if inittype == 'd'
    dinit = makesparselowsig(dico, runs*(K+paramLR.L), paramSig, paramLR, rho, 0);
    if masktype == 'e'
        mdum = RandMask(d, runs*(K+paramLR.L), paramMask);
    else
        mdum = maskTimeVar(d,runs*(K+paramLR.L), paramMask);
    end
    dinit = dinit.*mdum;        
else
    dinit=randn(d,runs*(K+paramLR.L));    
end
dinit = dinit*diag(1./sqrt(sum(dinit.*dinit)));
if paramLR.L>0
    dinitLR = dinit(:,1:(runs*paramLR.L));
    dinit=dinit(:,(runs*paramLR.L)+1:end);
end
%%%% end of initialisation signal creation

AllSigCor = zeros(runs, maxit);   %%% empirical average corruption
resultDict = zeros(maxit*runs,K);
resultLRS = zeros(maxitLR*runs,paramLR.L);

if cp2itkrm == 1;
    resultDictb = zeros(maxit*runs,K);
    resultLRSb = zeros(runs,1);
end

for i=1:runs
    
    %%% learn low rank component with new algo / svd 
    %%%% (in case of random init.)
    lrc = [];
    lrcb =[];
    if paramLR.L>0 && beta == 0
        
        %%%% using mask info 
        for jj = 1:paramLR.L
            
            %%%% initialisation for low rank component:
            atoml = dinitLR(:,(i-1)*paramLR.L+jj);
            if jj>1
                atoml = atoml-lrc*lrc'*atoml;
                atoml = atoml/norm(atoml);
            end
            
            for j = 1:maxitLR
                
                %%%% generate signals
                data = makesparselowsig(dico, NLR, paramSig, paramLR, rho, 0);
                %%%% generate masks
                if masktype == 't'
                    masks = maskTimeVar(d, NLR, paramMask);
                else
                    masks = RandMask(d, NLR, paramMask); 
                end
                %%%% corrupt signals
                data = data.*masks;
                
                %%%% one iteration low rank atom recovery
                atoml=rec_lratom(data,masks,lrc,1,atoml);
                
                resultLRS((i-1)*maxitLR + j,jj)= norm([lrc, atoml] - dico(:,1:paramLR.L)*dico(:,1:paramLR.L)'*[lrc,atoml]); 
            
                if display == 1;   
                    disp(['lrc error adapted atom ',num2str(jj),', it ', num2str(j),' = ', num2str(resultLRS((i-1)*maxitLR + j,jj))]);
                end
            end
            lrc = [lrc, atoml];
        end
        errLRS = norm(dico(:,1:paramLR.L) - lrc*lrc'*dico(:,1:paramLR.L));
        if display == 1
            disp(['final lrc error adapted = ', num2str(errLRS)]);
        end
        
        %%%% not using mask info == svd/eig
        if cp2itkrm == 1;
            [lrcb, ~] = eigs(data*data',paramLR.L);
            errLRSb = norm(dico(:,1:paramLR.L) - lrcb*lrcb'*dico(:,1:paramLR.L));
            resultLRSb(i) = errLRSb;
            if display == 1
                disp(['final lrc error unadapted = ',num2str(errLRSb)]);
            end
        end
    elseif paramLR.L>0  
        lrc = dico(:,1:paramLR.L);
        if cp2itkrm == 1;
            lrcb=lrc;
        end
    end
    %%% end of low rank comp reconstruction

    %%%% dictionary learning %%%  
    %%%% dictionary initialisation %%%
    Z = dinit(:,(i-1)*K+1:i*K);
   
    if beta==0
        %%% random initialisation:
        rdico = Z;   
    elseif alpha == 0
        %%% fairy godmother (true dico):
        rdico = dico(:,paramLR.L+1:end);            
    else
        for k=1:K
            Z(:,k)=Z(:,k)-(Z(:,k)'*dico(:,k))*dico(:,k);
            Z(:,k)=Z(:,k)/norm(Z(:,k));
        end
        dinitAlphaBeta = alpha * Z + beta*dico(:,paramLR.L+1:end);
        dinitAlphaBeta = dinitAlphaBeta*diag(1./sqrt(sum(dinitAlphaBeta.*dinitAlphaBeta)));
        rdico = dinitAlphaBeta;
    end
    
    if paramLR.L > 0
        rdico = rdico - lrc*lrc'*rdico;
        rdico = rdico*diag(1./sqrt(sum(rdico.*rdico)));
    end
    
    if cp2itkrm == 1
        rdicob=rdico;
        if paramLR.L > 0
            rdicob = rdicob - lrcb*lrcb'*rdicob;
            rdicob = rdicob*diag(1./sqrt(sum(rdicob.*rdicob)));
        end
    end
        
    
    %%% dictionary learning with ITKrMM / ITKrM
    for it=1:maxit;
        
        % generate signals
        data=makesparselowsig(dico, N, paramSig, paramLR, rho, 0);
        
        % generate masks
        if masktype == 't'
            masks = maskTimeVar(d,N,paramMask); % Time-varying masks
        else
            masks = RandMask(d,N,paramMask);   % Random Masks
        end
        
        % corrupt signals
        data = data.*masks;

      % Calculate the level of corruption:
        SigCorr = 100*(numel(masks) - nnz(masks))/numel(masks);
        AllSigCor(i, it) = SigCorr;

        %%%% Dictionary Learning %%%
        rdico=itkrmm(data,masks,K,paramSig.S, lrc, 1,rdico);
        resultDict((i-1)*maxit+it,:) = max(abs(rdico'*dico(:,paramLR.L+1:end)));
        
        if cp2itkrm == 1
            datab = data - lrcb*lrcb'*data;
            rdicob=itkrm(datab,K,paramSig.S, 1,rdicob); 
            datab=[];
            resultDictb((i-1)*maxit+it,:) = max(abs(rdicob'*dico(:,paramLR.L+1:end)));
        end
        
        if display == 1
            if cp2itkrm == 1;
                NoAt = sum(max(abs(rdico'*dico(:,paramLR.L+1:end)))>0.99);
                NoAtb = sum(max(abs(rdicob'*dico(:,paramLR.L+1:end)))>0.99);
                plot(1:K,resultDict((i-1)*maxit+it,:),'rx',1:K,resultDictb((i-1)*maxit+it,:),'bx');
                title(['Iteration ',num2str(it),': atoms found adap/unad. = ',num2str(NoAt),'/',num2str(NoAtb)]);
                drawnow;
            else
                NoAt = sum(max(abs(rdico'*dico(:,paramLR.L+1:end)))>0.99);
                plot(1:K,resultDict((i-1)*maxit+it,:),'rx');
                title(['Iteration ',num2str(it),': atoms found adap. = ',num2str(NoAt)]);
                drawnow;
            end
        end
    end
    %%%% end of dictionary recovery
    
   if exist('savefile','var') 
       data=data(:,1:100);   %%% reduce sizes to save whole workspace
       masks=masks(:,1:100);
       save(savefile); 
   end
end

AvSigCor = mean(mean(AllSigCor,2));
    