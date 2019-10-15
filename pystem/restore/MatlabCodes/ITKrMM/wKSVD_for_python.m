%% Load data

corrpatches = double(corrpatches);
maskpatches = double(maskpatches);

d = double(d);
K = double(K);
L = double(L);
S = double(S);
X = double(X);

init = double(init);

maxit = double(maxit);
maxitLR = double(maxitLR);

addpath('..')


%% learn low rank atoms 

lrc = [];
if verbose == 1 && L>0;
    disp('Learning low rank component using mask info');
end

for ll = 1 : L
    % initialise low rank atom
    inatoml = init(:,ll);
    if ll > 1
        inatoml = inatoml - lrc*lrc'*inatoml;
    end
    inatoml = inatoml/norm(inatoml);
    atoml = rec_lratom(corrpatches,maskpatches,lrc,maxitLR,inatoml);
    lrc = [lrc, atoml];     
end

%% learn dictionary with wKSVD

if verbose == 1;
    disp('learning wksvd dictionary');
    param.displayProgress = 1;
else
    param.displayProgress = 1;
end

param.K = K;              % dictionary size 
param.dSparsity = S; 
param.numIteration = maxit;    
param.InitializationMethod = 'GivenMatrix';

tic

indico = init(:,(L+1):(L+K-1));

if L == 1;
    param.preserveDCAtom = 1; %%% algo ensures orthogonality of initialisation to lrc         
elseif L > 1;
    param.preserveDCAtom = 1; 
    indico = indico - lrc*lrc'*indico; 
end

indico = indico*diag(1./sqrt(sum(indico.*indico)));

param.initialDictionary = [lrc,indico]; 
param.displayProgress = 1;
param.Xref = X;

[dico, output] = wKSVD(corrpatches, maskpatches, param); 
E = output.totalerr;

time = toc;

%% inpainting

if verbose == 1;
    disp('inpainting with wksvd dictionary');
end

coeff_wksvd = OMPm(dico, corrpatches, maskpatches, S);
outdata = dico*coeff_wksvd;


%% Save results

if verbose == 1;
    disp('Saving data ...')
end

save(outName,'outdata','time','lrc', 'dico', 'E')