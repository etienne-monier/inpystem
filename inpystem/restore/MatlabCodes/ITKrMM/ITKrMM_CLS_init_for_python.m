%% Load data

addpath('..')

corrpatches = double(corrpatches);
maskpatches = double(maskpatches);

d = double(d);
K = double(K);
L = double(L);
S = double(S);
X = double(X);

lrc_init = double(lrcomp);
dico = double(dicoinit);

maxit = double(maxit);
maxitLR = double(maxitLR);


%% learn low rank atoms 

tic

lrc = lrc_init;

% lrc = [];
% if verbose == 1 && L>0;
%     disp('Learning low rank component using mask info');
% end

% for ll = 1 : L
%     % initialise low rank atom
%     inatoml = lrc_init(:,ll);
%     if ll > 1
%         inatoml = inatoml - lrc*lrc'*inatoml;
%     end
%     inatoml = inatoml/norm(inatoml);
%     atoml = rec_lratom(corrpatches,maskpatches,lrc,maxitLR,inatoml);
%     lrc = [lrc, atoml];     
% end

%% learn dictionary with itkrmm

if verbose == 1;
    disp('learning dictionary using mask info (itkrmm)');
end

if L>0
    dico = dico - lrc*lrc'*dico;
end

dico = dico*diag(1./sqrt(sum(dico.*dico)));
[dico, E] = itkrmm(corrpatches, maskpatches, K, S, lrc, maxit, dico, verbose, X);

%% inpainting

if verbose == 1;
    disp('inpainting with itkrmm dictionary');
end

lrcdico = [lrc,dico];

coeff = OMPm(lrcdico, corrpatches, maskpatches, S);

outdata = lrcdico*coeff;

time = toc;

%% Save results

if verbose == 1;
    disp('Saving data ...')
end

save(outName,'outdata','time','lrc', 'dico', 'E')