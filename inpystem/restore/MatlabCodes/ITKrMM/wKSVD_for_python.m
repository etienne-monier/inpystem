% ITKrMM Code for Python inpystem library.
%
% Arguments:
%
%     with P the number of pixels per patch (incl. bands)
%
%     data: (P, N) array
%         The Y data in patch format. N is the number of patches.
%     mdata: (P, N) array
%         The mask in patch format. N is the number of patches.
%     K: int
%         Dictionary size.
%     L: int
%         Number of low-rank components to estimate.
%     S: int
%         Code sparsity. Should be lower than K.
%     save_it: int
%         If 1, the rec. data is estimated for each iter. and saved (that's
%         computationally heavy !).
%     CLS_init: int
%         1 if lr component should not be computed, else 0.
%     init: (P, K-L)
%         Dictionary initialization
%     init_lr: (P, L) array
%         Low-rank component.
%     Nit: int
%         Number of ITKrMM iterations.
%     Nit_lr: int
%         Number of low rank components iterations.
%     verbose: int
%         1 to be verbose else 0.
%
% Output:
%
%     outdata: (P, N) array
%         Reconstructed data.
%     time: float
%         Execution time
%     lrc: (P, L) array
%         Estimated low-rank component
%     dico: (P, K)
%         Estimated dictionary.
%

%% Load data

data = double(data);
mdata = double(mdata);

K = double(K);
L = double(L);
S = double(S);

init = double(init);
init_lr = double(init_lr);

Nit = double(Nit);
Nit_lr = double(Nit_lr);


%% learn low rank atoms 

if CLS_init

    lrc = init_lr;

else

    lrc = [];
    if verbose == 1 && L>0;
        disp('Learning low rank component using mask info ...');
    end

    for ll = 1 : L
        % initialise low rank atom
        inatoml = init_lr;
        if ll > 1
            inatoml = inatoml - lrc*lrc'*inatoml;
        end
        inatoml = inatoml/norm(inatoml);
        atoml = rec_lratom(data,mdata,lrc,Nit_lr,inatoml);
        lrc = [lrc, atoml];     
    end

end

%% learn dictionary with wKSVD

if verbose == 1;
    disp('Learning wksvd dictionary ...');
    param.displayProgress = 1;
else
    param.displayProgress = 1;
end

param.K = K;
param.dSparsity = S; 
param.numIteration = Nit;    
param.InitializationMethod = 'GivenMatrix';

tic

% If the dc component is preserved, I should remove one element in the init.
param.preserveDCAtom = 1;

indico = init;
if param.preserveDCAtom
    indico = init(:, 1:end-1);  % (L+1):(L+K-1));
end

% Preserves orthogonality wrt lrc
if L > 1;
    param.preserveDCAtom = 1; 
    indico = indico - lrc*lrc'*indico; 
end

% Dico atoms normalization
indico = indico*diag(1./sqrt(sum(indico.*indico)));

% Initialization.
param.initialDictionary = [lrc,indico]; 
param.displayProgress = 1;
param.save_it = save_it;

% Launching reconstruction.
[dico, output] = wKSVD(data, mdata, param); 

time = toc;

%% inpainting

if verbose == 1;
    disp('Inpainting with wksvd dictionary ...');
end

coeff_wksvd = OMPm(dico, data, mdata, S);
outdata = dico*coeff_wksvd;


%% Save results

if verbose == 1;
    disp('Saving data ...')
end

save(outName,'outdata','time','lrc', 'dico')