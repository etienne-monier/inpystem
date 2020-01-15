%%%% inpainting script %%%%

rng(0);                  % reset random generator for reproducible results
savefile = 'mysave.mat'; % save to savefile if defined
display = 1;             % display progress and learned dictionaries
cp2wksvd = 0;            % compare to wksvd dictionary !!! Attention slow !!!

%%%%%%%%% parameter settings %%%%%%%%%%
picname = 'mandrill.png';  %%% picture to destroy and reconstruct

corrtype = 'r';     %%% 'r'... random erasures or 'c' cracks 
corrlevel = 0.8;    %%% erasure probability per pixel for 'r'
runs = 1;           %%% for 'c', runs = 1 automatically

s1 = 8;             %%% patchheight
s2 = s1;            %%% patchwidth

L = 1;              %%% number of low rank atoms (for wKSVD L=1)
maxitLR = 10;       %%% number of iterations for learning lr atoms
K = 2*s1*s2 - L;    %%% dictionary size
S = s1-L;           %%% sparsity level in the learning step
maxit = 40;         %%% number of iterations for dictionary learning

Sinp = 20;          %%% sparsity level for inpainting;

%%%% avoid wksvd for L>3 or more than 40 iterations to save time
if L>1 || maxit>40
    cp2wksvd = 0;
end

%%%% load picture 
pic = imread(strcat('images/',picname));
pic = im2double(pic);
[d1,~,d3]=size(pic);
if d3 > 1
    pic = 0.2989*pic(:,:,1)+0.5870*pic(:,:,2)+0.1140*pic(:,:,3);
end
pic = imresize(pic,256/d1);   %%% get 256x256 or so 
[d1,d2]=size(pic);

%%%% only one run for uploaded mask
if corrtype == 'c'
    runs = 1;
    if d2~=256
        disp('need square picture for crack mask, 50% erasure mask used');
        corrtype == 'r';
        corrlevel = 0.5;
    end        
end

d = s1*s2;

psnr_noisy = zeros(runs,1);

psnr_itkrmm = zeros(runs,1);
runtime_itkrmm = zeros(runs,1);

if cp2wksvd == 1;
    psnr_wksvd = zeros(runs,1);
    runtime_wksvd = zeros(runs,1);
end

%%%% create masks and initial dicos (dl algos may use random generator)
if corrtype == 'c'
    load('images/cracks.mat')
    mask = cracks;
else 
    masks = ceil(rand(d1,d2,runs)-corrlevel);
end

inits = randn(d,K+L,runs);

for run = 1 : runs
    if corrtype == 'r'
        mask = masks(:,:,run);
    end
        
    corrpic = mask.*pic;   %%% corrupted picture
    
    psnr_noisy(run)=psnr(corrpic,pic);
        
    %%% get patches of picture and mask with locations
    [corrpatches, ploc] = pic2patches(corrpic,s1,s2);
    [maskpatches, mloc] = pic2patches(mask,s1,s2);
    
    %%% learning with mask info
    %%% learn low rank atoms 
    tic
    lrc = [];
    if display == 1 && L>0;
        disp('learning low rank component using mask info');
    end
  
    for ll = 1 : L
        % initialise low rank atom
        inatoml = inits(:,ll,run);
        if ll > 1
            inatoml = inatoml - lrc*lrc'*inatoml;
        end
        inatoml = inatoml/norm(inatoml);
        atoml = rec_lratom(corrpatches,maskpatches,lrc,maxitLR,inatoml);
        lrc = [lrc, atoml];     
    end
    
    %%% learn dictionary with itkrmm
    if display == 1;
        disp('learning dictionary using mask info (itkrmm)');
    end
    %%% initialise
    dico = inits(:,L+1:K,run);
    if L>0
        dico = dico - lrc*lrc'*dico;
    end
    dico = dico*diag(1./sqrt(sum(dico.*dico)));
    dico = itkrmm(corrpatches, maskpatches, K, S, lrc, maxit, dico);
    runtime_itkrmm(run) = toc;
    if display == 1;
        imagesc(showdico([lrc,dico]));
        title('itkrmm dictionary (and low rank component)');
        drawnow;
    end
    
    %%% learn dictionary with wksvd
    if cp2wksvd == 1;
        if display == 1;
            disp('learning wksvd dictionary');
            param.displayProgress = 1;
        else
            param.displayProgress = 0;
        end
        param.K = K+L;              % dictionary size 
        param.dSparsity = S+L; 
        param.numIteration = maxit;    
        param.InitializationMethod = 'GivenMatrix';
        tic
        indico = inits(:,L+1:K+L,run);
        if L == 1;
            param.preserveDCAtom = 1; %%% algo ensures orthogonality of initialisation to lrc         
        elseif L > 1;
            param.preserveDCAtom = 1; 
            indico = indico - lrc*lrc'*indico; 
        end
        indico = indico*diag(1./sqrt(sum(indico.*indico)));
        param.initialDictionary = [lrc,indico]; 
        [dico_wksvd, output] = wKSVD(corrpatches, maskpatches, param); 
        runtime_wksvd(run) = toc;
        if display == 1;
            imagesc(showdico(dico_wksvd));
            title('wksvd dictionary (and low rank component)');
            drawnow;
        end
    end
    
    %%%% inpainting
    if display == 1;
        disp('inpainting with itkrmm dictionary');
    end
    lrcdico = [lrc,dico];    
    
    coeff = OMPm(lrcdico, corrpatches, maskpatches, Sinp);
    inppatches = lrcdico*coeff;
    inppic = patches2pic(inppatches, ploc, s1);
    psnr_itkrmm(run) = psnr(inppic,pic);
    if display == 1;
        imagesc(inppic)
        title(strcat('psnr using itkrmm: ',num2str(psnr_itkrmm(run))));
        drawnow;
    end
    
    if cp2wksvd == 1
        if display == 1;
            disp('inpainting with wksvd dictionary');
        end
        coeff_wksvd = OMPm(dico_wksvd, corrpatches, maskpatches, Sinp);
        inppatches_wksvd = dico_wksvd*coeff_wksvd;
        inppic_wksvd = patches2pic(inppatches_wksvd, ploc, s1);
        psnr_wksvd(run) = psnr(inppic_wksvd,pic);
        if display == 1;
            imagesc(inppic_wksvd)
            title(strcat('psnr using wksvd: ',num2str(psnr_wksvd(run))));
            drawnow;
        end
    end
    
    if exist('savefile','var')
        if corrtype == 'r'
            save(savefile,'lrc*','dico*','psnr*','inppic*','masks','runtime*');
        else
            save(savefile,'lrc*','dico*','psnr*','inppic*','runtime*');
        end
    end         
end
