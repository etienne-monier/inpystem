function newatom=rec_lratomc(data,masks,dicoL,maxit,inatom)

% syntax: atom=rec_lratomc(data,masks,dicoL,maxit,inatom)
%
% recover new low rank atom 
%      equivalent to itkrmm with K=S=1;
% (described in the new paper)
%
% input:
% data... d x N matrix containing the (corrupted)
%                                     training signals as its columns
% masks ... d x N matrix containing the masks as its columns
%                    - masks(.,.) in {0,1} - default masks = 1                              
% dicoL...orthobasis for already recovered low rank component - default []
% maxit... number of iterations - default 10
% inatom... initialisation, unit vector in R^d - default random 
%
% output:
% atom....d x 1 atom of low rank component
%        
%
% last modified 09.08.16
% Karin Schnass 

%%%%%% preparations
if(nargin < 1)
    disp('syntax: dico=itkrmm(data,masks,K,S,dicoL,maxit,inatom)');
    dico=[];
    return;
end

[d,N]=size(data);

if (nargin < 2)
    masks=ones(d,N);
end

data=data.*masks; % safeguard against the massimo effect

if(nargin < 3)
    dicoL=[];
end

[~,L]=size(dicoL);

if(nargin < 4)
    maxit = 10;
end

if (nargin < 5) 
    inatom = randn(d,1); 
    inatom = inatom/norm(inatom);
end

if size(inatom)~=[d,1]
    disp('initialisation atom bugged - random initialisation used');	
    inatom = randn(d,1); 
    inatom = inatom/norm(inatom);
end

if L > 0
    inatom=inatom-dicoL*dicoL'*inatom;
    inatom = inatom/norm(inatom);   
end

%%% subtract lr comp from data
if L > 0
    for n=1:N
        dicoLMn = dicoL.*(masks(:,n)*ones(1,L));
        data(:,n) = data(:,n) - dicoLMn*(pinv(dicoLMn)*data(:,n));
    end
end

%%% learn new atom %%%%%
oldatom=inatom;

for it=1:maxit
    
    ip=oldatom'*data;
    maskweight = sum(masks,2);

    if L == 0 
        newatom = data*(sign(ip)');
    else
        newatom = zeros(d,1);
        for n=1:N
            oldatom_mn=oldatom.*masks(:,n);
            dicoLplus = [dicoL.*(masks(:,n)*ones(1,L)),oldatom_mn];
            resn=data(:,n)-dicoLplus*(pinv(dicoLplus)*data(:,n));
            newatom = newatom + sign(ip(n))*resn; 
            newatom = newatom + abs(ip(n))*oldatom_mn/(sum(oldatom_mn.*oldatom_mn));
        end
        
    end
        
    if min(maskweight) > 0
        newatom=newatom./maskweight;
    else
        newatom=newatom./(maskweight + 0.01);
    end
    
    if L>0 
        newatom=newatom-dicoL*(dicoL'*newatom);
    end
    
    newatom=newatom./norm(newatom);
    oldatom = newatom;  
end


