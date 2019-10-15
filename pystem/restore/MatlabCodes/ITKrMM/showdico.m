function dico2d = showdico(dico, s1, space, nrow);

% syntax: dico2d = showdico(dico, s1, space, nrow);
%
% converts vectorised 2d signals  
% back to 2d and puts them into a big matrix
% usage: imagesc(showdico(dico))
%
% input:
% dico... s_1*s_2 x K matrix of K 2d signals of size s_1xs_2
%                       (e.g. image patches) 
% s1... 2d width ...default sqrt(d)
% space... space between images ... default 2 pixel
% nrow...  number of images per column ... default min(s_1,ceil(sqrt(K)))
%                                                  
%
% last modified 12.01.17
% Karin Schnass 

[d,K]=size(dico);

if nargin < 2
    s1=ceil(sqrt(d));
end

s2 = d/s1;

if s2*s1 ~= d
    disp('wrong size s1');
    return;
end

if nargin < 3
    space = 2;
end

if nargin < 4
    nrow = min(s1,ceil(sqrt(K)));
end

[dummy,I]=max(abs(dico));
for k=1:K
    dummy(k)=sign(dico(I(k),k));
end
dico=dico*diag(dummy);
dico2d=[];
fac=-max(max(abs(dico)));
%nrow=ceil(sqrt(K));
ncol=ceil(K/nrow);
for row = 1:nrow
    newrow=[];
    for col =1:ncol
        k=(row-1)*ncol + col;
        if k <=K
            newrow=[newrow, reshape(dico(:,k),s1,s2),fac*ones(s1,space)];
        else
            newrow=[newrow, fac*ones(s1,s2),fac*ones(s1,space)];
        end
    end
    dico2d=[dico2d; newrow; fac*ones(space,ncol*(s2+space))];
end; 

dico2d=[fac*ones(space,ncol*(s2+space));dico2d];
dico2d=[fac*ones(nrow*(s1+space)+space,space) dico2d];

