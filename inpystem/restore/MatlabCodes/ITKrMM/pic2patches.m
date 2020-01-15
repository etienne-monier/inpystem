function [patches,locations] = pic2patches(pic,s1,s2,N)

% syntax: [patches,locations] = pic2patches(pic,s1,s2,N)
%
% selects N random patches of size s1xs2 out of a pic 
% and stores them as (s1.s2) x N matrix
%
% input:
% pic... d_1 x d_2 matrix 
% s1...patch width - default s1 = 8 
% s2...patch height - default s2 = s1
% N...number of desired patches
%                                                  
%
% output:
% patches... (s1.s2) x N matrix, each patch stored as s1.s2 column vector
%              to get to 2d shape use pn2d=reshape(patches(:,n),[s1,s2])
% locations...location of each patch in original image
%
% last modified 29.11.16
% Karin Schnass 


%%%% preparations
if(nargin < 1)
    disp('syntax: [patches, masks]=getpatches(pic,s1,s2,margin,N,mask)');
    patches=[];
    return;
end

[d1,d2]=size(pic); 

if(nargin < 2)
    s1=8;
end

if (nargin < 3)
    s2=s1;
end

Nmax = (d1-s1+1) * (d2-s2+1);

if nargin < 4
    N=Nmax;
end

if d1 < s1 || d2 < s2
    disp('patches larger than pic');
    patches = [];
    return;
end

if N > Nmax
    disp('N larger than maximal number of patches, maximal number used')
    N = Nmax;
end


patches=[];
locations=[];

if N>=Nmax
    p=1:Nmax;
else
    p=randperm(Nmax);
    p=p(1:N);
end

for n=1:N
    m1=mod(p(n)-1,(d1-s1+1))+1;
    m2=(p(n)-m1)/(d1-s1+1)+1;
    npatch=pic(m1:m1+s1-1,m2:m2+s2-1);
    patches=[patches, npatch(:)];
    locations=[locations [m1,m2]'];   
end