function [A]=OMPm(D, X, Mask,L)
%=============================================
% This is a modified version of OMP to account for corruptions in the signal.
% A slightly different modification of Masked OMP is available in
% "Sparse and Redundant Representations: From Theory to
% Applications in Signal and Image Processing," the book written by M. Elad in 2010.
% Sparse coding of a group of signals based on a given 
% dictionary and specified number of atoms to use. 
% input arguments: 
%       D - the dictionary (its columns MUST be normalized).
%       X - the masked signals to represent (MX=MDA)
%       Mask - the masks on the signal
%       L - the max. number of coefficients for each signal.
% output arguments: 
%       A - sparse coefficient matrix.
%=============================================


[n,P]=size(X);
[n,K]=size(D);

for k=1:1:P,
    a=[];
    x = X(:,k);
    m = Mask(:,k);
    xm=x.*m;
    residual=xm;
    Dm=D.*(m*ones(1,K));
    
    scale = sqrt(sum(Dm.*Dm));
    nonzero = find(scale > 0.001/sqrt(n));
    scale(nonzero) = 1./scale(nonzero);

    indx=zeros(L,1);
    for j=1:1:L,
        proj=Dm'*residual;
        proj = scale'.*proj;
        [maxVal,pos]=max(abs(proj));
        pos=pos(1);
        indx(j)=pos;
        a=pinv(Dm(:,indx(1:j)))*xm;
        residual=xm-Dm(:,indx(1:j))*a;
        if sum(residual.^2) < 1e-6
            break;
        end
    end;
    temp=zeros(K,1);
    temp(indx(1:j))=a;
    A(:,k)=sparse(temp);
end;



