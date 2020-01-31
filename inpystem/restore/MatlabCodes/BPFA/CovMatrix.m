function [R Ron Rn]=CovMatrix(WaveLength,Omega,P,MissIndex,P)
% caculated the covariance matrix for Gaussian process
nBand=length(WaveLength);
for i=1:nBand
    for j=1:nBand
        Dis(i,j)=abs(WaveLength(i)-WaveLength(j));
    end
end
% Dis = eye(nBand)/P;
Dis=double(Dis);
tempR=exp(-Dis/Omega);
RR=zeros(nBand*P^2,nBand*P^2);
for i=1:P.^2
    RR((i-1)*nBand+1:(i)*nBand,(i-1)*nBand+1:(i)*nBand)=tempR;
    
end
Rn=RR(MissIndex,MissIndex);
Ron=RR(MissIndex,~MissIndex);
R=RR(~MissIndex,~MissIndex);


