function [pic,seen] = patches2pic(patches,locations,s1)

% syntax: patches = pic2patches(pic,s1,s2)
%
% selects N random patches of size s1xs2 out of an pic 
% and stores them as (s1.s2) x N matrix
%
% input
% patches... (s1.s2) x N matrix, each patch stored as s1.s2 column vector
%              to get to 2d shape use pn2d=reshape(patches(:,n),[s1,s2])
% s1.... width of 2d signal
% locations...2 times N matrix of locations of each patch in original image
%
% output:
% pic... estimated picture
%                                                  
%
% last modified 29.11.16
% Karin Schnass 


%%%% preparations

if(nargin < 2)
    disp('synthax: pic = patches2pic(patches,locations,s1)');
    return;
end

[d,N]=size(patches);

if nargin < 3
    s1=sqrt(d);
end
    
s2=d/s1;

if s2*s1 ~= d
    disp('wrong or no size s1');
    return;
end

d1=max(locations(1,:)) + s1-1;
d2=max(locations(2,:)) + s2-1;

pic= zeros(d1,d2);
seen = zeros(d1,d2);

for n=1:N
    n1=locations(1,n);
    n2=locations(2,n);
    pic(n1:n1+s1-1, n2:n2+s2-1)=pic(n1:n1+s1-1, n2:n2+s2-1)+reshape(patches(:,n),s1,s2);
    seen(n1:n1+s1-1, n2:n2+s2-1)=seen(n1:n1+s1-1, n2:n2+s2-1)+ones(s1,s2);
end

seen=max(seen, ones(d1,d2));

pic=pic./seen;

