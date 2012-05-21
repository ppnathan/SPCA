function [dis] = emdL1( h1, h2 )
%EMDL1 Compute the Earth Mover's Distance with L1 ground distance.
%   emdL1(h1,h2) returns the earth mover's distance between histgrams h1 and h2
% 	using L1 ground distance. The algorithm is implemented in C++, this
% 	function provides a matlab interface for it.
%   
%Note:
%   1. h1 and h2 must have same dimensions and normalized.
%   2. Now the algorithm support distances upto 3 dimension.
% 
%Details of the algorithm is referred to 
% 	H. Ling and K. Okada, 
% 	An Efficient Earth Mover's Distance Algorithm for Robust Histogram Comparison, 
% 	IEEE Transaction on Pattern Analysis and Machine Intelligence (PAMI), 
% 	29(5):840-853, 2007.
% 
%   Haibin Ling (hbling AT temple.edu)
%   Date: 2008/09/18

s1	= size(h1);
s2	= size(h2);
dis	= -1;
if length(s1)~=length(s2) | sum(s1-s2)~=0
	disp('Histgram dimensions not match!');
elseif size(s1) > 3
	disp('Histgram dimension > 3 not supported!');
else
	n1	= s1(1);
	if(length(s1)>1)	n2	= s1(2);		
	else				n2	= 1;
	end
	if(length(s1)>2)	n3	= s1(3);
	else				n3	= 1;
	end
	dis	= emdL1_m(h1, h2, n1, n2, n3);
end
	
	