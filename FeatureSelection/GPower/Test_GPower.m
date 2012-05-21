% Script to test the GPower algorithms proposed in
% M. Journée, Y. Nesterov, P. Richtárik, R. Sepulchre, Generalized power 
% Method for sparse principal component analysis, arXiv:0811.4724v1, 2008
%

clear all

p=20; n=50;                         % p, number of samples; n, number of variables
A=randn(p,n);                       % data matrix
A=A-repmat((mean(A,1)),p,1);        % Centering of the data
m=5;                                % Number of components:

% ***** Single-unit algorithms *****
gamma=0.1*ones(1,m);                % sparsity weight factors -one for each component - 
                                    % in relative value with respect to the theoretical upper bound
Z1=GPower(A,gamma,m,'l1',0);        % Sparse PCA by deflation, l1 penalty
Z2=GPower(A,gamma.^2,m,'l0',0);     % Sparse PCA by deflation, l0 penalty

% ***** Block algorithms *****
mu=[1:m].^(-1);                     % distinct mu_i
%mu=ones(m,1);                      % identical mu_i
gamma=0.1*ones(1,m);
Z3=GPower(A,gamma,m,'l1',1,mu);     % Block sparse PCA, l1 penalty
Z4=GPower(A,gamma.^2,m,'l0',1,mu);  % Block sparse PCA, l0 penalty