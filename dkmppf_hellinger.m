%dkmppf.m
%
%DESCRIPTION:
%    implements destructive version of kernel matching pursuit with pre-fitting (see [1])
%
%INPUTS:
%    *D: columns are dictionary points over which function(s) to approximate are defined
%
%    *W: each rows contains weights that define a particular function
%
%    *kernel: kernel struct
%
%    *eps2: squared-norm distance for stopping criterion
%
%    *opt: options struct with the following fields
%        *KDD: kernel matrix for D
%
%OUTPUTS:
%    *idxdkmppf: indices of dictionary points used (refer to columns of D)
%
%    *Wdkmppf: approximation dictionary weights for D(:,idxkomp)
%
%REFERENCES:
%    [1] Vincent and Bengio. "Kernel matching pursuit." Machine Learning. 2002.
%    [2] https://en.wikipedia.org/wiki/Woodbury_matrix_identity#Derivation_from_LDU_decomposition. accessed 19 May 2016.

function [mu_removal_return,Sigma_removal_return,idxdkmppf] = dkmppf_hellinger(D,y,kernel,x,eps2,noise_prior)
%function [idxdkmppf,Wdkmppf] = dkmppf_hellinger(D,y,kernel,x,eps2,noise_prior)

% the set of indices of D that we are keeping
Y = 1:size(D,2);
% the set of indices of D that we are removing
Z = [];
%max_iter=1000;
%step_size=0.01;
KDD=kernel.f(D,D);
k_XX=kernel.f(x,x);
k_DX=kernel.f(D,x);

 %compute current posterior distribution parameters
    mu_full=k_DX'/(KDD + noise_prior^2*eye(size(KDD)))*y';
   
    Sigma_full=diag(k_XX-k_DX'/(KDD + noise_prior^2*eye(size(KDD)))*k_DX + noise_prior^2);
      
% continue removing points as long as we have them to remove
continue_pruning = 1; 
while continue_pruning
  % find the least-important element to delete
  gmin = Inf;
  for i = 1:length(Y)
      % removal set to consider
      Zi = [Z Y(i)];
     remains=setdiff( 1:length(Y),Zi);

%compute mean and covariance with index Zi removed
   mu_removal=k_DX(remains,:)'/(KDD(remains,remains) ...
            + noise_prior^2*eye(size(KDD(remains,remains))))*y(remains)';
   Sigma_removal=diag(k_XX-k_DX(remains,:)'/(KDD(remains,remains) ...
               + noise_prior^2*eye(size(KDD(remains,remains))))*k_DX(remains,:)+ noise_prior^2);
      
   

      % compute error for this removal set, see if smallest
      gi =abs(hellinger_distance(mu_removal,mu_full,diag(Sigma_removal),diag(Sigma_full)));
      if gi<gmin
          gmin = gi;
          imin = i;
      end
  end
%   pause
%   gmin
%   pause(5)
%   eps2
  % if best error is still okay, delete the corresponding element
  if gmin <= eps2
    Z = [Z Y(imin)];
    Y(imin) = [];
  % otherwise, we are done
  else
    continue_pruning = 0;
    if i>1
    %%%% calculate the new mu and covariance
       Zi = [Z Y(i-1)];
     remains=setdiff( 1:length(Y),Zi);

%compute mean and covariance with index Zi removed
   mu_removal_return=k_DX(remains,:)'/(KDD(remains,remains) ...
            + noise_prior^2*eye(size(KDD(remains,remains))))*y(remains)'; 
   Sigma_removal_return=diag(k_XX-k_DX(remains,:)'/(KDD(remains,remains) ...
               + noise_prior^2*eye(size(KDD(remains,remains))))*k_DX(remains,:)+ noise_prior^2); 
      
    else
        mu_removal_return=mu_removal; 
   Sigma_removal_return=Sigma_removal;
    end
  end
end

% return the indices that we kept
idxdkmppf = Y;
% project weights onto remaining indices
 %full weights
%  %W=(KDD + noise_prior^2*eye(size(KDD)))*y';
%  W_restricted=zeros(length(Y),max_iter);
%  target=zeros(1,max_iter);
% 
%  W_restricted(:,1)= inv(KDD(Y,Y) ...
%                     + noise_prior^2*eye(size(KDD(Y,Y))))*y(Y)'; 
%  target(1)=mu_removal;
%  for k=2:max_iter
%   mu_removal_estimate(k)=k_DX(Y,:)'*W_restricted(:,k); 
% 
%  %want to choose restricted indices weight s.t. to minimizer Hellinger distance
%  target(k)=hellinger_distance(mu_removal_estimate(k),mu_full,Sigma_removal,Sigma_full);
%  %do this with Euler backward differentiation
%  W_restricted(k+1)= W_restricted(k) - step_size* (target(k) - target(k-1))/(W_restricted(k) - W_restricted(k-1));
%  end
 %
%Wdkmppf =  W_restricted(:,end); 
