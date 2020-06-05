% Code for paper: Consistent Online Gaussian Process Regression
%                 Without the Sample Complexity Bottleneck 

% Link of the paper: https://arxiv.org/pdf/2004.11094.pdf
% Authors: Alec Koppel, Hrusikesha Pradhan, Ketan Rajawat 


% clearing the workspace
clc;
close all;
clear all;
warning off;

% Number of epochs the training data is passed through
epoch = 1;     
% Number of samples considered for running average (just to average out the fluctuations)
running_sample_size=1;

% Compression Budget: In paper it is \epsilon_t 
 Keps=0.01*10^(-3); 
 
%%
% Various data set (uncomment to use them)
% load data_sinc.mat
%  load data_sinc3d.mat
load data_kin40.mat
% load data_lidar.mat
%  load data_boston_housing.mat
% load data_abalone.mat
% load data_pendulum.mat
% load data_pumadyn.mat

% Size of the subset of data from the training data set for hyperparameter
% optimization
%100 for sinc and lidar, 300 for kin40 and 133 for abalone data, 50 for
%boston data, 50 for pendulum data
hyp_trainsize=300; 

xTrain=data.xtrain;
yTrain=data.ytrain;
xTest=data.xtest;
yTest=data.ytest;

% Dimension of the feature vector (input data)
dim = size(xTrain, 2);

global hyp_xtrain;
global y0_hyp_ytrain_pog;

%Select a subset of data from training data for Hyperparameter Optimization
hyp_xtrain=xTrain(1:hyp_trainsize, :);
hyp_ytrain=yTrain(1:hyp_trainsize, :);

%center/standardize the data
y0_hyp_ytrain_pog=hyp_ytrain-mean(hyp_ytrain);
% y0_hyp_ytrain_pog=(hyp_ytrain-mean(hyp_ytrain,1))./std(hyp_ytrain,0,1);



% Initializing Gaussian kernel bandwidth parameter. Here we use ARD
% (Automatic Relevance Determination), i.e., we have separate theta for
% every dimension of the input vector
% We have dim+2 bcoz we also consider the amplitude and variance of noise.
% Thus causing two extra parameters.

% theta=zeros(dim+2,1);
 theta= rand(dim+2,1);
%  theta(1:dim,1) = -2*log((max(hyp_xtrain)-min(hyp_xtrain))'/2); % log 1/(lengthscales)^2
% theta(dim+1,1) = log(var(y0_hyp_ytrain_pog,1)); % log size 
% theta(dim+2,1) = log(var(y0_hyp_ytrain_pog,1)/4); % log noise
% theta=exp(theta);
 
% Initialization of various variables required for hyperparameter
% optimisation
A = [];
b = [];
Aeq = [];
beq = [];
lb= 1e-8*ones(dim+2,1);%for kin40
ub=[];
nonlcon = [];

% Hyperparameter Optimisation using inbuilt matlab function "fmincon"
 options = optimset('PlotFcns',@optimplotfval);
%options = optimoptions(@fminunc,'Display','iter','Algorithm','quasi-newton');
% theta=fminsearch(@neglog_marginallikelihood,theta,options);
  theta=fmincon(@neglog_marginallikelihood,theta,A,b,Aeq,beq,lb,ub,nonlcon,options);
 
 noise_prior=sqrt(theta(end));
 theta=theta(1:end-1);
 

%Training and Test data
xTrain = xTrain(hyp_trainsize+1:end, :);
yTrain = yTrain(hyp_trainsize+1:end, :);
 nTrain = size(xTrain, 1);
 nTest = size(xTest, 1);
data.Xtrain=xTrain';
data.Xtest=xTest';
data.ytrain=yTrain';
data.ytest=yTest';
data.ycv=yTest';
data.Xcv=xTest';

%center/standardize the data
data.ytrain_pog=data.ytrain-mean(data.ytrain);
var_ytrain=var(data.ytrain_pog);
data.ytest_pog=data.ytest-mean(data.ytest);
% data.ytrain_pog=(data.ytrain-mean(data.ytrain,2))./std(data.ytrain,0,2);
% data.ytest_pog=(data.ytest-mean(data.ytest,2))./std(data.ytest,0,2);

% Initializing the kernel and various functions used for the kernel(look kRBF function) 
kernel = kRBF(theta);
T = nTrain;
M = zeros(1,nTrain*epoch);% Keeping track of the model order in each iteration of epoch
M(1) = 1;                           % Will be one intially for the first iteration
D = data.Xtrain(:,1);               % First point included in dictionary, D is dictionary

torigin = 1;
fprintf('The value of y is\n\n');
y=data.ytrain_pog(:,1);

%index updated from T to nTrain*epoch
mu_posterior_temp = zeros(1,nTrain*epoch);       % temporary mu posterior
Sigma_posterior_temp = zeros(1,nTrain*epoch);    % temporary sigma posterior
mu_posterior = zeros(1,T);                       % for training points
Sigma_posterior = zeros(1,T);
eval_estimate = zeros(1,nTrain);
%index updated from T to nTrain*epoch
lcv = zeros(1,nTrain*epoch);
%index updated from T to nTrain*epoch
hellinger = zeros(1,nTrain*epoch);               % Hellinger and bhattacharya distance for each iteration
bhattacharyya = zeros(1,nTrain*epoch);
mu_test_posterior = zeros(1,nTest);   % for test points
sigma_test_posterior = zeros(1,nTest);
sample_test_error_pog = zeros(1, nTrain*epoch);
smse=zeros(1, nTrain*epoch); % Standardized mean square error
msll=zeros(1, nTrain*epoch); % Mean standardized log loss

for epo = 1:epoch
    
    if (epo==1)
        init=M(1) + 1;
    else
        init=1;
    end
    
    count=0;
    
    for t=init:T
        
        % handle indexing
        tidx = t:min(T,t);
        
        % handle possible wraparound
        tidxtrain = mod(tidx-1,nTrain)+1;
        
        % %%%     % [debug: grabber for if training point is already in the dictionary]
        % %%%     if any(ismember(tidxtrain,torigin))
        % %%%         fprintf('\tREPEAT!\n');
        % %%%     end
        
        
        KDD = kernel.f(D,D);
        %% Calculation of various metric on test data
        for i=1:nTest
            xtest_pt = data.Xtest(:,i);
            KxtestD = kernel.f(D, xtest_pt);
            mu_test_posterior(1,i)= KxtestD'/(KDD + noise_prior^2*eye(size(KDD)))*y';
            sigma_test_posterior(1,i) = kernel.f(xtest_pt, xtest_pt)  - KxtestD'/(KDD + noise_prior^2*eye(size(KDD)))*KxtestD + noise_prior^2;
            stan_sq_test_error(1,i)=(norm(mu_test_posterior(1,i)-data.ytest_pog(1,i),2)^2)/var_ytrain;
            stan_log_loss(1,i)=(norm(mu_test_posterior(1,i)-data.ytest_pog(1,i),2)^2)/sigma_test_posterior(1,i) + log(2*pi*sigma_test_posterior(1,i));
        end
       
        testerror_variance(t,:)=sigma_test_posterior;
        sample_test_error_pog(1,(epo-1)*T + tidx -1) = mean(abs(mu_test_posterior - data.ytest_pog));
        %storing test mean and variance for all training index
        mean_of_test_samples(t,:)=mu_test_posterior;
        var_of_test_samples(t,:)=sigma_test_posterior;
        actual_y_test_samples(t,:)=data.ytest_pog;
        % standardized mean squared error
        smse(1,(epo-1)*T + tidx -1)=mean(stan_sq_test_error);
        % mean standardized log loss (MSLL)
        msll(1,(epo-1)*T + tidx -1)=0.5*mean(stan_log_loss);
        
        %% training process
        toriginold = torigin;
        Dold=D;
        yold=y;
        
        k_DX=kernel.f(D,data.Xtrain(:,tidxtrain));
        k_XX=kernel.f(data.Xtrain(:,tidxtrain),data.Xtrain(:,tidxtrain));
        KDaugDaug=kernel.f(D,D);
        
        Maug = size(D,2);
        
        %%% Updates before the projection
        mu_posterior_temp((epo-1)*T+t)=k_DX'/(KDaugDaug + noise_prior^2*eye(size(KDaugDaug)))*y';
        Sigma_posterior_temp((epo-1)*T+t)=k_XX-k_DX'/(KDaugDaug + noise_prior^2*eye(size(KDaugDaug)))*k_DX + noise_prior^2;
        toriginaug = [toriginold tidx];
        %Calculation of Hellinger distance
        [hellinger((epo-1)*T +tidx),bhattacharyya((epo-1)*T +tidx)]=hellinger_distance(mu_posterior_temp((epo-1)*T + t-1),mu_posterior_temp((epo-1)*T +t),Sigma_posterior_temp((epo-1)*T+t-1),Sigma_posterior_temp((epo-1)*T+t));
        
        
        % set approximation budget
        %eps_h = Keps*hellinger(tidx); % experimental rule
        eps_h = Keps; % experimental rule
       
        
        if eps_h==0
            
            Daug=[D data.Xtrain(:,tidxtrain)];
            yaug=[y data.ytrain_pog(tidxtrain)];
            D = Daug;
            y = yaug;
            KDD = KDaugDaug;
            torigin = toriginaug;
            M(tidx) = Maug;
            
        else
            
            % Addition of point before the calculation of pseudo
            % dictionary
            D=[D data.Xtrain(:,tidxtrain)];
            y=[y data.ytrain_pog(tidxtrain)];
            
            %% dkmppf compression and pseudoinput search
            [~,~,idxdkmppf] = dkmppf_hellinger(D,y,kernel,data.Xtrain(:,tidxtrain),eps_h,noise_prior);
            D = D(:,idxdkmppf);
            y = y(idxdkmppf);
                      
            k_DX = kernel.f(D,data.Xtrain(:,tidxtrain));
            k_XX = kernel.f(data.Xtrain(:,tidxtrain),data.Xtrain(:,tidxtrain));
            KDaugDaug=kernel.f(D,D);
            mu_posterior(t)=k_DX'/(KDaugDaug + noise_prior^2*eye(size(KDaugDaug)))*y';
            Sigma_posterior(t)=k_XX-k_DX'/(KDaugDaug + noise_prior^2*eye(size(KDaugDaug)))*k_DX + noise_prior^2;
            M((epo-1)*T +tidx) = size(D,2);
            %uncomment if dictionary search is not done on full model
            %order
            %   dict_indices=mod(idxdkmppf,nTrain);
                       
        end
        
        % [debug] compute cross-validation loss
        
        eval_estimate(tidxtrain)=mu_posterior(tidx);
        %  lcv(tidx) = norm(eval_estimate - data.ytrain)^2;
        %(epo-1)*T + added to the index tidx of lcv
        lcv((epo-1)*T +tidx)=mean(abs(eval_estimate - data.ytrain_pog));
        
        % [debug] report
        fprintf('\titerate %d: M=%d, theta2=%.3e, rcond(K)=%.3e, Test error=%.3e, SMSE=%.3e\n',  ...
            (epo-1)*T +t,M((epo-1)*T +t),theta(end),rcond(KDaugDaug), sample_test_error_pog(1,(epo-1)*T + tidx -1), smse(1,(epo-1)*T + tidx -1));
        
        % Test error calculation for last sample of training data
        if mod(t, nTrain)== 0
            
            KDD = kernel.f(D,D);
            
            for i = 1:nTest
                xtest_pt = data.Xtest(:,i);
                KxtestD = kernel.f(D, xtest_pt);
                mu_test_posterior(1,i)= KxtestD'/(KDD + noise_prior^2*eye(size(KDD)))*y';
                sigma_test_posterior(1,i) = kernel.f(xtest_pt, xtest_pt)  - KxtestD'/(KDD + noise_prior^2*eye(size(KDD)))*KxtestD + noise_prior^2;
                stan_sq_test_error(1,i)=(norm(mu_test_posterior(1,i)-data.ytest_pog(1,i),2)^2)/var_ytrain;
                stan_log_loss(1,i)=(norm(mu_test_posterior(1,i)-data.ytest_pog(1,i),2)^2)/sigma_test_posterior(1,i) + log(2*pi*sigma_test_posterior(1,i));
            end
            testerror_variance(t,:)=sigma_test_posterior;
            
            mean_of_test_samples(t,:)=mu_test_posterior;
            var_of_test_samples(t,:)=sigma_test_posterior;
            actual_y_test_samples(t,:)=data.ytest_pog;
            
            sample_test_error_pog(1,T*epo) = mean(abs(mu_test_posterior - data.ytest_pog));
            % standardized mean squared error
            smse(1,T*epo)=mean(stan_sq_test_error);
            msll(1,T*epo)=0.5*mean(stan_log_loss);
            
        end
    end
end
        
running_sample_test_error_pog=runningaverage(sample_test_error_pog,running_sample_size);
figure;
p1 = plot(running_sample_test_error_pog,'b','LineWidth',1);
legend('POG')
xlabel('t, index of training samples','interpreter','Latex','FontSize',12);
ylabel('Absolute Difference Test Error','interpreter','Latex','FontSize',12); set(gca,'FontSize',12);

figure;
p1 = plot(smse,'r','LineWidth',1);
legend('POG')
xlabel('t, index of training samples','interpreter','Latex','FontSize',12);
ylabel('Standardized mean squared error','interpreter','Latex','FontSize',12); set(gca,'FontSize',12);

figure;
p1 = plot(msll,'r','LineWidth',1);
legend('POG')
xlabel('t, index of training samples','interpreter','Latex','FontSize',12);
ylabel('Mean standardized log loss','interpreter','Latex','FontSize',12); set(gca,'FontSize',12);

figure;
semilogy(lcv(1:(epo-1)*T +t),'LineWidth',2);%title('Training Loss');
xlabel('t, index of training samples','interpreter','Latex','FontSize',12);
ylabel('Posterior Mean Square Error','interpreter','Latex','FontSize',12);
set(gca,'FontSize',12);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
semilogy(hellinger(1:(epo-1)*T+ t),'LineWidth',2);%title('Training Loss');
%plot(bhattacharyya(1:t),'LineWidth',2);
%str{1}='Hellinger'; str{2}='Bhattacharyya';
%legend(str,'Location','Best','interpreter','Latex','FontSize',30);
xlabel('t, index of training samples','interpreter','Latex','FontSize',12);
ylabel('Hellinger Distance','interpreter','Latex','FontSize',12); set(gca,'FontSize',12);

figure;
plot(M(1:(epo-1)*T+t),'b','LineWidth',2); hold on; plot(1:(epo-1)*T+t,'r','LineWidth',2)
xlabel('t, index of training samples','interpreter','Latex','FontSize',12)
ylabel('Model Order','interpreter','Latex','FontSize',12)
set(gca,'FontSize',12);
str{1}='POG'; str{2}='Dense GP';
legend(str,'Location','Best','interpreter','Latex','FontSize',12);

figure;
hold on; x =1:nTrain;
plot(x,data.ytrain,'g','LineWidth',1.5);
plot(x,mu_posterior(T-nTrain+1:end),'LineWidth',1.5,'LineStyle','--'); 
xlabel('$t$, number of samples','interpreter','Latex','FontSize',12)
ylabel('Target Value','interpreter','Latex','FontSize',12); 
curve1 = mu_posterior(T-nTrain+1:end)+2*Sigma_posterior(T-nTrain+1:end);
curve2 = mu_posterior(T-nTrain+1:end)-2*Sigma_posterior(T-nTrain+1:end);
plot(x, curve1, 'r', 'LineWidth', 1.5);
plot(x, curve2, 'c', 'LineWidth', 1.5);
x2 = [x, fliplr(x)];
inBetween = [curve1, fliplr(curve2)];
h=fill(x2, inBetween, 'r'); set(h, 'FaceAlpha', 0.08)
str{1}='Training Data'; str{2}='Posterior Estimate'; 
legend(str,'Location','Best','interpreter','Latex','FontSize',12);set(gca,'FontSize',12);
hold off;

save('pog_data_kin.mat')