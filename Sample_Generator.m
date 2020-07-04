% Matthew Bailey Project 3 -> Statistical classifiers using Bayesians
C = 2;
% Covariance matrices

S(:,:,1)=[0.8 0.2 0.1 0.05 0.01;
0.2 0.7 0.1 0.03 0.02;
0.1 0.1 0.8 0.02 0.01;
0.05 0.03 0.02 0.9 0.01;
0.01 0.02 0.01 0.01 0.8];

S(:,:,2)=[0.9 0.1 0.05 0.02 0.01;
0.1 0.8 0.1 0.02 0.02;
0.05 0.1 0.7 0.02 0.01;
0.02 0.02 0.02 0.6 0.02;
0.01 0.02 0.01 0.02 0.7];

Mu = [zeros(5,1), ones(5,1)]; % two 5x1 columns

P = [1/2 1/2]'; % apriori probability

training_samples = [100 1000]; % both training sample sizes
test_samples = 10000;

train_set_100 = [];
train_set_1k = [];

rng(0);
for j = 1:2 % both training sets
    rng(0);
    for i = 1:C % both classes
        if(j == 1)
            t100 = mvnrnd(Mu(:,i),S(:,:,i),fix(P(i)*training_samples(1)));
            train_set_100 = [train_set_100; t100];
        else
            t1k = mvnrnd(Mu(:,i),S(:,:,i),fix(P(i)*training_samples(2)));
            train_set_1k = [train_set_1k;t1k];
        end           
    end
end

test_set = [];
rng(100)
for i = 1:C % both classes
    test_temp = mvnrnd(Mu(:,i),S(:,:,i),fix(P(i)*test_samples));
    
    test_set = [test_set; test_temp];
end

save('Samples.mat','train_set_100','train_set_1k','test_set','S','Mu')