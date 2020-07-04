import scipy.io 
import numpy as np
import math

# DEVELOPER: Matthew Bailey
# Imported the training and testing sets using my matlab code which will be included with this in a zip folder



TRAINING_SET_DICT = {0:'100 samples',1:'1000 samples'}

# Going to try and import the training and testing sets

M = scipy.io.loadmat('Samples.mat')
# print(M['test_set'])
# print(M['train_set_100'])
# print(M['train_set_1k'])

# First 50:  C1
# Second 50: C2
train_set_100 = np.array(M['train_set_100'])

# First 500: C1
# Second 500:C2
train_set_1k = np.array(M['train_set_1k'])

# First 5k:  C1
# Second 5k: C2
test_set = np.array(M['test_set'])
S_mat = np.array(M['S'])
# print(train_set_100)

# Now have all sets of data, create the Mean vectors and Covariance matrices of distribution generated

Mu = np.array(M['Mu'])
Mu = Mu.T
# print(Mu)


P = np.array([[1/2],
			  [1/2]])

# print(P)

# Covariance matrices

				# CLASS 1 DISTRIBUTION
S = np.array([[[0.8, 0.2, 0.1, 0.05, 0.01],
			   [0.2, 0.7, 0.1, 0.03, 0.02],
			   [0.1, 0.1, 0.8, 0.02, 0.01],
			   [0.05,0.03,0.02, 0.9, 0.01],
			   [0.01,0.02,0.01,0.01, 0.8]],
			   	# CLASS 2 DISTRIBUTION
			  [[0.9, 0.1, 0.05, 0.02,0.01],
			   [0.1, 0.8, 0.1, 0.02, 0.02],
			   [0.05,0.1, 0.7, 0.02, 0.01],
			   [0.02,0.02,0.02, 0.6, 0.02],
			   [0.01,0.02,0.01,0.02, 0.7]]])

# print(S.shape) 		# (2,5,5) --> Looks like ^ Being printed
# print(S_mat.shape) 	# (5,5,2) --> Looks really weird being printed

def Univariate_PDF(x,mu,var):
    ret_val = (1/math.sqrt(2*(math.pi)*var)) * math.exp((-(x-mu)**2)/(2*var)) # DOUBLE CHECK IF THIS USES VARIANCE OR STD DEV
    return ret_val
# This PDF represents a normal univariate gaussian distribution

def Multivariate_PDF(x,mu,cov,dims): 
	ret_val = (1/( ((2*(math.pi))**(dims/2)) * (math.sqrt(np.linalg.det(cov))) )) * math.exp((-0.5)*(x-mu).T @ np.linalg.inv(cov) @ (x - mu)) 
	return ret_val

L = 5; C = 2; dims = 5

N = np.array([100, 1000])

test_labels = np.array([np.ones(int(5000)), np.ones(5000)*(-1)])
test_labels = test_labels.flatten()
# print(test_labels)
# print(len(N))

for i in range(len(N)): # goes thru training set twice
	# 3) Use the training samples to learn the following classifiers:
			# i]	Naive Bayes
			# ii]	Bayes classifier using MLE
			# iii]	Bayes classifier using true parameters

	# NAIVE BAYES CLASSIFIER
	# remember: assume no covariance therefore do the multiplication of each univariate probability
	if(i == 0):
		samples = train_set_100
	else:
		samples = train_set_1k

	train_labels = np.array([np.ones(int(N[i]/2)), np.ones(int(N[i]/2))*(-1)])
	train_labels = train_labels.flatten()

	C1_Samples = samples[train_labels==1]
	C2_Samples = samples[train_labels==-1]

	mean_C1 = np.mean(C1_Samples,axis = 0) # do the mean of all columns (features)
	mean_C2 = np.mean(C2_Samples,axis = 0)
	
	var_C1 = np.var(C1_Samples,axis = 0,ddof = 0) # 1/N
	var_C2 = np.var(C2_Samples,axis = 0,ddof = 0)

	# Now we have trained our classifier, must use the testing sample

	Naive_Bayes_Classifications = []

	for j in range(10000): # Go through each testing sample
		Prob_C1 = 1
		Prob_C2 = 1
		for k in range(dims): # must multiply each probability per feature
			Prob_C1 = Prob_C1*(Univariate_PDF(test_set[j,k],mean_C1[k],var_C1[k]))
			Prob_C2 = Prob_C2*(Univariate_PDF(test_set[j,k],mean_C2[k],var_C2[k]))			
		if(Prob_C1 > Prob_C2):
			Naive_Bayes_Classifications.append(1)
		else:
			Naive_Bayes_Classifications.append(-1)
	Naive_Bayes_Classifications = np.array(Naive_Bayes_Classifications)
	equal_cases = np.equal(test_labels,Naive_Bayes_Classifications)
	equal_cases = np.where(equal_cases == True)
	equal_cases = len(np.array(equal_cases).flatten())
	# print(len(equal_cases))
	Naive_Bayes_ER = 10000 - equal_cases
	# print(equal_cases)
	print("Naive Bayes Errors from ",TRAINING_SET_DICT[i]," training set: ",Naive_Bayes_ER)

	# BAYES CLASIFIER USING MLE
		# we know the form of the distribution, parametric
		# Most Likely Estimate
		# Need to calculate: Mean vectors, Covariance matrix

	# We already have the means (mean_C1, mean_C2) so just need covariance matrices

	cov_C1 = np.cov(C1_Samples.T,ddof = 0)
	cov_C1 = cov_C1.T
	cov_C2 = np.cov(C2_Samples.T,ddof = 0)
	cov_C2 = cov_C2.T

	# print("Covariance for class 1")
	# print(cov_C1)
	# print("Covariance for class 2")
	# print(cov_C2)

	Bayes_MLE_Classifications = []

	for j in range(10000):
		Prob_C1 = Multivariate_PDF(test_set[j],mean_C1,cov_C1,dims)*P[0]
		Prob_C2 = Multivariate_PDF(test_set[j],mean_C2,cov_C2,dims)*P[1]
		if(Prob_C1 > Prob_C2):
			Bayes_MLE_Classifications.append(1)
		else:
			Bayes_MLE_Classifications.append(-1)
	Bayes_MLE_Classifications = np.array(Bayes_MLE_Classifications)
	equal_cases = np.equal(test_labels,Bayes_MLE_Classifications)
	equal_cases = np.where(equal_cases == True)
	equal_cases = len(np.array(equal_cases).flatten())

	Bayes_MLE_ER = 10000 - equal_cases
	print("Bayes MLE Errors from ",TRAINING_SET_DICT[i]," training set: ", Bayes_MLE_ER)

	# BAYES CLASSIFIER USING TRAINING SET GENERATION PARAMETERS

	Bayes_GEN_Classifications = []

	for j in range(10000):
		pass
		Prob_C1 = Multivariate_PDF(test_set[j],Mu[0,:],S[0,:,:],dims)*P[0]
		Prob_C2 = Multivariate_PDF(test_set[j],Mu[1,:],S[1,:,:],dims)*P[1]
		if(Prob_C1 > Prob_C2):
			Bayes_GEN_Classifications.append(1)
		else:
			Bayes_GEN_Classifications.append(-1)
	Bayes_GEN_Classifications = np.array(Bayes_GEN_Classifications)
	equal_cases = np.equal(test_labels,Bayes_GEN_Classifications)
	equal_cases = np.where(equal_cases == True)
	equal_cases = len(np.array(equal_cases).flatten())

	Bayes_GEN_ER = 10000 - equal_cases
	print("Bayes GEN Errors from ",TRAINING_SET_DICT[i]," training set: ", Bayes_GEN_ER)
