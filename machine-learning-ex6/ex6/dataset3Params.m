function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.


x1 = [1 2 1]; x2 = [0 4 -1]; 
trials =[0.01 0.03 0.1 0.3 1 3 10 30];
J=zeros(length(trials), length(trials));

for i=1:length(trials)
	for j=1:length(trials)
		model= svmTrain(X, y, trials(i), @(x1, x2) gaussianKernel(x1, x2, trials(j)));
		predictions = svmPredict(model, Xval);
		J(i,j)=mean(double(predictions(:,1) ~= yval(:,1)));
	end
end

[a,C]=min(min(J,[],2));
[a,sigma]=min(min(J,[],1));
C=trials(C)
sigma=trials(sigma)


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
