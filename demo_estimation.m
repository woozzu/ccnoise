function demo_estimation
img = im2double(imread('data/test.jpg'));

disp('Calculating noise level...');
np = estimate_ccnoise('train/', 'deploy.prototxt', 'demo_iter_100000.caffemodel', img);

disp('The covariance matrix of (1, 1)')
disp([np(1, 1, 1), np(1, 1, 4), np(1, 1, 5);
    np(1, 1, 4), np(1, 1, 2), np(1, 1, 6);
    np(1, 1, 5), np(1, 1, 6), np(1, 1, 3)]);

% The estimated covariance matrices must be positive definite
%
% icov = zeros(size(np));
% for i = 1:size(np, 1)
%     for j = 1:size(np, 1)
%          c = [np(i, j, 1), np(i, j, 4), np(i, j, 5);
%              np(i, j, 4), np(i, j, 2), np(i, j, 6);
%              np(i, j, 5), np(i, j, 6), np(i, j, 3)];
%          % Sigma^(-1)
%          c = inv(c);
%          % Repair covariance matrix to positive definite
%          [V, D] = eig(c);
%          D = diag(max(diag(D), 10e-4));
%          c = V * D * V';
%          icov(i, j, :) = [c(1), c(5), c(9), c(2), c(3), c(6)];
%     end
% end