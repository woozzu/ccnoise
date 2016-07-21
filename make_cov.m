function covar = make_cov(prefix, start_idx, end_idx, postfix, img_mean)
img_mean = img_mean * 255;
covar = zeros(size(img_mean, 1), size(img_mean, 2), 6);
for i = start_idx:end_idx
    img = double(imread(strcat(prefix, sprintf('%06d', i), postfix)));
    covar(:, :, 1) = covar(:, :, 1) + (img(:, :, 1) - img_mean(:, :, 1)).^2;
    covar(:, :, 2) = covar(:, :, 2) + (img(:, :, 2) - img_mean(:, :, 2)).^2;
    covar(:, :, 3) = covar(:, :, 3) + (img(:, :, 3) - img_mean(:, :, 3)).^2;
    covar(:, :, 4) = covar(:, :, 4) + (img(:, :, 1) - img_mean(:, :, 1)) .* (img(:, :, 2) - img_mean(:, :, 2));
    covar(:, :, 5) = covar(:, :, 5) + (img(:, :, 1) - img_mean(:, :, 1)) .* (img(:, :, 3) - img_mean(:, :, 3));
    covar(:, :, 6) = covar(:, :, 6) + (img(:, :, 2) - img_mean(:, :, 2)) .* (img(:, :, 3) - img_mean(:, :, 3));
end
covar = covar ./ (end_idx - start_idx + 1);