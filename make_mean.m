function m = make_mean(rows, cols, prefix, start_idx, end_idx, postfix)
m = zeros(rows, cols, 3);
for i = start_idx:end_idx
    m = m + im2double(imread(strcat(prefix, sprintf('%06d', i), postfix)));
end
m = m ./ (end_idx-start_idx+1);