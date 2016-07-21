function nl = estimate_ccnoise(model_dir, net_model, net_weights, img)
net_model = [model_dir net_model];
net_weights = [model_dir net_weights];
phase = 'test';

caffe.set_mode_cpu();
net = caffe.Net(net_model, net_weights, phase);
nl = zeros(size(img, 1), size(img, 2), 6);

for i = 1:8:size(img, 1)
    for j = 1:8:size(img, 2)
        patch = permute(img(i:i+8-1, j:j+8-1, :), [3, 2, 1]);
        data = [reshape(patch, [3, 64]); repmat(reshape(patch, [192, 1]), 1, 64)];
        out = net.forward({reshape(data, [1, 1, 195, 64])});
        out = out{1};
        nl(i:i+8-1, j:j+8-1, :) = permute(reshape(out, [6, 8, 8]), [3, 2, 1]);
    end
end

caffe.reset_all();