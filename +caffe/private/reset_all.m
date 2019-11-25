function reset_all()
% reset_all()
%   clear all solvers and stand-alone nets and reset Caffe to initial status
% addpath('private/caffe_.mexw64')
caffe_('reset');
is_valid_handle('get_new_init_key');

end
