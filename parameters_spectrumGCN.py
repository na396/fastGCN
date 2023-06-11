#
drp_first = True
#
trainVal_percent_perClass = 0.01
train_percent_perClass = 0.80
train_num_perClass = None
val_num_perClass = None

#
train_percent_allClasses = 0.65
val_percent_allClasses = 0.15
train_num_allClasses = None
val_num_allClasses = None
#

mask_type = "original" # either "perClass" or "allClasses" or "original"

# spec_in=2*graph.num_classes
# spec_out= graph.num_classes
num_linear = 2
add_relu = True
conv1_out = 128
conv_bias = True
#
data_verobse = True
#
train_verbose = True
train_keep = True

test_keep = True
test_verbose = True

plt_sh = False
plt_keep = True
maskInd = 0
imagetype="svg"

use_gpu = True