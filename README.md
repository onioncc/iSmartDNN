# iSmartDNN
This is a repository for FPGA-based neural network inference of iSmart2 team of Design Automation Conference 2018 (https://www.dac.com/content/2018-system-design-contest). iSmart2 teammembers are from UIUC, the Boeing company, IBM and Inspirit Iot.


# Repo organization
DNN_train: The DNN model definition and training scripts

DNN_HLS: The DNN model implemented using Vivado High Level Synthesis (HLS), written in C++.

Host: Host code run on CPU for FPGA control

Overlay: The bitstream and tcl file for FPGA configuration

# model
onioncc upload the code in different version
DNN_train: 12layer 256channel
DNN_HLS:   12layer 384channel which may be the model used in the 2018-system-design-contest
Host:maybe 14layer 512channel

you can change the code in the first cell of `ismart2.ipynb` in Host like this:

```py
conv_weight_1x1_all = xlnk.cma_array(shape=(405, 16, 16), dtype=np.uint16)
conv_weight_3x3_all = xlnk.cma_array(shape=(22, 16, 3, 3), dtype=np.uint16)
bias_all = xlnk.cma_array(shape=(67, 16), dtype=np.uint16)
DDR_pool_3_out = xlnk.cma_array(shape=(48, 82, 162), dtype=np.uint16)
DDR_pool_6_out = xlnk.cma_array(shape=(96, 42, 82), dtype=np.uint16)
DDR_buf = xlnk.cma_array(shape=(36, 16, 22, 42), dtype=np.uint16)
predict_box = xlnk.cma_array(shape=(5,), dtype=np.float32)
```

so that it may match the code in `DNN_HLS`.

# note
In `DNN_HLS`,there is `conv_1x1_fl.cc` and `conv_1x1_fl_fix.cc`.I found that `conv_1x1_fl.cc` goes wrong in vivado 2018.3 so I write `conv_1x1_fl_fix.cc` to fix the bug.You can delete `conv_1x1_fl_fix.cc` or delete `conv_1x1_fl.cc` to test which can goes right on your vivado.

You can download the dataset from https://github.com/xyzxinyizhang/2018-DAC-System-Design-Contest
