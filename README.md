# iSmartDNN
This is a repository for FPGA-based neural network inference of iSmart2 team of Design Automation Conference 2018 (https://www.dac.com/content/2018-system-design-contest). iSmart2 teammembers are from UIUC, the Boeing company, IBM and Inspirit Iot.


# Repo organization
DNN_train: The DNN model definition and training scripts

DNN_HLS: The DNN model implemented using Vivado High Level Synthesis (HLS), written in C++.

Host: Host code run on CPU for FPGA control

Overlay: The bitstream and tcl file for FPGA configuration


