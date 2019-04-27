The source code implements the tracking method in 
"Multiple Cues-Aware Visual Tracking for UAV with Online Two-Stage Evaluation".

The code is implemented in Matlab 2017a on Windows 10.

How to run the code:

1. Install Windows Caffe at https://github.com/happynear/caffe-windows.

2. Install the siamese networks for verifier. The deploy document and caffe model are located in the serial_ptav_v1\siamese_networks\. 

3. Run the MCVT_Demo_all_seq.m can run the MCVT tracker on the 100 image sequences from UAV123. The precision plots of all sequences will be saved in the MCVT\Test_MCVT\MCVT_results\res_picture\. All the .mat files will be saved in the MCVT\Test_MCVT\MCVT_results\.

4.The Siamese networks caffe model and prototxt can be downloaded at:
https://drive.google.com/open?id=1YEgRj_KhI1U7ELogl98ZS0ufCYaByBRj

5.Any question, please feel free to contact us.