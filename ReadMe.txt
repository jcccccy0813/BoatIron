GetPicture.cpp采图用，程序开始时读入一个数字（1或2），1代表单目采图，2代表双目采图。如果输入1则再读入一个数（0或者1），0代表左相机，1代表右相机。
SingleCalibration.cpp单目标定用
DoubleCalibration.cpp双目标定用
DoubleMatch.cpp双目极限矫正、视差计算、生成点云用

2.0更新了getpicture.cpp 单目采集后放入相应single文件夹 双目采集后放入stereo文件夹
3.0更新了双目标定程序，在原有opencv示例代码基础上从计算内参改为读取通过单目标定完成的左右摄像头内参，同时生成intrinsics.yml。
