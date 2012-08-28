#pragma once
#include "opencv2\opencv.hpp"
#include <vector>
using namespace std;
using namespace cv;
#define GABOR_DIRECTION		8
#define GABOR_SHAPE_SIZE	5
#define GABOR_FEATURE_NUM	(GABOR_DIRECTION*GABOR_SHAPE_SIZE)
class GaborFR
{
public:
	GaborFR();
	//静态方法！
	static Mat	getImagGaborKernel(Size ksize, double sigma, double theta, 
									double nu,double gamma=1, int ktype= CV_32F);
	//获取gabor卷积核的虚数部分
	static Mat	getRealGaborKernel( Size ksize, double sigma, double theta, 
									double nu,double gamma=1, int ktype= CV_32F);
	static Mat	getPhase(Mat &real,Mat &imag);//计算出相位
	static Mat	getMagnitude(Mat &real,Mat &imag);//计算出角度
	static void getFilterRealImagPart(Mat& src,Mat& real,Mat& imag,Mat &outReal,Mat &outImag);
	static Mat	getFilterRealPart(Mat& src,Mat& real);//计算src和实数部分的卷积结果
	static Mat	getFilterImagPart(Mat& src,Mat& imag);//计算src和复数部分的卷积结果

	//内成员函数
	void		Init(Size ksize=Size(19,19), double sigma=2*CV_PI,
					double gamma=1, int ktype=CV_32FC1);//计算40个卷积核

	vector<Mat>	calculateOneExampleReturnMatVecort(Mat& src,int method=GABOR_MANITUDE);
	Mat			calculateOneExampleReturnOneRow(Mat& src,int method=GABOR_MANITUDE);
	enum{
		GABOR_MANITUDE,
		GABOR_PHASE,
		GABOR_BOTH
	};
	//一些酱油函数
	bool		setGaborSize(Size size);
	Size		getGaborSize();
private:
	void		iCalculateGaborFilterMat(Mat &src,int method=GABOR_MANITUDE);
//	void		iCalculateGaborFilterCol(Mat &src,int method=GABOR_MANITUDE);
private:
	vector<Mat> gaborRealKernels;
	vector<Mat> gaborImagKernels;
	vector<Mat> gaborResultPhase;
	Mat			gaborResultPhaseRow;
	vector<Mat> gaborResultMagnitude;
	Mat			gaborResultMagnitudeRow;
	bool		isInited;
	Size		gaborSize;
};