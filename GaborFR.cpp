#include <stdafx.h>
#include "GaborFR.h"
GaborFR::GaborFR()
{
	isInited = false;
	gaborSize = Size(19,19);
}
void GaborFR::Init(Size ksize, double sigma,double gamma, int ktype)
{
	gaborRealKernels.clear();
	gaborImagKernels.clear();
	double mu[8]={0,1,2,3,4,5,6,7};
	double nu[5]={0,1,2,3,4};
	int i,j;
	for(i=0;i<GABOR_SHAPE_SIZE;i++)
	{
		for(j=0;j<GABOR_DIRECTION;j++)
		{
			gaborRealKernels.push_back(getRealGaborKernel(ksize,sigma,mu[j]*CV_PI/8+CV_PI/2,nu[i],gamma,ktype));
			gaborImagKernels.push_back(getImagGaborKernel(ksize,sigma,mu[j]*CV_PI/8+CV_PI/2,nu[i],gamma,ktype));
		}
	}
	isInited = true;

}

Mat GaborFR::getImagGaborKernel(Size ksize, double sigma, double theta, double nu,double gamma, int ktype)
{
	double	sigma_x		= sigma;
	double	sigma_y		= sigma/gamma;
	int		nstds		= 9;
	double	kmax		= CV_PI/2;
	double	f			= cv::sqrt(2.0);
	int xmin, xmax, ymin, ymax;
	double c = cos(theta), s = sin(theta);
	if( ksize.width > 0 )
	{
		xmax = ksize.width/2;
	}
	else//这个和matlab中的结果一样，默认都是19 !
	{
		xmax = cvRound(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));
	}
	if( ksize.height > 0 )
	{
		ymax = ksize.height/2;
	}
	else
	{
		ymax = cvRound(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));
	}
	xmin = -xmax;
	ymin = -ymax;
	CV_Assert( ktype == CV_32F || ktype == CV_64F );
	float*	pFloat;
	double*	pDouble;
	Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
	double k		=	kmax/pow(f,nu);
	double scaleReal=	k*k/sigma_x/sigma_y;
	for( int y = ymin; y <= ymax; y++ )
	{
		if( ktype == CV_32F )
		{
			pFloat = kernel.ptr<float>(ymax-y);
		}
		else
		{
			pDouble = kernel.ptr<double>(ymax-y);
		}
		for( int x = xmin; x <= xmax; x++ )
		{
			double xr = x*c + y*s;
			double v = scaleReal*exp(-(x*x+y*y)*scaleReal/2);
			double temp=sin(k*xr);
			v	=  temp*v;
			if( ktype == CV_32F )
			{
				pFloat[xmax - x]= (float)v;
			}
			else
			{
				pDouble[xmax - x] = v;
			}
		}
	}
	return kernel;
}
//sigma一般为2*pi
Mat GaborFR::getRealGaborKernel( Size ksize, double sigma, double theta, 
	double nu,double gamma, int ktype)
{
	double	sigma_x		= sigma;
	double	sigma_y		= sigma/gamma;
	int		nstds		= 9;
	double	kmax		= CV_PI/2;
	double	f			= cv::sqrt(2.0);
	int xmin, xmax, ymin, ymax;
	double c = cos(theta), s = sin(theta);
	if( ksize.width > 0 )
	{
		xmax = ksize.width/2;
	}
	else//这个和matlab中的结果一样，默认都是19 !
	{
		xmax = cvRound(std::max(fabs(nstds*sigma_x*c), fabs(nstds*sigma_y*s)));
	}

	if( ksize.height > 0 )
		ymax = ksize.height/2;
	else
		ymax = cvRound(std::max(fabs(nstds*sigma_x*s), fabs(nstds*sigma_y*c)));
	xmin = -xmax;
	ymin = -ymax;
	CV_Assert( ktype == CV_32F || ktype == CV_64F );
	float*	pFloat;
	double*	pDouble;
	Mat kernel(ymax - ymin + 1, xmax - xmin + 1, ktype);
	double k		=	kmax/pow(f,nu);
	double exy		=	sigma_x*sigma_y/2;
	double scaleReal=	k*k/sigma_x/sigma_y;
	int	   x,y;
	for( y = ymin; y <= ymax; y++ )
	{
		if( ktype == CV_32F )
		{
			pFloat = kernel.ptr<float>(ymax-y);
		}
		else
		{
			pDouble = kernel.ptr<double>(ymax-y);
		}
		for( x = xmin; x <= xmax; x++ )
		{
			double xr = x*c + y*s;
			double v = scaleReal*exp(-(x*x+y*y)*scaleReal/2);
			double temp=cos(k*xr) - exp(-exy);
			v	=	temp*v;
			if( ktype == CV_32F )
			{
				pFloat[xmax - x]= (float)v;
			}
			else
			{
				pDouble[xmax - x] = v;
			}
		}
	}
	return kernel;
}
Mat GaborFR::getMagnitude(Mat &real,Mat &imag)
{
	CV_Assert(real.type()==imag.type());
	CV_Assert(real.size()==imag.size());
	int ktype=real.type();
	int row = real.rows,col = real.cols;
	int i,j;
	float*	pFloat,*pFloatR,*pFloatI;
	double*	pDouble,*pDoubleR,*pDoubleI;
	Mat		kernel(row, col, real.type());
	for(i=0;i<row;i++)
	{
		if( ktype == CV_32FC1 )
		{
			pFloat = kernel.ptr<float>(i);
			pFloatR= real.ptr<float>(i);
			pFloatI= imag.ptr<float>(i);
		}
		else
		{
			pDouble = kernel.ptr<double>(i);
			pDoubleR= real.ptr<double>(i);
			pDoubleI= imag.ptr<double>(i);
		}
		for(j=0;j<col;j++)
		{
			if( ktype == CV_32FC1 )
			{
				pFloat[j]= sqrt(pFloatI[j]*pFloatI[j]+pFloatR[j]*pFloatR[j]);
			}
			else
			{
				pDouble[j] = sqrt(pDoubleI[j]*pDoubleI[j]+pDoubleR[j]*pDoubleR[j]);
			}
		}
	}
	return kernel;
}
Mat GaborFR::getPhase(Mat &real,Mat &imag)
{
	CV_Assert(real.type()==imag.type());
	CV_Assert(real.size()==imag.size());
	int ktype=real.type();
	int row = real.rows,col = real.cols;
	int i,j;
	float*	pFloat,*pFloatR,*pFloatI;
	double*	pDouble,*pDoubleR,*pDoubleI;
	Mat		kernel(row, col, real.type());
	for(i=0;i<row;i++)
	{
		if( ktype == CV_32FC1 )
		{
			pFloat = kernel.ptr<float>(i);
			pFloatR= real.ptr<float>(i);
			pFloatI= imag.ptr<float>(i);
		}
		else
		{
			pDouble = kernel.ptr<double>(i);
			pDoubleR= real.ptr<double>(i);
			pDoubleI= imag.ptr<double>(i);
		}
		for(j=0;j<col;j++)
		{
			if( ktype == CV_32FC1 )
			{
				pFloat[j] = asin(pFloatI[j]/sqrt(pFloatR[j]*pFloatR[j]+pFloatI[j]*pFloatI[j]));
			}//CV_32F
			else
			{
				pDouble[j] = asin(pDoubleI[j]/sqrt(pDoubleR[j]*pDoubleR[j]+pDoubleI[j]*pDoubleI[j]));
			}//CV_64F
		}
	}
	return kernel;
}
Mat GaborFR::getFilterRealPart(Mat& src,Mat& real)
{
	CV_Assert(real.type()==src.type());
	Mat dst;
	Mat kernel;
	flip(real,kernel,-1);//中心镜面
	filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_REPLICATE);
	return dst;
}
Mat GaborFR::getFilterImagPart(Mat& src,Mat& imag)
{
	CV_Assert(imag.type()==src.type());
	Mat dst;
	Mat kernel;
	flip(imag,kernel,-1);//中心镜面
	filter2D(src,dst,CV_32F,kernel,Point(-1,-1),0,BORDER_REPLICATE);
	return dst;
}
void GaborFR::getFilterRealImagPart(Mat& src,Mat& real,Mat& imag,Mat &outReal,Mat &outImag)
{
	outReal = getFilterRealPart(src,real);
	outImag = getFilterImagPart(src,imag);
}
void GaborFR::iCalculateGaborFilterMat(Mat &src,int method)
{
	if(src.type()!=CV_32FC1)
		src.convertTo(src,CV_32F);
	CV_Assert(gaborRealKernels.size()==gaborImagKernels.size());
	int num = gaborRealKernels.size();
	if(method == GABOR_BOTH||method==GABOR_MANITUDE)
	{
		gaborResultMagnitude.clear();
	}
	if(method == GABOR_BOTH||method==GABOR_PHASE)
	{
		gaborResultPhase.clear();
	}
	Mat outReal,outImag;
	for(int k=0;k<num;k++)
	{
		outReal = getFilterRealPart(src,gaborRealKernels[k]);
		outImag = getFilterImagPart(src,gaborImagKernels[k]);
		if(method==GABOR_BOTH||method==GABOR_MANITUDE)
		{
			gaborResultMagnitude.push_back(getMagnitude(outReal,outImag));
		}
		if(method==GABOR_BOTH||method==GABOR_PHASE)
		{
			gaborResultPhase.push_back(getPhase(outReal,outImag));
		}
	}
}
Mat GaborFR::calculateOneExampleReturnOneRow(Mat &src,int method)
{
	if(src.empty())
		return Mat();
	iCalculateGaborFilterMat(src,method);
	if(method==GABOR_BOTH||method==GABOR_MANITUDE)
	{
		gaborResultMagnitudeRow.release();
	}
	if(method==GABOR_BOTH||method==GABOR_PHASE)
	{
		gaborResultPhaseRow.release();
	}
	if(method==GABOR_MANITUDE||method==GABOR_BOTH)
	{
		int num1 = gaborResultMagnitude.size();
		if(num1 <= 0)
			return Mat();
		for(int i=0;i<num1;i++)
		{
			gaborResultMagnitudeRow.push_back(Mat(gaborResultMagnitude[i].reshape(1,1).t()));
		}
	}
	if(method==GABOR_PHASE||method==GABOR_BOTH)
	{
		int num2=gaborResultPhase.size();
		if(num2<=0)
			return Mat();
		for(int i=0;i<num2;i++)
		{
			gaborResultPhaseRow.push_back(Mat(gaborResultPhase[i].reshape(1,1).t()));
		}
	}
	if(method==GABOR_MANITUDE)
	{
		return gaborResultMagnitudeRow.t();
	}
	else if(method==GABOR_PHASE)
	{
		return gaborResultPhaseRow.t();
	}
	else
	{
		Mat M;
		M.push_back(gaborResultMagnitudeRow);
		M.push_back(gaborResultPhaseRow);
		return M.t();
	}

}
vector<Mat>	GaborFR::calculateOneExampleReturnMatVecort(Mat& src,int method)
{
	iCalculateGaborFilterMat(src,method);
	//怎么返回数据
	if(method==GABOR_MANITUDE)
	{
		return gaborResultMagnitude;
	}
	else if(method==GABOR_PHASE)
	{
		return gaborResultPhase;;
	}
	else 
	{
		vector<Mat> M;
		M=gaborResultMagnitude;
		for(int i=0;i<(int)gaborResultPhase.size();i++)
		{
			M.push_back(gaborResultPhase[i]);
		}
		return M;
	}
}
//酱油函数
bool	GaborFR::setGaborSize(Size size)
{
	if(size.height<=0||size.width<=0)
		return false;
	gaborSize = size;
	return true;
}
Size	GaborFR::getGaborSize()
{
	return gaborSize;
}