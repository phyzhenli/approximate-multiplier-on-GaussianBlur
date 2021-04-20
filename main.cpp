#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <opencv2/opencv.hpp>
#include "split.h"

using namespace std;
using namespace cv;

double getPSNR(const Mat& I1, const Mat& I2)
{
    Mat s1;
    absdiff(I1, I2, s1);       // |I1 - I2|
    s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
    s1 = s1.mul(s1);           // |I1 - I2|^2

    Scalar s = sum(s1);         // sum elements per channel

    double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

    if( sse <= 1e-10) // for small values return zero
        return 0;
    else
    {
        double  mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        return psnr;
    }
}

Scalar getMSSIM( const Mat& i1, const Mat& i2)
{
    const double C1 = 6.5025, C2 = 58.5225;
    /***************************** INITS **********************************/
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);            // cannot calculate on one byte large values
    i2.convertTo(I2, d);
    Mat I2_2   = I2.mul(I2);        // I2^2
    Mat I1_2   = I1.mul(I1);        // I1^2
    Mat I1_I2  = I1.mul(I2);        // I1 * I2
    /*************************** END INITS **********************************/
    Mat mu1, mu2;                   // PRELIMINARY COMPUTING
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2   =   mu1.mul(mu1);
    Mat mu2_2   =   mu2.mul(mu2);
    Mat mu1_mu2 =   mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    Mat ssim_map;
    divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
    Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
    return mssim;
}


double ssim(Mat &i1, Mat & i2){
	const double C1 = 6.5025, C2 = 58.5225;
	int d = CV_32F;
	Mat I1, I2;
	i1.convertTo(I1, d);
	i2.convertTo(I2, d);
	Mat I1_2 = I1.mul(I1);
	Mat I2_2 = I2.mul(I2);
	Mat I1_I2 = I1.mul(I2);
	Mat mu1, mu2;
	GaussianBlur(I1, mu1, Size(11,11), 1.5);
	GaussianBlur(I2, mu2, Size(11,11), 1.5);
	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);
	Mat sigma1_2, sigam2_2, sigam12;
	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;
 
	GaussianBlur(I2_2, sigam2_2, Size(11, 11), 1.5);
	sigam2_2 -= mu2_2;
 
	GaussianBlur(I1_I2, sigam12, Size(11, 11), 1.5);
	sigam12 -= mu1_mu2;
	Mat t1, t2, t3;
	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigam12 + C2;
	t3 = t1.mul(t2);
 
	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigam2_2 + C2;
	t1 = t1.mul(t2);
 
	Mat ssim_map;
	divide(t3, t1, ssim_map);
	Scalar mssim = mean(ssim_map);
 
	double ssim = (mssim.val[0] + mssim.val[1] + mssim.val[2]) /3;
	return ssim;
}

int main()
{
    ifstream infile("SiEi_6.txt");
    int LUT[256][256];
    string s;
    int i = 0;
    while(getline(infile, s)) {
        if (!s.empty()) {
            vector<string> v;
            split(s, back_inserter(v));
            for (int j = 0; j < 256; ++j)
                LUT[i][j] = stoi(v[j]);
        }
        i++;            
    }
    // 生成高斯操作核
    static const int ksize = 3;
    double window[ksize][ksize];
    int window_int[ksize][ksize];

    static const double sigma = 1;
    static const double pi = 3.1415926;
    int sum = 0;
    int center = ksize / 2; // 模板的中心位置，也就是坐标的原点
    double x2, y2;
    for (int i = 0; i < ksize; i++) {
        x2 = pow(i - center, 2);
        for (int j = 0; j < ksize; j++) {
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2) / (2 * sigma * sigma));
            g /= 2 * pi * sigma * sigma;
            window[i][j] = g;
        }
    }
    double k = 1 / window[0][0]; // 将左上角的系数归一化为1
    for (int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            window[i][j] *= k;
            window_int[i][j] = round(window[i][j]);  // 如果左上角是0，需要对double类型用round()函数四舍五入
            sum += window_int[i][j];
            cout << window_int[i][j] << "\t";
        }
        cout << endl;
    }
    cout << sum << endl;

    Mat src = imread("lena512.bmp", IMREAD_GRAYSCALE); //从文件中加载灰度图像
    Mat dst = src.clone();

    // 高斯滤波
    for (int nrow = center; nrow < src.rows-center; nrow++) {
        for (int ncol = center; ncol < src.cols-center; ncol++) {
            int point = 0;
            for (int i = 0; i < ksize; i++) {
                for (int j = 0; j < ksize; j++) {
                    point += LUT[window_int[i][j]][src.ptr<uchar>(nrow+i-center)[ncol+j-center]];
                    // point += window_int[i][j] * src.ptr<uchar>(nrow+i-center)[ncol+j-center];
                }
            }
            dst.ptr<uchar>(nrow)[ncol] = point/sum;
        }
    }
    // imwrite("gas.bmp", dst);
    // 采用Unsharpen Mask算法锐化
    Mat usm;
    addWeighted(src, 1.5, dst, -0.5, 0, usm);

    imshow("src", src);
    imshow("dst", dst);
    imshow("usm", usm);
    cout << "dst psnr " << getPSNR(src, dst) << endl;
    cout << "dst ssim " << ssim(src, dst) * 3 * 100 << endl;
    cout << "usm psnr " << getPSNR(src, usm) << endl;
    cout << "usm ssim " << ssim(src, usm) * 3 * 100 << endl;
    waitKey();

    return 0;

}