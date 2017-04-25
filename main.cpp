#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

unsigned char kern[9] = {
        0, 0, 0,
        0, 1, 1,
        0, 1, 0
};
size_t kern_size = 3;

Mat GetEnergy(Mat const &im){
    Mat En (Size(im.rows, im.cols), CV_8UC1);
    Mat EnBuf;//(Size(im.rows, im.cols), im.step);

    Mat kernel_matrix = Mat(3,3,CV_32FC1, kern);

    filter2D(im, EnBuf, -1, kernel_matrix, Point(-1, -1), 0, BORDER_DEFAULT);

    cout<<"OK"<<endl;

    for(size_t x = 0; x < im.rows; x++){
        for (size_t y = 0; y < im.cols; y++){
            En.at<uchar>(x, y) = (EnBuf.at<Vec3b>(x, y)[0] / 1 + EnBuf.at<Vec3b>(x, y)[1] / 1 + EnBuf.at<Vec3b>(x, y)[2] / 1) / 1;
            cout<< static_cast<int>(En.at<uchar>(x, y))<<" ";
        }
    }
    return En;
}

Mat MyFilter(Mat & src){


    Mat kernel_matrix = Mat(3,3,CV_8UC1, kern);

    size_t summ_k = 0;
    for (size_t kx = 0; kx < kern_size; kx++) {
        for (size_t ky = 0; ky < kern_size; ky++) {
            if (kernel_matrix.at<uchar>(kx, ky) != 0){
                summ_k += abs(kernel_matrix.at<uchar>(kx, ky));
            }
        }
    }
    if (kernel_matrix.at<uchar>(1, 1) != 0){
        summ_k -= abs(kernel_matrix.at<uchar>(1, 1));
    }

    Mat enrg(Size(src.cols, src.rows), CV_8UC1);

    unsigned char diff = 0;


    for (size_t x = 0 + 1; x < src.rows - 1; x++){
        for (size_t y = 0 + 1; y < src.cols - 1; y++){
            for (int color = 0; color < 3; color++) {
                for (size_t kx = 0; kx < kern_size; kx++){
                    for (size_t ky = 0; ky < kern_size; ky++) {
                        if (kernel_matrix.at<uchar>(kx, ky) != 0) {
                            diff += abs(src.at<Vec3b>(x, y)[color] -
                                        src.at<Vec3b>(x - 1 + kx, y - 1 + ky)[color] * kernel_matrix.at<uchar>(kx, ky));
                        }
                    }
                }
            }
            //cout<<diff<<" ";
            enrg.at<uchar>(x, y) = diff / (1);
            diff = 0;
        }
    }
    return enrg;
}

Mat CalcEnergy(Mat &enrg){
    Mat calc_energ(Size(enrg.cols, enrg.rows), CV_16UC1);

    for(size_t y = 0; y < enrg.cols; y++){
        calc_energ.at<uchar>(0, y) = enrg.at<uchar>(0, y);
    }

    for (size_t x = 1; x < enrg.rows; x++){
        for(size_t y = 0; y < enrg.cols; y++){
            unsigned short buff = calc_energ.at<ushort>(x - 1, y);
            if ((y > 0) && (calc_energ.at<ushort>(x - 1, y - 1) < buff)){
                buff = calc_energ.at<ushort>(x - 1, y - 1);
            }
            if ((y + 1 < enrg.cols) && (calc_energ.at<ushort>(x - 1, y + 1) < buff)){
                buff = calc_energ.at<ushort>(x - 1, y + 1);
            }
            calc_energ.at<ushort>(x, y) = enrg.at<uchar>(x, y) + buff;
        }
    }
    return calc_energ;
}
void GetMinVector(Mat const & vec_energ, Mat & dst){

    size_t Xmin = 0;
    size_t Ymin = vec_energ.rows - 1;
    size_t minbuff = vec_energ.at<ushort>(Ymin, 10);

    for (size_t i = 1; i < vec_energ.cols; i++){
        cout<<vec_energ.at<ushort>(Ymin, i)<<" ";
        if (minbuff > vec_energ.at<ushort>(Ymin, i) && vec_energ.at<ushort>(Ymin, i) != 0){
            Xmin = i;
            minbuff = vec_energ.at<ushort>(Ymin, i);
        }
    }
    cout<<Xmin<<" - "<<minbuff<<" ";
    dst.at<Vec3b>(Ymin, Xmin)[0] = 0;
    dst.at<Vec3b>(Ymin, Xmin)[1] = 255;
    dst.at<Vec3b>(Ymin, Xmin)[2] = 0;

    for (size_t x = vec_energ.rows - 2; x > 0; x--){
        int delt = 0;
        minbuff = vec_energ.at<ushort>(x, Xmin);
        if ((Xmin + 1 < vec_energ.cols) && (vec_energ.at<ushort>(x, Xmin + 1) < minbuff)){
            minbuff = vec_energ.at<ushort>(x, Xmin + 1);
            delt = 1;
        }
        if ((Xmin - 1 >= 0) && (vec_energ.at<ushort>(x, Xmin - 1) < minbuff)){
            minbuff = vec_energ.at<ushort>(x, Xmin - 1);
            delt = -1;
        }
        Xmin += delt;
        dst.at<Vec3b>(x, Xmin)[0] = 0;
        dst.at<Vec3b>(x, Xmin)[1] = 255;
        dst.at<Vec3b>(x, Xmin)[2] = 0;
        cout<<Xmin<<" - "<<minbuff<<" ";
    }
}
/*
Mat (Mat const & im){
    vector <vector<int> >
}*/


int main() {

    string imn = "/home/slowbro/Pictures/2.jpg";

    Mat stock = imread(imn);
    namedWindow("stock", CV_WINDOW_AUTOSIZE);


    cout<<"[i] file open: "<<imn <<endl;
    Mat Energy = MyFilter(stock);

    namedWindow(imn, CV_WINDOW_AUTOSIZE);


    Mat Calc_Energy = CalcEnergy(Energy);
    GetMinVector(Calc_Energy, stock);

    namedWindow("calcenrgy", CV_WINDOW_AUTOSIZE);
    resize(Calc_Energy, Calc_Energy, Size(600, 600), INTER_CUBIC);
    imshow("calcenrgy", Calc_Energy);

    resize(Energy, Energy, Size(600, 600), INTER_CUBIC);
    imshow(imn, Energy);

    resize(stock, stock, Size(600, 600), INTER_CUBIC);
    imshow("stock", stock);


    //Mat VectorEnergy(stock.rows, stock.cols, CV_8UC1);


    while(1) {
        char key = waitKey(33);
        if (key == 27) {
            break;
        }
    }

    std::cout << "Hello, World!" << std::endl;
    return 0;
}