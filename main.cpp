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

const uchar COLOR_CONST = 120;

Mat GetEnergy(Mat const &im){
    Mat En (Size(im.rows, im.cols), CV_8UC1);
    Mat EnBuf;//(Size(im.rows, im.cols), im.step);

    Mat kernel_matrix = Mat(3,3,CV_32FC1, kern);

    filter2D(im, EnBuf, -1, kernel_matrix, Point(-1, -1), 0, BORDER_DEFAULT);


    for(size_t x = 0; x < im.rows; x++){
        for (size_t y = 0; y < im.cols; y++){
            En.at<uchar>(x, y) = (EnBuf.at<Vec3b>(x, y)[0] / 1 + EnBuf.at<Vec3b>(x, y)[1] / 1 + EnBuf.at<Vec3b>(x, y)[2] / 1) / 1;
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
            enrg.at<uchar>(x, y) = diff / (1);
            diff = 0;
        }
    }
    return enrg;
}

void ZeroChek(Mat const & src){
    cout<<"Energy Matrix"<<endl;
    for (size_t i = 0; i < src.rows; i++){
        for(size_t j = 0; j < src.cols; j++){
            cout<< static_cast<int>(src.at<uchar>(i, j))<<"\t";
        }
        cout<<endl;
    }
}

Mat ColoriseImage(Mat const & calc_energy, Mat const & stock_image){
    Mat colorise_image(Size(stock_image.cols, stock_image.rows), CV_8UC3, Scalar(0, 0, 255));
    cvtColor(colorise_image, colorise_image, CV_BGR2HSV);

    ushort max_energy = calc_energy.at<ushort>(calc_energy.rows - 1, 0);
    ushort min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, 0);

    for (size_t j = 0; j < calc_energy.cols; j++){
        if (calc_energy.at<ushort>(calc_energy.rows - 1, j) > max_energy){
            max_energy = calc_energy.at<ushort>(calc_energy.rows - 1, j);
        }
        if (calc_energy.at<ushort>(calc_energy.rows - 1, j) < min_energy){
            min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, j);
        }
    }
    for (size_t i = 0; i < colorise_image.rows; i++){
        for (size_t j = 0; j < colorise_image.cols; j++){
            colorise_image.at<Vec3b>(i, j)[0] = static_cast<uchar>(
                    COLOR_CONST - (COLOR_CONST * (calc_energy.at<ushort>(i, j) - min_energy)) / max_energy);
            colorise_image.at<Vec3b>(i, j)[1] = 255;
            colorise_image.at<Vec3b>(i, j)[2] = 255;
        }
    }
    cvtColor(colorise_image, colorise_image, CV_HSV2BGR);
    return  colorise_image;
}

Mat CalcEnergy(Mat &enrg){
    Mat calc_energ(Size(enrg.cols, enrg.rows), CV_16UC1);


    for(size_t y = 0; y < enrg.cols; y++){
        calc_energ.at<ushort>(0, y) = enrg.at<uchar>(0, y);
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

Mat DeleteVertiaclLine(Mat const & stock){

    Mat dst(Size(stock.cols - 1, stock.rows), stock.type());
    Mat energy = GetEnergy(stock);

    Mat calc_energy = CalcEnergy(energy);

    ushort min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, 0);
    ushort min_energy_index = 0;

    for (size_t i = 0; i < calc_energy.cols; i++){
        if (calc_energy.at<ushort>(calc_energy.rows - 1, i) < min_energy){
            min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, i);
            min_energy_index = i;
        }
    }
    cout<<min_energy<<" min_index "<<min_energy_index<<endl;
    for(int i = 0; i < stock.cols; i++){
        if (min_energy_index != i){
            dst.at<Vec3b>(stock.rows - 1, i) = stock.at<Vec3b>(stock.rows - 1, i);
        }
    }
    for (size_t j = stock.rows - 2; j > 0; j--) {
        min_energy = calc_energy.at<ushort>(j , min_energy_index);
        int delt = 0;

        if ((min_energy_index + 1 < calc_energy.cols) && (calc_energy.at<ushort>(j, min_energy_index + 1) < min_energy)){
            min_energy = calc_energy.at<ushort>(j, min_energy_index + 1);
            delt = 1;
        }
        if ((min_energy_index - 1 >= 0) && (calc_energy.at<ushort>(j, min_energy_index - 1) < min_energy)){
            min_energy = calc_energy.at<ushort>(j, min_energy_index - 1);
            delt = -1;
        }
        min_energy_index += delt;
        for(size_t i = 0; i < stock.cols; i++){
            if (min_energy_index != i){
                dst.at<Vec3b>(j, i) = stock.at<Vec3b>(j, i);
            }
        }
    }
    return dst;
}

Mat DeleteVectors(Mat const & stock, size_t const LinesNumber){

    Mat Fin = DeleteVertiaclLine(stock);

    Mat Buff;

    for (size_t n = 1; n < LinesNumber; n++){
        Buff = DeleteVertiaclLine(Fin);
        Buff.copyTo(Fin);
    }

    return Fin;

}




int main() {

    string imn = "/home/slowbro/Pictures/5.jpg";

    Mat stock = imread(imn);
    namedWindow("stock", CV_WINDOW_AUTOSIZE);


    cout<<"[i] file open: "<<imn <<endl;
 //   Mat Energy = MyFilter(stock);

//    namedWindow(imn, CV_WINDOW_AUTOSIZE);
//
//
//    Mat Calc_Energy = CalcEnergy(Energy);
    Mat Final = DeleteVectors(stock, 100);
    imwrite("/home/slowbro/Pictures/5new.jpg", Final);

//    Mat Vec_Energy = ColoriseImage(Calc_Energy, stock);
//    namedWindow("ColoriseImage", CV_WINDOW_AUTOSIZE);
//    imwrite("/home/slowbro/Pictures/5Vec2.jpg", Vec_Energy);
//    resize(Vec_Energy, Vec_Energy, Size(600, 600), INTER_CUBIC);
//    imshow("ColoriseImage", Vec_Energy);
//
//
//    namedWindow("calcenrgy", CV_WINDOW_AUTOSIZE);
//    resize(Calc_Energy, Calc_Energy, Size(600, 600), INTER_CUBIC);
//    imshow("calcenrgy", Calc_Energy);
//
//    resize(Energy, Energy, Size(600, 600), INTER_CUBIC);
//    imshow(imn, Energy);
//
//    resize(stock, stock, Size(600, 600), INTER_CUBIC);
//    imshow("stock", stock);


    //Mat VectorEnergy(stock.rows, stock.cols, CV_8UC1);


    std::cout << "Hello, World!" << std::endl;
    return 0;
}