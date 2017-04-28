//
// Created by slowbro on 28.04.17.
//
#include "HelpFunctions.h"

void CopyLine(Mat &stock, Mat &dst, ushort min_energy_index, int j) {
    for(int i = 0; i < min_energy_index; i++){
        dst.at<Vec3b>(j, i) = stock.at<Vec3b>(j, i);
    }
    for(int i = min_energy_index + 1; i < stock.cols; i++){
        dst.at<Vec3b>(j, i - 1) = stock.at<Vec3b>(j, i);
    }
}

void CopyLineAddPixel(Mat &stock, Mat &dst, ushort x_index, int row_number){
    for (auto i = 0; i < x_index; i++){
        dst.at<Vec3b>(row_number, i) = stock.at<Vec3b>(row_number, i);
    }
    if ((x_index > 0) && (x_index + 1 < stock.cols)) {
        dst.at<Vec3b>(row_number, x_index)[0] = static_cast<uchar>((stock.at<Vec3b>(row_number, x_index - 1)[0] +
                stock.at<Vec3b>(row_number, x_index + 1)[0]) / 2);
        dst.at<Vec3b>(row_number, x_index)[1] = static_cast<uchar>((stock.at<Vec3b>(row_number, x_index - 1)[1] +
                stock.at<Vec3b>(row_number, x_index + 1)[1]) / 2);
        dst.at<Vec3b>(row_number, x_index)[2] = static_cast<uchar>((stock.at<Vec3b>(row_number, x_index - 1)[2] +
                stock.at<Vec3b>(row_number, x_index + 1)[2]) / 2);
    }else if (x_index == 0){
        dst.at<Vec3b>(row_number, x_index)[0] = stock.at<Vec3b>(row_number, x_index + 1)[0];
        dst.at<Vec3b>(row_number, x_index)[1] = stock.at<Vec3b>(row_number, x_index + 1)[1];
        dst.at<Vec3b>(row_number, x_index)[2] = stock.at<Vec3b>(row_number, x_index + 1)[2];
    }else{
        dst.at<Vec3b>(row_number, x_index)[0] = stock.at<Vec3b>(row_number, x_index - 1)[0];
        dst.at<Vec3b>(row_number, x_index)[1] = stock.at<Vec3b>(row_number, x_index - 1)[1];
        dst.at<Vec3b>(row_number, x_index)[2] = stock.at<Vec3b>(row_number, x_index - 1)[2];
    }
    for(auto i =0; i < x_index; i++){
        dst.at<Vec3b>(row_number, i + 1) = stock.at<Vec3b>(row_number, i);
    }
}

Mat GetEnergy(Mat const &im){
    Mat En (Size(im.cols, im.rows), CV_8UC1);
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

Mat DeleteVectors(Mat const & stock, size_t const LinesNumber){

    Mat Buff = stock.clone();

    for (size_t n = 0; n < LinesNumber; n++){
        DeleteVertiaclLine(Buff);
    }
    return Buff;
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

Mat CalcEnergy(Mat const &enrg){
    Mat calc_energ(Size(enrg.cols, enrg.rows), CV_16UC1);


    for(size_t y = 0; y < enrg.cols; y++){
        calc_energ.at<ushort>(0, y) = enrg.at<uchar>(0, y);
    }

    for (auto x = 1; x < enrg.rows; x++){
        for(auto y = 0; y < enrg.cols; y++){
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

void DeleteVertiaclLine(Mat &stock){

    Mat dst(Size(stock.cols - 1, stock.rows), stock.type());
    Mat energy = MyFilter(stock);
    Mat calc_energy = CalcEnergy(energy);

    ushort min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, 0);
    ushort min_energy_index = 0;

    for (auto i = 0; i < calc_energy.cols; i++){
        if (calc_energy.at<ushort>(calc_energy.rows - 1, i) < min_energy){
            min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, i);
            min_energy_index = i;
        }
    }

    cout<<"Min energy: "<<min_energy<<" X position: "<<min_energy_index<<endl;
    CopyLine(stock, dst, min_energy_index, stock.rows - 1);

    for (int j = stock.rows - 2; j >= 0; j--) {
        min_energy = calc_energy.at<ushort>(j , min_energy_index);
        int delt = 0;

        if ((min_energy_index + 1 < calc_energy.cols) && (calc_energy.at<ushort>(j, min_energy_index + 1) < min_energy)){
            min_energy = calc_energy.at<ushort>(j, min_energy_index + 1);
            delt = 1;
        }
        if ((min_energy_index - 1 >= 0) && (calc_energy.at<ushort>(j, min_energy_index - 1) < min_energy)){
            delt = -1;
        }
        min_energy_index += delt;

        CopyLine(stock, dst, min_energy_index, j);
    }

    stock = dst;
}

void AddVerticalLine(Mat &stock, vector<ushort> const & x_add_index){
    /*
     * сделать за 1 проход
     * x_add_index с каким сравнивать (левым или правым) (с тем у кого энерджи меньше) (тип куда сдвинется)
     *
     */

    if (x_add_index.size() == 0){
        return;
    }
    Mat dst(Size(stock.cols + x_add_index.size(), stock.rows), stock.type());
    Mat energy = MyFilter(stock);
    Mat calc_energy = CalcEnergy(energy);


    for (auto id = 0; id < x_add_index.size(); id++){
        ushort min_energy_index = x_add_index[0];
        ushort min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, x_add_index[id]);

        CopyLineAddPixel(stock, dst, min_energy_index, calc_energy.rows - 1);

        for (auto i = calc_energy.rows - 2; i > 0; i--){

            int delt = 0;

            if ((min_energy_index + 1 < calc_energy.cols) && (calc_energy.at<ushort>(i, min_energy_index + 1) < min_energy)){
                min_energy = calc_energy.at<ushort>(i, min_energy_index + 1);
                delt = 1;
            }
            if ((min_energy_index - 1 >= 0) && (calc_energy.at<ushort>(i, min_energy_index - 1) < min_energy)){
                delt = -1;
            }
            min_energy_index += delt;

            CopyLine(stock, dst, min_energy_index, i);
        }

    }
    stock = dst;
}




