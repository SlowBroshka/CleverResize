//
// Created by slowbro on 28.04.17.
//
#include "HelpFunctions.h"

template <typename T>
T mx(T a1, T a2){
    return ((a1 > a2) ? a1 : a2);
}
template <typename T>
T mn(T a1, T a2){
    return ((a1 < a2) ? a1 : a2);
}

template <typename T>
bool sort(T a, T b){
    return (a < b);
}
Mat deltResize(const Mat&img, int const width, int const height){
    if ((img.cols + width > 1) && (img.rows + height > 1)){
        Mat dst = ResizeX(img, width);
        dst = Rotate(dst, 90);
        dst = ResizeX(dst, height);
        dst = Rotate(dst, 270);
        return dst;
    }else{
        cout<<"bad size"<<endl;
        return img;
    }
}

Mat Resize(const Mat &img, ushort const width, ushort const height){
    int deltaX = width - img.cols;
    int deltaY = height - img.rows;
    Mat dst = ResizeX(img, deltaX);
    dst = Rotate(dst, 90);
    dst = ResizeX(dst, deltaY);
    dst = Rotate(dst, 270);
    return dst;
}
Mat Rotate(const Mat &img, double const angle){
    Mat dst;

    Size img_sz = img.size();
    Size dst_sz(img_sz.height, img_sz.width);

    int len = std::max(img.cols, img.rows);
    Point2f center(len/2., len/2.);
    Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
    warpAffine(img, dst, rot_mat, dst_sz);

    return dst;
}
Mat ResizeX(const Mat & img, int const width){
    Mat Final;
    if (width == 0){
        return img;
    }
    if (width > 0){
        Final = AddVectors(img, static_cast<ushort>(width));
    }else{
        Final = DeleteVectors(img, abs(width));
    }
    return Final;
}

vector<pair<int, ushort> > GetNextLine(vector<VerticalLine> const &VertLines, ushort row){

    vector <pair<int, ushort> > horizontal_line;
    for(auto i = 0; i < VertLines.size(); i++){
        horizontal_line.push_back(make_pair(VertLines[i].Merge ,VertLines[i].x_points[row]));
    }
    return horizontal_line;
}

ushort GetXmin(Mat const &calc_energy, int const row){
    ushort min = calc_energy.at<ushort>(row, 0);
    ushort x_min = 0;
    for (auto i = 0; i < calc_energy.cols; i++){
        if (calc_energy.at<ushort>(row, i) < min){
            min = calc_energy.at<ushort>(row, i);
            x_min = i;
        }
    }
    return x_min;
}

void DrawLine(Mat &draw, vector<ushort> x_data, Scalar Color){
    int j = 0;
    for (auto i = draw.rows - 1; (i >= 0  && j < x_data.size()); i--){
        draw.at<Vec3b>(i , x_data[j])[0] = static_cast<uchar>(Color[2]);
        draw.at<Vec3b>(i , x_data[j])[1] = static_cast<uchar>(Color[1]);
        draw.at<Vec3b>(i , x_data[j])[2] = static_cast<uchar>(Color[0]);
        j++;
    }
}

Mat DrawLines(Mat const & _draw, vector<VerticalLine> const data_lines){

    Mat draw(_draw);

    for(auto row = draw.rows - 1; row >= 0; row--) {
        for (auto vr = 0; vr < data_lines.size(); vr++) {
            draw.at<Vec3b>(row, data_lines[vr].x_points[(draw.rows - 1) - row])[0] = 0;
            draw.at<Vec3b>(row, data_lines[vr].x_points[(draw.rows - 1) - row])[1] = 255;
            draw.at<Vec3b>(row, data_lines[vr].x_points[(draw.rows - 1) - row])[2] = 0;
        }
    }
    return draw;
}

bool second_sort(pair<int, ushort> a, pair<int, ushort> b){
    return (a.second < b.second);
}

void LongerHorisontalLine(Mat &stock, Mat &dst, vector<pair<int, ushort>> &AddIndexes, int const RowNumber){

    int offset = 0;
    sort(AddIndexes.begin(), AddIndexes.end(), second_sort);

    for(auto i = 0; i < stock.cols; i++){
        dst.at<Vec3b>(RowNumber, i + offset) = stock.at<Vec3b>(RowNumber, i);
        if ((i < AddIndexes[offset].second) || (offset >= AddIndexes.size())){
            dst.at<Vec3b>(RowNumber, i + offset) = stock.at<Vec3b>(RowNumber, i);
        }else {
            if (i == AddIndexes[offset].second) {
                if (AddIndexes[offset].first == 1) {
                    dst.at<Vec3b>(RowNumber, i + offset) = stock.at<Vec3b>(RowNumber, i);
                    offset += 1;
                    dst.at<Vec3b>(RowNumber, i + offset)[0] = static_cast<uchar>((stock.at<Vec3b>(RowNumber, i + 1)[0] +
                                                                                  stock.at<Vec3b>(RowNumber, i)[0]) / 2);
                    dst.at<Vec3b>(RowNumber, i + offset)[1] = static_cast<uchar>((stock.at<Vec3b>(RowNumber, i + 1)[1] +
                                                                                  stock.at<Vec3b>(RowNumber, i)[1]) / 2);
                    dst.at<Vec3b>(RowNumber, i + offset)[2] = static_cast<uchar>((stock.at<Vec3b>(RowNumber, i + 1)[2] +
                                                                                  stock.at<Vec3b>(RowNumber, i)[2]) / 2);
                }else{
                    if (AddIndexes[offset].first == -1) {
                        dst.at<Vec3b>(RowNumber, i + offset)[0] = static_cast<uchar>((stock.at<Vec3b>(RowNumber, i)[0] +
                                                                                      dst.at<Vec3b>(RowNumber, i + offset - 1)[0]) / 2);
                        dst.at<Vec3b>(RowNumber, i + offset)[1] = static_cast<uchar>((stock.at<Vec3b>(RowNumber, i)[1] +
                                                                                      dst.at<Vec3b>(RowNumber, i + offset - 1)[1]) / 2);
                        dst.at<Vec3b>(RowNumber, i + offset)[2] = static_cast<uchar>((stock.at<Vec3b>(RowNumber, i)[2] +
                                                                                      dst.at<Vec3b>(RowNumber, i + offset - 1)[2]) / 2);
                        offset += 1;
                        dst.at<Vec3b>(RowNumber, i + offset) = stock.at<Vec3b>(RowNumber, i);
                    }else{
                        dst.at<Vec3b>(RowNumber, i + offset) = stock.at<Vec3b>(RowNumber, i);
                        offset += 1;
                        dst.at<Vec3b>(RowNumber, i + offset) = stock.at<Vec3b>(RowNumber, i);
                    }
                }
            }else{
                if (i > AddIndexes[offset].second){
                    i-=2;
                }
            }
        }
    }
}

void CopyLine(const Mat &stock, Mat &dst, const ushort min_energy_index, const int j) {
    for(int i = 0; i < min_energy_index; i++){
        dst.at<Vec3b>(j, i) = stock.at<Vec3b>(j, i);
    }
    for(int i = min_energy_index + 1; i < stock.cols; i++){
        dst.at<Vec3b>(j, i - 1) = stock.at<Vec3b>(j, i);
    }
}
void CopyEnergyLine(const Mat &stock, Mat &dst, const ushort min_energy_index, const int j) {
    for(int i = 0; i < min_energy_index; i++){
        dst.at<uchar>(j, i) = stock.at<uchar>(j, i);
    }
    for(int i = min_energy_index + 1; i < stock.cols; i++){
        dst.at<uchar>(j, i - 1) = stock.at<uchar>(j, i);
    }
}

Mat DeleteVectors(Mat const &stock, size_t const LinesNumber){

    Mat Buff = stock.clone();
    Mat energy = GetEnergy(Buff);
    int timeBar = 0;
    int lastTime = -1;

    for (size_t n = 0; n < LinesNumber; n++){
        DeleteVertiaclLine(Buff, energy);
        timeBar = n  * 100 / LinesNumber;
        if (timeBar % 10 == 0 && timeBar != lastTime){
            lastTime = timeBar;
            cout<<timeBar / 10<<endl;
        }
    }
    return Buff;
}
Mat AddVectors(const Mat &stock, const ushort LinexNumber){

    Mat Buff = stock.clone();
    ushort block_size;
    ushort function = 7;
    block_size = function;

    for (int i = LinexNumber; i > 0; ){
        Mat energy = GetEnergy(Buff);
        vector<VerticalLine> x_inedx_pixels;

        if ( i - block_size > 0){
            i -= block_size;
            x_inedx_pixels= AddXPixels(energy, block_size);
        }else{
            x_inedx_pixels = AddXPixels(energy, i);

//            cout<<"x_index_pixels= "<<x_inedx_pixels.size()<<endl;
//            Mat draw2 = DrawLines(Buff, x_inedx_pixels);
//            imwrite("/home/slowbro/Pictures/draw2.jpg", draw2);

            i = 0;
        }
        AddVerticalLine(Buff, x_inedx_pixels);

        function = trunc(function * 1.5);
        block_size = function;
        cout<<"now i = "<<i<<endl;
    }
    return Buff;
}

Mat GetEnergy(Mat const &src){

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

    Mat enrg(Size(src.cols, src.rows), CV_8UC1, Scalar(255, 255, 255));

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
            enrg.at<uchar>(x, y) = diff;
            diff = 0;
        }
    }
    return enrg;
}

Mat ColoriseImagebyStock(Mat const &stock){
    Mat energy = GetEnergy(stock);
    Mat calc_energy = CalcEnergy(energy);
    return ColoriseImage(calc_energy);
}

Mat ColoriseImage(Mat const & calc_energy){
    Mat colorise_image(Size(calc_energy.cols, calc_energy.rows), CV_8UC3, Scalar(0, 0, 255));
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
            calc_energ.at<ushort>(x, y) = ((enrg.at<uchar>(x, y) + buff > numeric_limits<ushort>::max()) ?
                                          numeric_limits<ushort>::max() : enrg.at<uchar>(x, y) + buff);
        }
    }
    return calc_energ;
}

void DeleteVertiaclLine(Mat &stock, Mat &energy) {

    Mat dst(Size(stock.cols - 1, stock.rows), stock.type());
    Mat _energy(Size(energy.cols - 1, energy.rows), energy.type());
    Mat calc_energy = CalcEnergy(energy);

    ushort min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, 0);
    ushort min_energy_index = 0;

    for (auto i = 0; i < calc_energy.cols; i++){
        if (calc_energy.at<ushort>(calc_energy.rows - 1, i) < min_energy){
            min_energy = calc_energy.at<ushort>(calc_energy.rows - 1, i);
            min_energy_index = static_cast<ushort>(i);
        }
    }
    CopyLine(stock, dst, min_energy_index, stock.rows - 1);
    CopyEnergyLine(energy, _energy, min_energy_index, stock.rows - 1);
    for (int j = stock.rows - 2; j >= 0; j--) {
        int delt = GetOffset(calc_energy, j, min_energy_index);
        min_energy_index += delt;
        CopyLine(stock, dst, min_energy_index, j);
        CopyEnergyLine(energy, _energy, min_energy_index, j);
    }
    stock = dst;
    energy = _energy;
}

int GetOffset(Mat const &calc_energy, int const row, ushort const x_energy) {
    ushort energy = calc_energy.at<ushort>(row , x_energy);
    int delt = 0;

    if ((x_energy + 1 < calc_energy.cols) && (calc_energy.at<ushort>(row, x_energy + 1) < energy)){
            energy = calc_energy.at<ushort>(row, x_energy + 1);
            delt = 1;
        }
    if ((x_energy - 1 >= 0) && (calc_energy.at<ushort>(row, x_energy - 1) < energy)){
            delt = -1;
        }
    return delt;
}

vector<VerticalLine> AddXPixels(const Mat &_energy, const ushort pixnums){

    Mat energy(_energy);
    Mat calc_energy = CalcEnergy(energy);

    vector<VerticalLine> x_cord_of_lines;

    int last_row = calc_energy.rows - 1;
    for (auto count = 0; count < pixnums; count++) {
        VerticalLine buff;
        ushort x_min = GetXmin(calc_energy, calc_energy.rows - 1);
        buff.start_x_cord = x_min;

        int offset = GetOffset(calc_energy, last_row, x_min);
        buff.Merge = offset;
        buff.x_points.push_back(x_min);
        energy.at<uchar>(energy.rows - 1, x_min) = numeric_limits<uchar>::max();

        for (auto i = energy.rows - 2; i >= 0; i--) {
            x_min += GetDelt(calc_energy, i, x_min);
            buff.x_points.push_back(x_min);
            energy.at<uchar>(i, x_min) = numeric_limits<uchar>::max();
        }
        x_cord_of_lines.push_back(buff);
        calc_energy = CalcEnergy(energy);
    }

    return x_cord_of_lines;
}

void AddVerticalLine(Mat &stock, vector<VerticalLine> &Lines){

    if (Lines.size() == 0){
        return;
    }
    Mat dst(Size(stock.cols + Lines.size(), stock.rows), stock.type(), Scalar(0, 255, 0));

    for(auto i = stock.rows - 1; i >= 0; i--) {
        vector<pair<int, ushort> > Line = GetNextLine(Lines, (stock.rows - 1) - i);
        LongerHorisontalLine(stock, dst, Line, i);
    }
    stock = dst;
}

int GetDelt(Mat const &calc_energy, int row, ushort min_energy_index) {
    int delt = 0;
    if ((min_energy_index + 1 < calc_energy.cols) && (min_energy_index > 0)){
        if ((calc_energy.at<ushort>(row, min_energy_index) <= calc_energy.at<ushort>(row, min_energy_index + 1)) &&
            (calc_energy.at<ushort>(row, min_energy_index) <= calc_energy.at<ushort>(row, min_energy_index - 1))){
            delt = 0;
        }else if((calc_energy.at<ushort>(row, min_energy_index - 1) < calc_energy.at<ushort>(row, min_energy_index + 1)) &&
                 (calc_energy.at<ushort>(row, min_energy_index - 1) < calc_energy.at<ushort>(row, min_energy_index))){
            delt = -1;
        }else{
            delt = 1;
        }
    }else if(min_energy_index == 0){
        if (calc_energy.at<ushort>(row, min_energy_index) < calc_energy.at<ushort>(row, min_energy_index + 1)){
            delt = 0;
        }else{
            delt = 1;
        }
    }else{
        if (calc_energy.at<ushort>(row, min_energy_index) < calc_energy.at<ushort>(row, min_energy_index - 1)){
            delt = 0;
        }else{
            delt = -1;
        }
    }
    return delt;
}

/*
int GetOffset(Mat &const calc_energy, const int last_row, const ushort x_min) {
    int offset = 0;

    if ((((x_min > 0) && (x_min + 1 < calc_energy.cols)) &&
         (calc_energy.at<ushort>(last_row, x_min + 1) < calc_energy.at<ushort>(last_row, x_min - 1))) ||
        ((x_min == calc_energy.cols - 1) && (calc_energy.at<ushort>(last_row, x_min) < calc_energy.at<ushort>(last_row, x_min - 1)))){
        offset = -1;
    }else if ((((x_min + 1 < calc_energy.cols) && (x_min > 0)) &&
               (calc_energy.at<ushort>(last_row, x_min - 1) < calc_energy.at<ushort>(last_row, x_min + 1))) ||
              ((x_min == 0) && (calc_energy.at<ushort>(last_row, x_min) > calc_energy.at<ushort>(last_row, x_min + 1)))){
        offset = 1;
    }else{
        offset = 0;
    }
    return offset;
}*/