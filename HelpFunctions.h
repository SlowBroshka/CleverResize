//
// Created by slowbro on 28.04.17.
//
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>
#include "Constants.h"

using  namespace cv;
using namespace std;
#ifndef CLEVERRESIZE_HELPFUNCTIONS_H
#define CLEVERRESIZE_HELPFUNCTIONS_H

#endif //CLEVERRESIZE_HELPFUNCTIONS_H

struct VerticalLine
{
    int Merge;
    int start_x_cord;
    vector<ushort> x_points;

};
Mat deltResize(const Mat&img, int const dwidth, int const dheight);
Mat Resize(const Mat &img, ushort const width, ushort const height);
Mat Rotate(const Mat &img, double const angle);
Mat ResizeX(const Mat & img, int const width);


void CopyLine(const Mat &stock, Mat &dst, const ushort min_energy_index, const int j);
void CopyEnergyLine(const Mat &stock, Mat &dst, const ushort min_energy_index, const int j);
void LongerHorisontalLine(Mat &stock, Mat &dst, vector<pair < int, ushort> > &AddIndexes, int const RowNumber);

Mat GetEnergy(Mat const &src);

Mat ColoriseImage(Mat const & calc_energy);
Mat ColoriseImagebyStock(Mat const &stock);

Mat DrawLines(Mat const & _draw, vector<VerticalLine> const data_lines);
void DrawLine(Mat &draw, vector<ushort> x_data, Scalar Color);

Mat CalcEnergy(Mat const & enrg);

void DeleteVertiaclLine(Mat &stock, Mat &energy);

Mat DeleteVectors(Mat const &, size_t const);
Mat AddVectors(const Mat &stock, const ushort LinexNumber);

vector<VerticalLine> AddXPixels(const Mat &_energy, const ushort pixnums);

void AddVerticalLine(Mat &stock, vector<VerticalLine> &Lines);

bool second_sort(pair<int, ushort> a, pair<int, ushort> b);

int GetDelt(Mat const &calc_energy, int row, ushort min_energy_index);
int GetOffset(Mat const &calc_energy, int const row, ushort const x_energy);

//int GetOffset(Mat &const calc_energy, const int last_row, const ushort x_min);