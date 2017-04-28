//
// Created by slowbro on 28.04.17.
//
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>
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

void CopyLine(Mat &stock, Mat &dst, ushort min_energy_index, int j);

Mat GetEnergy(Mat const &im);

Mat MyFilter(Mat & src);

Mat ColoriseImage(Mat const & calc_energy, Mat const & stock_image);

Mat CalcEnergy(Mat const & enrg);

void DeleteVertiaclLine(Mat &stock);

Mat DeleteVectors(Mat const &, size_t const);