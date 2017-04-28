
#include "HelpFunctions.h"

using namespace std;
using namespace cv;


int main() {

    string imn = "/home/slowbro/Pictures/5.jpg";

    Mat stock = imread(imn);
    namedWindow("stock", CV_WINDOW_AUTOSIZE);


    cout<<"[i] file open: "<<imn <<endl;

    Mat Final = DeleteVectors(stock, 400);
    imwrite("/home/slowbro/Pictures/5new.jpg", Final);

    return 0;
}
