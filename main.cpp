
#include "HelpFunctions.h"


using namespace std;
using namespace cv;


int main() {

    string imn = "/home/slowbro/Изображения/img1.jpg";

    Mat stock = imread(imn);
    cout<<"[i] file open: "<<imn <<endl;
    //Mat energy = ColoriseImagebyStock(stock);
    //imwrite("/home/slowbro/Изображения/img2energy.jpg", energy);


    Mat Final = deltResize(stock, -500, 0);
    imwrite("/home/slowbro/Изображения/img5resize.jpg", Final);

    return 0;
}
