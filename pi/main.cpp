//
//  main.cpp
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//

#include <opencv2/opencv.hpp>
#include "lib.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    
    // Carrega a imagem do experimento
    Mat img = imread("/Users/felipeteles/Development/ufma/dataset/train/img/pancreas_095_46.png", IMREAD_GRAYSCALE);
    Mat mask = imread("/Users/felipeteles/Development/ufma/dataset/train/mask/pancreas_095_46.png", IMREAD_GRAYSCALE);

    // log de erros
    if (img.empty()) {
        cout << "Invalid image or path." << endl;
        return -1;
    }
    
    
    // pancrease segmentation function
    pancreas_segmentation(img,mask);
    
    waitKey(0);
    
    destroyAllWindows();
    return 0;
}
