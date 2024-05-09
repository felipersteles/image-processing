//
//  main.cpp
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "lib.h"

using namespace cv;
using namespace std;

int main() {
    
    // Carrega a imagem do experimento
    Mat img = imread("/Users/felipeteles/Development/ufma/pi/pi/data/lena.jpeg", IMREAD_GRAYSCALE);

    // log de erros
    if (img.empty()) {
        cout << "Invalid image or path." << endl;
        return -1;
    }

    Mat otsu = otsu_threshold(img);
    
    // Exibe os resultados
    imshow("Imagem Original", img);
    imshow("Imagem com o OTSU Threshold", otsu);

    waitKey(0);
    
    return 0;
}
