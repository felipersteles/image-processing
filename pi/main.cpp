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

int main() {
    
    // Carrega a imagem do experimento
    Mat img = imread("/Users/felipeteles/Development/ufma/pi/pi/data/pancreas/1-010.jpg", IMREAD_GRAYSCALE);

    // log de erros
    if (img.empty()) {
        cout << "Invalid image or path." << endl;
        return -1;
    }
    
    Mat imgBlur = apply_blur(img);
    
    Mat kmeans = apply_kmeans(imgBlur);
    
    // Exibe os resultados
    imshow("Imagem Original", img);
    imshow("Imagem depois do Kmeans", kmeans);
    
    
    waitKey(0);
    
    destroyAllWindows();
    return 0;
}
