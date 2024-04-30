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
    
    
    // Definindo o tamanho do kernel
    int rows = 5;
    int cols = 5;

    // Definindo o desvio padrão
    float sigma = 1.0;
    
    // Gerando o kernel gaussiano
      Mat kernel = gauss_kernel(rows, cols, sigma);

      // Exibindo o kernel
      for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
          cout << kernel.at<float>(i, j) << " ";
        }
        cout << endl;
      }

    Mat filteredImage = apply_kernel(img, kernel);
    
    // Exibe os resultados
    imshow("Imagem Original", img);
    imshow("Imagem Filtrada", filteredImage);

    waitKey(0);
    
    return 0;
}
