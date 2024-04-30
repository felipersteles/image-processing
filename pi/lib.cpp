//
//  lib.cpp
//  pi
//
//  Created by Felipe Teles on 15/04/24.
//
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// Função para quantizar os valores de pixel
Mat quantize_image(Mat img,int maxLevel, int level) {
  Mat quantized = img.clone();

    int interval = maxLevel / level;
  for (int i = 0; i < img.rows; i++) {
    for (int j = 0; j < img.cols; j++) {
        // Encontrar o nível de quantização para o valor Y
        uchar& pixel = quantized.at<uchar>(i, j, 0);
        pixel = (pixel / interval) * interval + interval / 2;
    }
  }

  return quantized;
}

// Calcula o histograma da imagem
vector<int> calculate_histogram(const Mat &image, int max_lvl) {
    vector<int> histogram(max_lvl, 0);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            histogram[(int)image.at<uchar>(i, j)]++;
        }
    }
    
    return histogram;
}

// Desenha o histograma de acordo com um nivel maximo e uma altura
Mat draw_histogram(const vector<int>& histogram, int max_lvl, int height) {
    int histSize = max_lvl;
    int histHeight = height;
    
    Mat histImage(histHeight, histSize, CV_8UC3, Scalar(0,0,0));

    // 1: normalizar o histograma
    int maxHist = *max_element(histogram.begin(), histogram.end());
    // evita divisao por zero
    if (maxHist == 0) maxHist = 1;

    // 2: desenha a linha
    for (int i = 0; i < histSize; i++) {
        float binHeight = ((float)histogram[i] / maxHist) * histHeight;
        line(histImage, Point(i, histHeight),
             Point(i, histHeight - cvRound(binHeight)),
             Scalar(255, 255, 255));
    }
    return histImage;
}

// calcula a aparicao dos valores dos pixels
vector<int> caculate_cumulative_histogram(const Mat &image, int max_lvl) {
    vector<int> histogram = calculate_histogram(image, max_lvl);

    vector<int> cumulative_histogram(256, 0);
    cumulative_histogram[0] = histogram[0];
    for (int i = 1; i < max_lvl; i++) {
        cumulative_histogram[i] = cumulative_histogram[i - 1] + histogram[i];
    }
    
    return cumulative_histogram;
}

// Aplica a especificação do histograma com base num histograma de referencia
Mat match_histograms(const Mat &target, const vector<int> &source_cumalative_histogram, int max_lvl) {

    // calcula o histograma da imagem alvo
    vector<int> target_hist = calculate_histogram(target, max_lvl);
    
    int total_pixels = target.rows * target.cols;
    vector<uchar> lut(max_lvl, 0);

    // 1: comparação do valor das probabilidades
    for (int i = 0; i < max_lvl; i++) {
        int j = 0;
        double min_diff = abs(target_hist[i] - source_cumalative_histogram[j]);
        for (int k = 0; k < max_lvl; k++) {
            double diff = abs(target_hist[i] - source_cumalative_histogram[k]);
            if (diff < min_diff) {
                min_diff = diff;
                j = k;
            }
        }
        
        lut[i] = uchar(j);
    }
    
    // criar imagem resultante
    Mat matched = target.clone();
    
    // 2: aplicando a especificação
    for (int i = 0; i < target.rows; i++) {
        for (int j = 0; j < target.cols; j++) {
            matched.at<uchar>(i, j) = lut[matched.at<uchar>(i, j)];
        }
    }
    
    return matched;
}

// função que gera o kernel gaussiano com base em um desvio padrão
Mat gauss_kernel(int kernel_size_x, int kernel_size_y, float sigma){
    
    float sum_term {0};
    Mat kernel = Mat::zeros(kernel_size_x, kernel_size_y, CV_32F);
    
    // criar kernel
    // formula do kernel: G(x, y) = (1 / (2 * M_PI * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2)) -> (fonte: gemini do google)
    for (int i = -kernel_size_y/2; i <= kernel_size_y/2; i++)
    {
        for (int j = -kernel_size_x/2; j <= kernel_size_x/2; j++)
        {
            kernel.at<float>(i + kernel_size_y/2, j + kernel_size_x/2) = exp(-(pow(i,2) + pow(j,2))/(2*pow(sigma,2)));
            sum_term += kernel.at<float>(i + kernel_size_y/2,j + kernel_size_x/2);
        }
    }
    
    kernel /= sum_term;
    
    return kernel;
}

// função que aplica o kernel em uma imagem
Mat apply_kernel(Mat img, Mat kernel){
    Mat result;
    
    // função nativa do opencv
    filter2D(img, result, -1, kernel);
    
    return result;
}
