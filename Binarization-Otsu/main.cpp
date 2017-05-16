#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <time.h>

int OsuThreshold(const cv::Mat& img)
{
    uchar minI = img.at<uchar>(0, 0);
    uchar maxI = img.at<uchar>(0, 0);

    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            minI = std::min(minI, img.at<uchar>(i, j));
            maxI = std::max(maxI, img.at<uchar>(i, j));
        }
    }

    int histSize = maxI - minI + 1;
    std::vector<int> hist(histSize, 0);

    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            hist[img.at<uchar>(i, j) - minI]++;
        }
    }

    int m = 0;
    int n = 0;
    for (int i = 0; i < int(hist.size()); ++i)
    {
        m += i * hist[i];
        n += hist[i];
    }

    double maxSigma = -1;
    int threshold = 0;
    int alpha1 = 0;
    int beta1 = 0;

    for (int i = 0; i < int(hist.size()); ++i)
    {
        alpha1 += i * hist[i];
        beta1 += hist[i];

        double w1 = double(beta1) / n;
        double a = double(alpha1) / beta1 - double(m - alpha1) / (n - beta1);
        double sigma = w1 * (1 - w1) * a * a;

        if (sigma > maxSigma)
        {
            maxSigma = sigma;
            threshold = i;
        }
    }

    return threshold + minI;

}

cv::Mat OsuBinarization(const cv::Mat& img)
{
    cv::Mat dst = cv::Mat(img.rows, img.cols, CV_8UC1);
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols; ++j)
        {
            cv::Vec3b pixel = img.at<cv::Vec3b>(i, j);
            dst.at<uchar>(i, j) = uchar(0.2125 * pixel.val[0] + 0.7154 * pixel.val[1] + 0.0721 * pixel.val[2]);
        }
    }

    int threshold = OsuThreshold(dst);
    for (int i = 0; i < dst.rows; ++i)
    {
        for (int j = 0; j < dst.cols; ++j)
        {
            dst.at<uchar>(i, j) = dst.at<uchar>(i, j) < threshold ? 0 : 255;
        }
    }

    return dst;
}

int main(int argc, char** argv)
{

#ifdef FOR_EXAMPLES  
    int countImg = 15;

    double timeSum = 0;

    for (int i = 1; i <= countImg; ++i)
    {
        std::string name;
        name.clear();
        if (i < 10)
        {
            name = '0';
            name += char(i + '0');
        }
        else
        {
            name = char(i / 10 + '0');
            name += char(i % 10 + '0');
        }      

        cv::Mat image;
        image = cv::imread(name + ".jpg", CV_LOAD_IMAGE_COLOR);

        clock_t start, end;
        
        start = clock();
        cv::Mat binarizationImage = OsuBinarization(image);
        end = clock();

        timeSum += double(end - start) / (image.rows * image.cols);

        imwrite(name + ".bmp", binarizationImage);        
    }

    std::cout << "average time: " << timeSum / countImg << '\n';

#else    
    if (argc != 3)
    {
        std::cout << "You must give parametrs (1) WHAT BINARIZATION (2) WHERE RESULT";
    }
    else
    {
        cv::Mat image;
        image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
        cv::Mat binarizationImage = OsuBinarization(image);
        imwrite(argv[2], binarizationImage);
    }
#endif
    return 0;
}