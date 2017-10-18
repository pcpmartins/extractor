#ifndef UTILITY_H
#define UTILITY_H
#include <iostream>
#include <string>
#include<numeric>
#include <opencv2/core.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
class utility
{
    public:
        utility();
        virtual ~utility();
        std::string printHelp();
        void drawHist(Mat h_hist,int histSize);
        Mat resizeRuleImg(Mat ruleImage,Mat frame);
        float angleBetween(const Point &v1, const Point &v2);
        float innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1);

    protected:

    private:
};

#endif // UTILITY_H
