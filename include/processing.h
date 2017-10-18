#ifndef PROCESSING_H
#define PROCESSING_H
#include "utility.h"
#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>

using namespace std;
using namespace cv;
using namespace saliency;

class processing
{
    public:
        processing();
        virtual ~processing();
        Scalar processEdges(Mat colorMat, int edgeMethod);
        pair<double, double> processMoments(vector <pair<double,double> > vec);
        double processFocus(Mat colorMat);
        int processHues(Mat src, int testOutput);
        vector<pair<double,double> > processColors(Mat colorMat);
        vector<double> processHaarCascade(Mat colorMat, CascadeClassifier finder, CascadeClassifier aditional_cascade,
bool insideFace, int testOutput, Mat ruleImage);
        vector<float> processFrameGabor(Mat colorMat, int testOutput);
        vector<Mat> splitMat(Mat allMat,double divide,bool bgr2gray, int testOutput);
        int processEdgeHistogram(Mat greyMat);
        int processEHGroup(vector<int> edgeComplete);
        float entropy(Mat seq, Size size, int index);
        Mat myEntropy(Mat seq, int histSize);
       // vector<double> processBgSub(Mat colorMat,  Ptr<BackgroundSubtractor> bg_model);
       // double processStaticSaliency(Mat image);

        struct Mean
        {
            unsigned int n;
            Mean(unsigned int n) : n(n) {}
            double operator()(double sum, double x)
            {
                return sum + x/n;
            }
        };

        struct Std
        {
            unsigned int n;
            double mean;
            Std(unsigned int n, double mean) : n(n), mean(mean) {}
            double operator()(double sum,double x)
            {
                return sum + pow(mean - x,2)/n;
            }
        };

    protected:

    private:

        struct DistCenter
        {
            unsigned int n;
            DistCenter(unsigned int n) : n(n) {}
            double operator()(double sum, pair<double, double >x)
            {
                return sum + x.first/n;
            }
        };

        struct DistSpread
        {
            unsigned int n;
            double mean;
            DistSpread(unsigned int n, double mean) : n(n), mean(mean) {}
            double operator()(double sum, pair<double, double >x)
            {
                return sum + pow(mean - x.first,2)/n;
            }
        };
};

#endif // PROCESSING_H
