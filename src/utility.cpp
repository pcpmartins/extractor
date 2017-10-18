#include "utility.h"
using namespace std;
utility::utility()
{
    //ctor
}

utility::~utility()
{
    //dtor
}
string utility::printHelp()
{

    string h ="Help\n"
         "Usage: execfile <inputFile> <outputFile> [arguments]\n\n"
         "Arguments:\n"

         "  -v, example usage: -v=false\n"
         "      Set video processing off (for image files)\n"
         "  -q, example usage: -q=false\n"
         "      Quiet mode set off.\n"
         "  -t, example usage: -t=0\n"
         "      test histogram output off.\n"
         "      0-none, 1-hue histogram, 2-static saliency, 3- optical flow \n"
         "      4-dynamic saliency, 5-background subtraction.\n"
         "      6-faces. 7-Gabor filter.\n"
         "  -d, example usage: -d=true\n"
         "      process files from a folder.\n"
         "  -s, example usage: -s=5\n"
         "      set sampling factor to 5.\n"
         "  -r, example usage: -r=3\n"
         "      set resize mode to 640x480.\n"
         "      0-no resize,1-320x240, 2-480x360, 3-640x480.\n"
         "  -b, example usage: -b=0\n"
         "      background subtraction off.\n"
         "      0-none, 1-knn, 2-mog2.\n"
         "  -f, example usage: -f=false\n"
         "      optical flow off.\n"
         "  -i, example usage: -i=true\n"
         "      ligar interseção de classificadores Haar.\n"
         "  -h, example usage: -h=true\n"
         "  -g, example usage: -g=true\n"
         "      extrair features Gabor.\n"
         "  -e, example usage: -e=true\n"
         "      extrair edge histograms.\n"
         "      Print help.\n\n"
         "Note: some arguments not yet implemented..\n";


 return h;
}
void utility::drawHist(Mat h_hist,int histSize ){
 // Draw the histograms for Hue
        int hist_w = 512;
        int hist_h = 400;
        int bin_w = cvRound( (double) hist_w/histSize );

        Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

        /// Normalize the result to [ 0, histImage.rows ]
        normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

        /// Draw hue histogram
        for( int i = 1; i < histSize; i++ )
        {
            line( histImage, Point( bin_w*(i-1), hist_h - cvRound(h_hist.at<float>(i-1)) ),
                  Point( bin_w*(i), hist_h - cvRound(h_hist.at<float>(i)) ),
                  Scalar( 255, 0, 0), 2, 8, 0  );

        }
        /// Display
        namedWindow("calcHist", CV_WINDOW_AUTOSIZE );
        imshow("calcHist", histImage );

        waitKey(0);

}
Mat utility::resizeRuleImg(Mat ruleImage,Mat frame){
 //we need to resize the rule of thirds template to exactly the same size of the video
                        Size size(frame.cols,frame.rows);//the dst image size,e.g.100x100
                        Mat dst;//dst image
                        resize(ruleImage,dst,size);//resize image
                        return dst;
}

float utility::angleBetween(const Point &v1, const Point &v2)
{
	float len1 = sqrt(v1.x * v1.x + v1.y * v1.y);
	float len2 = sqrt(v2.x * v2.x + v2.y * v2.y);

	float dot = v1.x * v2.x + v1.y * v2.y;

	float direction = (v1.cross(v2) >= 0 ? 1.0 : -1.0);

	float a = dot / (len1 * len2);

	if (a >= 1.0)
		return 0.0;
	else if (a <= -1.0)
		return 3.14159265359;
	else
		return direction * acos(a); // 0..PI
}
//Compute angle between two vectors

float utility::innerAngle(float px1, float py1, float px2, float py2, float cx1, float cy1)
{

	float dist1 = sqrt((px1 - cx1)*(px1 - cx1) + (py1 - cy1)*(py1 - cy1));
	float dist2 = sqrt((px2 - cx1)*(px2 - cx1) + (py2 - cy1)*(py2 - cy1));

	float Ax, Ay;
	float Bx, By;
	float Cx, Cy;

	//find closest point to C
	//printf("dist = %lf %lf\n", dist1, dist2);

	Cx = cx1;
	Cy = cy1;
	if (dist1 < dist2)
	{
		Bx = px1;
		By = py1;
		Ax = px2;
		Ay = py2;


	}
	else {
		Bx = px2;
		By = py2;
		Ax = px1;
		Ay = py1;
	}


	float Q1 = Cx - Ax;
	float Q2 = Cy - Ay;
	float P1 = Bx - Ax;
	float P2 = By - Ay;


	float A = acos((P1*Q1 + P2*Q2) / (sqrt(P1*P1 + P2*P2) * sqrt(Q1*Q1 + Q2*Q2)));

	A = A * 180 / 3.14159265359;

	return A;
}

