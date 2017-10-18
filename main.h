#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/saliency.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/video/background_segm.hpp"
#include <opencv2/saliency.hpp>

int samplingFactor;             //the rate of processing
int resizeMode;                 //resize video input
bool quietMode;          //do we want huge amount of feedback?
bool videoProcess = true;       //are we processing videos or images?
int testOutput;                 //test vizualization?
bool folderMode = false;        //from folder or text file
bool gaborMode;
string outputPath;              //file to save metadata output from processing
string inputFile;               //file with list of items to process, can be video or image
string inputFolder;             //folder with files
string outputMlFile;            //feature vector for svm classification
int nFiles;                     //number of files to process
VideoCapture cap;               //current video for processing
processing p;                   //processing class object
utility u;                      //uility class object
vector<double> rankSumVec;      //weigthed average of features
vector<int> widthVec;
vector<int> heightVec;
vector<double> fpsVec;
// vector<double> edgeHistogramExtracted;

//we save intermediate processing results in vectors, the method extract() will compute final values
//based on this vectors data

vector<String> fileNames;       //lets save all filenames
vector<double> focusVec;        //to save each item focus value
vector<double> luminanceVec;    //to save each item luminance value
vector<double> luminanceStdVec;    //to save each item luminance std value

//Color Moments (first and second)
pair<double, double> red_moments;
pair<double, double> green_moments;
pair<double, double> blue_moments;

//Color distribution (average,standard deviation)
vector<pair<double,double> > redMomentsVec;
vector<pair<double,double> > greenMomentsVec;
vector<pair<double,double> > blueMomentsVec;

//color ratios
double redRatio;
double greenRatio;
double blueRatio;
vector <double> redRatioVec;
vector<double> greenRatioVec;
vector<double> blueRatioVec;

//luminance value
double luminance_center;
double luminance_center_std;

//Color distributions (average,standard deviation) if (image)size = 1 else size > 1
vector<pair<double,double> > red_distribution;
vector<pair<double,double> > green_distribution;
vector<pair<double,double> > blue_distribution;
vector<pair<double,double> > luminance_distribution;

//Edge summarizations
double vert_edges_center;
double horz_edges_center;
double diag_edges_center;

//Edge distributions
vector<double> vert_edges_distribution;
vector<double> horz_edges_distribution;
vector<double> diag_edges_distribution;
vector<double> diag_135_edges_distribution;
vector<double> vEdgesVec;
vector<double> hEdgesVec;
vector<double> dEdgesVec;
//orientation
vector<int> orientationVec;
int edgeMethod;
//Simplicity as Aesthetic measure (number of diferent hues and edges)
double aesthetic;
//to save different hues value
vector<double> differentHues;
double totalHues;
vector<double> aestheticsVec;

///face recognition
//our face classifier is not invariant to rotation
//using smile detection as aditional cascade classifier
String face_cascade_name = "bin/data/haar/haarcascade_frontalface_alt.xml";
String aditional_cascade_name = "bin/data/haar/smiled_05.xml";
CascadeClassifier face_cascade;
CascadeClassifier aditional_cascade;
bool insideFace;
int totalFaces;
int totalEyes;
float totalFaceArea;
double totalRof3;
vector<double> facesVec;
vector<double> eyesVec;
vector<float> facesAreaVec;
vector<double> facesRof3Vec;
Mat ruleImage;

//saliency
//static and dynamic
String static_saliency_algorithm;
String dynamic_saliency_algorithm;
String training_path ;
float accumStaticSaliency;
vector<double> staticSaliencyVec;

//opticflow
bool opticalFlow;
float totalFlowX;
float totalFlowY;
float totalFlowXBorder;
float totalFlowYBorder;
vector<double> flowxVec, flowyVec, flowxAvgVec, flowyAvgVec, magFlowVec, shackinessVec; //total and average flow;
double shackiness;

//background subtraction
bool bgSub;
Mat  fgmask, fgimg, backimg, bgmask;
bool smoothMask ;
bool update_bg_model;
int method ;
Ptr<BackgroundSubtractor> bg_model;
vector<vector<double> > bgSubVec;
float percentBg, percentShadow, percentForegorund,percentFocus, percentCameraMove;

//ml
vector<int> svmScoreVec;

//gabor
vector<vector<float> > gaborVec;
bool gabor;

vector<vector<int> > edgeHistogramVec;
vector<int> EH_edges_distribution;
vector<int> EH0_distribution;
vector<int> EH1_distribution;
vector<int> EH2_distribution;
vector<int> EH3_distribution;
vector<int> EH4_distribution;
vector<int> EH5_distribution;
vector<int> EH6_distribution;
vector<int> EH7_distribution;
vector<int> EH8_distribution;
vector<int> EH9_distribution;
vector<int> EH10_distribution;
vector<int> EH11_distribution;
vector<int> EH12_distribution;
vector<int> EH13_distribution;
vector<int> EH14_distribution;
vector<int> EH15_distribution;

vector<double> entropyVec;
vector<double> frame_entropy_distribution;
//vector<vector<int> > EHTempVideoVec;

bool edgeHist;

#endif // MAIN_H_INCLUDED
