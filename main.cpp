#include <fstream>
#include "processing.h"
#include "utility.h"
#include "main.h"
#include<numeric>
#include <opencv2/core/core.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>

#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include<math.h>
#include<sstream>


#define arg(name) cmd.get<string>(name)
#define argi(name) cmd.get<int>(name)

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace saliency;

void initVectors(int nFiles)
{

    //lets start processing files, can be videos or images
    //we initialize vectors where metadata will be saved with the number of files to process
    focusVec.assign(nFiles,0);
    luminanceVec.assign(nFiles,0);
    redRatioVec.assign(nFiles,0);
    greenRatioVec.assign(nFiles,0);
    blueRatioVec.assign(nFiles,0);
    redMomentsVec.assign(nFiles,pair <double,double>(0.0,0.0));
    greenMomentsVec.assign(nFiles,pair<double,double>(0.0,0.0));
    blueMomentsVec.assign(nFiles,pair<double,double>(0.0,0.0));
    vEdgesVec.assign(nFiles,0);
    hEdgesVec.assign(nFiles,0);
    dEdgesVec.assign(nFiles,0);
    orientationVec.assign(nFiles,0);
    aestheticsVec.assign(nFiles,0);
    facesVec.assign(nFiles,0);
    facesAreaVec.assign(nFiles,0);
    facesRof3Vec.assign(nFiles,0);
    differentHues.assign(nFiles,0);
    fpsVec.assign(nFiles,0);
    widthVec.assign(nFiles,0);
    heightVec.assign(nFiles,0);
    staticSaliencyVec.assign(nFiles,0);
    flowxVec.assign(nFiles,0);
    flowyVec.assign(nFiles,0);
    flowxAvgVec.assign(nFiles,0);
    flowyAvgVec.assign(nFiles,0);
    magFlowVec.assign(nFiles,0);
    shackinessVec.assign(nFiles,0);
    bgSubVec.assign(nFiles,vector<double>(5, 0));
    svmScoreVec.assign(nFiles,0);
    eyesVec.assign(nFiles,0);
    luminanceStdVec.assign(nFiles,0);
    gaborVec.assign(nFiles,vector<float>(12, 0));
    edgeHistogramVec.assign(nFiles,vector<int>(17, 0));
    entropyVec.assign(nFiles,0);

    //edgeHistogramExtracted.assign(5,0);
}

double processStaticSaliency( Ptr<Saliency> staticSaliencyAlgorithm, Mat image)
{

    // double saliencyPercent= 0.0;
    // Mat binaryMap;
    Mat saliencyMap;


    staticSaliencyAlgorithm->computeSaliency( image, saliencyMap );

    // saliency::StaticSaliencySpectralResidual spec;
    //spec.computeBinaryMap( saliencyMap, binaryMap );
    // spec.~Algorithm();

    if(testOutput==2)
    {
        imshow( "Saliency Map", saliencyMap );
        imshow( "Original Image", image );
        //imshow( "Binary Map", binaryMap );


        waitKey( 0 );
    }
    cv::Scalar avgPixelIntensity = cv::mean( saliencyMap  );
    if(isnan(avgPixelIntensity.val[0])) return 0.0;
    else return avgPixelIntensity.val[0];

}

void processColors(Mat colorMat)
{
    Scalar colAvg, colStds, greyAvg, greyStds;

    //Compute moments of color for the image
    meanStdDev(colorMat, colAvg, colStds);
    blue_distribution.push_back(make_pair(colAvg[0], colStds[0]));
    green_distribution.push_back(make_pair(colAvg[1], colStds[1]));
    red_distribution.push_back(make_pair(colAvg[2], colStds[2]));

    //Compute moments of luminance for the image (grey-scale)
    Mat greyMat;
    cvtColor(colorMat, greyMat, CV_BGR2GRAY);
    meanStdDev(greyMat, greyAvg, greyStds);
    luminance_distribution.push_back(make_pair(greyAvg[0], greyStds[0]));
}


double computeRanks(int nv,bool videoProcess)
{
    double normFocus = focusVec.at(nv)/1300; //adhoc treshold

    double wLuminance = luminanceVec.at(nv);
    double normLuminance = wLuminance/255;
    double normAesthetic = aestheticsVec.at(nv); // /2 qoeficient

    double normFaces = facesVec.at(nv);
    double normFaceArea =  facesAreaVec.at(nv);
    double normRuleOfThirds =  facesRof3Vec.at(nv);
    double normFaceQuality = (normFaces+normFaceArea+normRuleOfThirds)/3 ;

    double normDifHues = (double)differentHues.at(nv)/360; // 360 different hues
    double normSize = (widthVec.at(nv)*heightVec.at(nv))/30000000; //taking as maximun 720p
    double fpsSize = fpsVec.at(nv)/50; //taking as maximun 50fps

    double normFpsSize = (normSize+fpsSize)/4;
    double normStaticSaliency = staticSaliencyVec.at(nv);

    double rankSum = 0.0;

    if(videoProcess)
        rankSum = (fpsSize+normSize)/2 ;
    else  rankSum = normSize  ;

    // cout<< "rank "<< rankSum<<endl;
    // cout <<"focus "<< normFocus<<"lum "<< normLuminance <<"faces "<<  normFaces<<"hues "<< normDifHues << "aest "<< normAesthetic<<endl;

    return rankSum;
}
void extract(int nv,int frameCount)
{
    /**
     * Colors and luminance
     */

    luminance_center_std = p.processMoments(luminance_distribution).second;
    luminance_center = p.processMoments(luminance_distribution).first;
    red_moments = p.processMoments(red_distribution);
    green_moments = p.processMoments(green_distribution);
    blue_moments = p.processMoments(blue_distribution);
    double total = red_moments.first + green_moments.first + blue_moments.first;
    redRatio = red_moments.first / total;
    greenRatio = green_moments.first / total;
    blueRatio = blue_moments.first / total;

    luminanceStdVec.at(nv)= luminance_center_std/255; //luminance std normalization
    luminanceVec.at(nv)= luminance_center/255; //luminance normalization
    redRatioVec.at(nv)= redRatio;
    greenRatioVec.at(nv)= greenRatio;
    blueRatioVec.at(nv)= blueRatio;
    redMomentsVec.at(nv)= red_moments;
    greenMomentsVec.at(nv)= green_moments;
    blueMomentsVec.at(nv)= blue_moments;
    /**
     * Edges
     */
    processing::Mean meanv(vert_edges_distribution.size());
    vert_edges_center = std::accumulate(vert_edges_distribution.begin(),
                                        vert_edges_distribution.end(), 0.0, meanv);
    processing::Mean meanh(horz_edges_distribution.size());
    horz_edges_center = std::accumulate(horz_edges_distribution.begin(),
                                        horz_edges_distribution.end(), 0.0, meanh);
    processing::Mean meand(diag_edges_distribution.size());
    diag_edges_center = std::accumulate(diag_edges_distribution.begin(),
                                        diag_edges_distribution.end(), 0.0, meand);

    // cut-off and normalization
/*
    if( vert_edges_center > 18)
        vert_edges_center = 18;

    if( horz_edges_center > 20)
        horz_edges_center = 20;

*/
//vert_edges_center = vert_edges_center;
//horz_edges_center = horz_edges_center;

    vEdgesVec.at(nv) = vert_edges_center;
    hEdgesVec.at(nv) = horz_edges_center;
    dEdgesVec.at(nv) = diag_edges_center;

    //compute orientation, we must finish this!!!
    double bigest = horz_edges_center;
    int result = 2;

    if (diag_edges_center > bigest)  //we need to fix this relation to include diag?
    {
        bigest = diag_edges_center;
        result = 3;
    }
    if (vert_edges_center > bigest)
    {
        bigest = vert_edges_center;
        result = 1;
    }
    orientationVec.at(nv) = result;
    cout <<result<<endl;
    // orientationVec.at(nv) = p.processEHGroup(EH_edges_distribution);


    double hue_norm = (double)totalHues/285;

    if((abs(redRatio - greenRatio)< 0.0006)) // Threshold-> can be greater than zero. eg 0.006
    {
        //   cout<< "it's grayscale media! "<< endl;
        differentHues.at(nv) = 0;
    }
    else differentHues.at(nv) = hue_norm;

    //Aesthetic measure
    //Lets combine the edges value, no normalisation possible
    double edges_norm = log(vert_edges_center + horz_edges_center);
    //The aesthetic measure is the harmonic mean between hues and edges
    double aesthetic = (hue_norm * edges_norm)/(hue_norm + edges_norm);
    if (isnan(aesthetic))
        aesthetic = 0.0;
    if (aesthetic > 0.99)
        aesthetic = 0.99 ;
    aestheticsVec.at(nv) = aesthetic; //normalized aesthetic value

    processing::Mean meanE(frame_entropy_distribution.size());
    entropyVec.at(nv) = std::accumulate(frame_entropy_distribution.begin(),
                                        frame_entropy_distribution.end(), 0.0, meanE);

    cout<<"entropy " <<entropyVec.at(nv) <<endl;

    ///print results
    double divider = frameCount/samplingFactor;
    if(!videoProcess) divider = 1;
    if(!quietMode)
    {
        cout << "\nFocus: " <<  focusVec.at(nv) <<  " lum: " << luminanceVec.at(nv) <<  " lumStd: " << luminanceStdVec.at(nv) <<"\n";
        cout << "Faces: " << (double)totalFaces/(divider)<<" ";
        cout << "Face area: " << (double)totalFaceArea/(divider)<< " Smiles: " << (double)totalEyes/(divider)<<"\n";
        cout << "hues: "<< differentHues.at(nv)<< " " << "simplicity: " << aesthetic
             << " rule of thirds: "<<(totalRof3/(divider))/255;
    }

}

int main(int argc, const char **argv)
{
    cout << "OpenCV image/video feature extraction tool.\n";
    try
    {
        const char *keys =
            "{ v             | true| video mode}"
            "{ q             | true| quiet mode}"
            "{ t             | 0| test mode }"
            "{ d             | false| directory mode}"
            "{ r             | 1| resize mode}"
            "{ s             | 1 | sampling rate}"
            "{ b             | 1| background sub}"
            "{ f             | true| optical flow}"
            "{ g             | false| gabor}"
            "{ e             | false| edge histogram}"
            "{ h             | false | help console }";
        CommandLineParser cmd(argc, argv, keys);

        //cout << "gahhh "<< gabor <<" "<< edgeHist<< endl;

        ///parse the arguments
        if ( argc < 3)  // argc should be at least 2 for correct execution(input and output file path)
        {
            string helpstring = u.printHelp();
            cout << helpstring;
            return 0;
        }

        //gabor = false;
        if(arg("g")=="true")
            gabor = true;

        // edgeHist= false;
        if(arg("e")=="true")
            edgeHist = true;

        // cout << "gahhh "<< gabor <<" "<< edgeHist<< endl;
        // cout << argc<<endl;

        if(arg("h")=="true")
        {
            string h = u.printHelp();
            cout << h;
        }

        quietMode = true;
        if(arg("q")=="false")
            quietMode = false;

        bgSub = false;
        method = 0;
        if((!(argi("b")==0)))
        {
            bgSub = true;
            method=argi("b");
        }

        opticalFlow = true;
        if(arg("f")=="false")
            opticalFlow= false;

        if(arg("v")=="false")
        {
            videoProcess = false;
            opticalFlow= false;
            bgSub = false;
        }

        resizeMode=argi("r");
        samplingFactor=argi("s");

        cout << "sampling factor: "<< argi("s") << " resize mode: " << resizeMode << "\n";

        testOutput = 0;
        testOutput = argi("t");
        cout << "test mode: "<< testOutput<< endl;

        if(arg("d")=="true")
            folderMode = true;

        insideFace = true;

        /// check if source and output file were specified
        inputFile = argv[1];
        outputPath = argv[2];
        outputMlFile = "mloutput.csv";

        static_saliency_algorithm = "SPECTRAL_RESIDUAL" ;
        dynamic_saliency_algorithm = "BinWangApr2014" ;

        //ml
        Ptr<SVM> svm = SVM::create();

        if(svm->load("bin/data/ml_data/classifier.xml")) //load classifier
            cout<<"classifier loaded!"<<endl;

        if (inputFile.empty()||outputPath.empty())
        {
            throw runtime_error("specify video input and output file or path please!");
        }
        else
        {
            //load rule of thirds template

            ruleImage = imread("bin/data/templates/rule.jpg", CV_LOAD_IMAGE_GRAYSCALE);   // Read the iamge from file

            if(! ruleImage.data )                              // Check for invalid input
            {
                cout <<  "Could not open or find the rule image" << endl ;
                return -1;
            }
            cout << "\n>> parsing input: " << inputFile;
            if(folderMode) cout << " folder"<<endl;
            else cout <<" text file"<< endl;
            nFiles = 0; //number of files

            if(folderMode)
            {
                //input is a folder with video or image files
                cv::glob(inputFile.c_str(), fileNames); // loads path to filenames into vector
                nFiles=fileNames.size();

            }
            else
            {
                // read filenames from input txt file
                ifstream myReadFile;
                myReadFile.open(inputFile.c_str());
                char output[100];
                if (myReadFile.is_open())
                {
                    while (!myReadFile.eof())
                    {

                        myReadFile >> output;
                        // if(!quietMode)  cout<< "file: "<< nFiles <<" " << output <<"\n";
                        fileNames.push_back(output); // insert new filename
                        nFiles++;
                    }
                }
                myReadFile.close();
            }
        }

        ///lets start processing files, can be videos or images
        ///we initialize vectors were metadata will be saved with the number of files to process
        initVectors(nFiles);

        //-- 1. Load the cascades for object detection
        if( !face_cascade.load( face_cascade_name ) )
        {
            printf("--(!)Error loading\n");
            return -1;
        };
        if( !aditional_cascade.load( aditional_cascade_name ) )
        {
            printf("--(!)Error loading\n");
            return -1;
        };

        ///main loop
        int nv = 0;
        while(nv < nFiles)
        {
            cout<< "\n> processing: " << fileNames.at(nv);
            totalHues=0;
            totalFaces=0;
            totalEyes=0;
            accumStaticSaliency=0.0;
            edgeMethod = 0;

            //instantiates the specific static Saliency
            Ptr<Saliency> staticSaliencyAlgorithm = Saliency::create( static_saliency_algorithm );
            // Ptr<Saliency> dynamicSaliencyAlgorithm = Saliency::create( dynamic_saliency_algorithm );

            totalFlowX = 0.0;
            totalFlowY = 0.0;
            totalFlowXBorder = 0.0;
            totalFlowYBorder = 0.0;

            //bg subtraction parameters
            smoothMask = true;
            update_bg_model = true;

            percentBg = 0.0;
            percentShadow = 0.0;
            percentForegorund = 0.0;
            percentCameraMove = 0.0;
            percentFocus = 0.0;

            if(bgSub)
            {
                if(method ==1) bg_model = createBackgroundSubtractorKNN().dynamicCast<BackgroundSubtractor>() ;
                else if(method ==2) bg_model = createBackgroundSubtractorMOG2(200, 16.0, true).dynamicCast<BackgroundSubtractor>();
            }

            if(videoProcess) //related argument -v

                ///we are processing videos!

            {
                cap.open(fileNames.at(nv).c_str());
                double fps = cap.get(CV_CAP_PROP_FPS);
                // int format  = int(cap.get(CAP_PROP_FORMAT ));
                //cout << "\nformat: " << format<< " ";
                int width  = int(cap.get(CV_CAP_PROP_FRAME_WIDTH));
                int height = int(cap.get(CV_CAP_PROP_FRAME_HEIGHT));

                cout << fps << " fps "<< width <<"x"<<height<<" ";

                fpsVec.at(nv) = fps;
                widthVec.at(nv) = width;
                heightVec.at(nv) = height;
                float totalFocus = 0.0;
                int frameCount = 0;
                bool once = false;
                shackiness = 0.0;
                int shakes = 0;

                Mat flow, cflow;
                UMat grayFrame, prevgray, uflow; //optic flow
                Point v1, v2; //for flow angle measure

                for(;;) //loop trough the video frames
                {
                    Mat frame;
                    int n= 0;

                    while(n < samplingFactor)   //we will jump as much frames as the sampling rate
                    {
                        cap >> frame;            // get a new frame from video
                        n++;
                        frameCount++;

                        if(frameCount ==1)
                        {
                             cout<<"channels: "<< frame.channels() <<endl;
                        }

                        if (frameCount % 250 == 0) cout << "|"; //progress bar
                            std::flush(std::cout);
                    }
                    if(!frame.data) break;  //if frame doesnt contain data exit loop

                    Mat frameTemp;

                    switch(resizeMode)    //resize acording to -r argument
                    {
                    case 1 :
                        resize(frame, frameTemp,  Size(320,240),0,0,INTER_NEAREST);
                        // imshow("resize", frameTemp);
                        //imshow("resize", frame);
                        frameTemp.copyTo( frame );
                        break;
                    case 2  :
                        resize(frame, frameTemp,  Size(480,360),0,0,INTER_NEAREST);
                        frameTemp.copyTo( frame );
                        break;
                    case 3  :
                        resize(frame, frameTemp,  Size(640,480),0,0,INTER_NEAREST);
                        frameTemp.copyTo( frame );
                        break;
                    }


                    if(once=false)
                    {
                        //we need to resize the rule of thirds template to exactly the same size of the video frame
                        ruleImage = u.resizeRuleImg(ruleImage,frame);
                        once= true;
                        v1= Point2f(0.0,0.0);


                    }

                    if(bgSub)
                    {
                        if( fgimg.empty() )
                            fgimg.create(frame.size(), frame.type());
                        if(backimg.empty() )
                            backimg.create(frame.size(), frame.type());

                        //update the model
                        bg_model->apply(frame, fgmask, update_bg_model ? -1 : 0);

                        int w = frame.cols;
                        int h = frame.rows;
                        double total = w*h;
                        float frameFG = 0.0;
                        float frameShadow = 0.0;
                        float frameBg = 0.0;
                        float frameFocus = 0.0;
                        float bgFocus = 0.0;

                        for (int y = 0; y < h; y++)
                        {

                            for (int x = 0; x < w; x++)
                            {

                                const int pixelValue = fgmask.at<uchar>(y,x);

                                if(pixelValue==255)
                                    frameFG+=1;

                                if(pixelValue==127)
                                    frameShadow+=1;
                            }

                        }

                        if( smoothMask )
                        {
                            GaussianBlur(fgmask, fgmask, Size(11, 11), 3.5, 3.5);
                            threshold(fgmask, fgmask, 10, 255, THRESH_BINARY);
                        }

                        fgimg = Scalar::all(0);
                        frame.copyTo(fgimg, fgmask);
                        backimg = Scalar::all(0);
                        frame.copyTo(backimg, bgmask);

                        Mat bgimg;
                        bg_model->getBackgroundImage(bgimg);

                        frameFG = frameFG/total;
                        frameShadow = frameShadow/total;
                        frameBg = 1-frameFG-frameShadow;
                        percentForegorund+=frameFG;
                        percentShadow+=frameShadow;
                        percentBg+=frameBg;

                        if(frameFG >= 0.80) percentCameraMove +=1;

                        frameFocus =  p.processFocus(fgimg);
                        bgFocus = p.processFocus(backimg);
                        float focusDiff = frameFocus/bgFocus;

                        if(isinf(focusDiff))focusDiff =30;
                        if(isnan(focusDiff))focusDiff=0;
                        if (focusDiff > 30) focusDiff = 30;
                        percentFocus += focusDiff;

                        if(testOutput==5)
                        {

                            imshow("image", frame);
                            imshow("foreground mask", fgmask);
                            imshow("foreground image", fgimg);
                            if(!bgimg.empty())imshow("mean background image", bgimg );
                            char k = (char)waitKey(0);
                            if( k == 27 ) break;
                        }
                    }

                    if(opticalFlow)
                    {

                        cvtColor( frame, grayFrame, COLOR_BGR2GRAY );

                        Point2f frameFlow = Point2f(0.0,0.0);
                        Point2f frameFlowBorder = Point2f(0.0,0.0);
                        Point2f signedFrameFlow = Point2f(0.0,0.0);
                        int w = grayFrame.cols;
                        int h = grayFrame.rows;

                        if( !prevgray.empty() )
                        {

                            cv::calcOpticalFlowFarneback(prevgray, grayFrame, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
                            cvtColor(prevgray, cflow, COLOR_GRAY2BGR);
                            uflow.copyTo(flow);

                            for (int y = 0; y < h; y += 5)
                            {
                                for (int x = 0; x < w; x += 5)
                                {

                                    // get the flow from y, x position * 10 for better visibility
                                    const Point2f flowatxy = flow.at<Point2f>(y, x) ;//*10

                                    frameFlow.x+=abs(flowatxy.x);
                                    frameFlow.y+=abs(flowatxy.y);
                                    signedFrameFlow.x+=flowatxy.x;
                                    signedFrameFlow.y+=flowatxy.y;

                                    if(x>=20 && x <= w-20 && y >= 20)
                                    {
                                        line(cflow, Point(x, y), Point(cvRound(x + flowatxy.x), cvRound(y + flowatxy.y)), Scalar(255,0,0));
                                    }

                                    if(x < 20 || x > w-20 || y < 20)
                                    {
                                        frameFlowBorder.x += flowatxy.x;
                                        frameFlowBorder.y += flowatxy.y;
                                        circle(cflow, Point(x, y), 1, Scalar(0, 0, 0), -1); // draw initial point
                                    }
                                }
                            }

                            if(testOutput==3)
                            {
                                line(cflow, Point(w/2, h/2), Point(cvRound(w/2 +  frameFlowBorder.x/10),
                                                                   cvRound(h/2 +  frameFlowBorder.y/10)), Scalar(0,255,0), 4);
                                line(cflow, Point(w/2, h/2), Point(cvRound(w/2 + signedFrameFlow.x/100),
                                                                   cvRound(h/2 + signedFrameFlow.y/100)), Scalar(0,0,255), 2);
                                imshow("flow", cflow);
                                waitKey(0);
                            }

                        }
                        totalFlowX += frameFlow.x;
                        totalFlowY += frameFlow.y;
                        v2.x = frameFlow.x;
                        v2.y = frameFlow.y;
                        //v2.x = frameFlowBorder.x;
                        //v2.y = frameFlowBorder.y;

                        float angle = u.innerAngle(v1.x, v1.y, v2.x, v2.y, w/2, h/2); //compute angle between two vectors
                        float frameMagnitude = (sqrt(pow(totalFlowX,2)+pow( totalFlowY,2)))/70000;
                        // cout<< "angle: "<<angle<<" mag: "<<frameMagnitude<<endl;

                        if(angle >= 45.0 && frameMagnitude > 10.0) //check shake based on vectors angle with threshold of 10
                        {
                            shakes ++;
                            // cout<< "shakes: "<<shakes<<endl;
                        }

                        std::swap(prevgray, grayFrame);
                        v1.x = v2.x ;
                        v1.y = v2.y;

                    }

                    accumStaticSaliency += processStaticSaliency( staticSaliencyAlgorithm, frame);

                    double focusMeasure = p.processFocus(frame);
                    totalFocus+= focusMeasure;

                    double frameHues = p.processHues(frame,testOutput);
                    totalHues += frameHues/(frameCount/samplingFactor);

                    processColors(frame);

                    Scalar allEdge=p.processEdges(frame, edgeMethod);
                    vert_edges_distribution.push_back(allEdge[0]); //Add first channel (Grey-Scale)
                    horz_edges_distribution.push_back(allEdge[1]);
                    diag_edges_distribution.push_back(allEdge[2]);

                    //entropy
                    Mat src, hist;
                    cvtColor( frame, src, CV_BGR2GRAY );
                    // Establish the number of bins
                    int histSize = 256;
                    hist = p.myEntropy(src, histSize);
                    float entropy = p.entropy(hist,src.size(), histSize);
                    frame_entropy_distribution.push_back(entropy);

                    // cout<<hist<<endl;
                   // cout<<"entropy: "<<entropy<<endl;



                    if (frameCount % 30 == 0)  //30 frames interval
                    {
                        //cout<< "ola!!"<<endl;

                        vector<Mat> cutImage = p.splitMat(frame, 0.25, true, testOutput); //0.25(=1/4) <=> split in 16 squares(4x4)

                        //for each piece compute edge histogram coefficients
                        vector<int> edgeComplete;
                        edgeComplete.assign(17,0);
                        for(int i=0; i< cutImage.size(); i++)
                        {
                            int edgeHistogramExtracted = p.processEdgeHistogram(cutImage[i]);
                            edgeComplete[i]= edgeHistogramExtracted;

                            if(i==0)EH0_distribution.push_back(edgeHistogramExtracted);
                            if(i==1)EH1_distribution.push_back(edgeHistogramExtracted);
                            if(i==2)EH2_distribution.push_back(edgeHistogramExtracted);
                            if(i==3)EH3_distribution.push_back(edgeHistogramExtracted);
                            if(i==4)EH4_distribution.push_back(edgeHistogramExtracted);
                            if(i==5)EH5_distribution.push_back(edgeHistogramExtracted);
                            if(i==6)EH6_distribution.push_back(edgeHistogramExtracted);
                            if(i==7)EH7_distribution.push_back(edgeHistogramExtracted);
                            if(i==8)EH8_distribution.push_back(edgeHistogramExtracted);
                            if(i==9)EH9_distribution.push_back(edgeHistogramExtracted);
                            if(i==10)EH10_distribution.push_back(edgeHistogramExtracted);
                            if(i==11)EH11_distribution.push_back(edgeHistogramExtracted);
                            if(i==12)EH12_distribution.push_back(edgeHistogramExtracted);
                            if(i==13)EH13_distribution.push_back(edgeHistogramExtracted);
                            if(i==14)EH14_distribution.push_back(edgeHistogramExtracted);
                            if(i==15)EH15_distribution.push_back(edgeHistogramExtracted);

                        }
                        EH_edges_distribution.push_back(p.processEHGroup(edgeComplete));

                        // cout << "edges "<< edgeHistogramVec[nv][17] << endl;
                    }

                    vector<double> faceData = p.processHaarCascade(frame, face_cascade,aditional_cascade, insideFace,
                                              testOutput, ruleImage);

                    if(faceData[0]>=1)
                        totalFaces +=1;
                    else  totalFaces +=faceData[0];

                    totalFaceArea += faceData[1];
                    totalRof3 += faceData[2];
                    totalEyes += faceData[3];

                }

                edgeHistogramVec[nv][0] = p.processEHGroup(EH0_distribution);
                edgeHistogramVec[nv][1] = p.processEHGroup(EH1_distribution);
                edgeHistogramVec[nv][2] = p.processEHGroup(EH2_distribution);
                edgeHistogramVec[nv][3] = p.processEHGroup(EH3_distribution);
                edgeHistogramVec[nv][4] = p.processEHGroup(EH4_distribution);
                edgeHistogramVec[nv][5] = p.processEHGroup(EH5_distribution);
                edgeHistogramVec[nv][6] = p.processEHGroup(EH6_distribution);
                edgeHistogramVec[nv][7] = p.processEHGroup(EH7_distribution);
                edgeHistogramVec[nv][8] = p.processEHGroup(EH8_distribution);
                edgeHistogramVec[nv][9] = p.processEHGroup(EH9_distribution);
                edgeHistogramVec[nv][10] = p.processEHGroup(EH10_distribution);
                edgeHistogramVec[nv][11] = p.processEHGroup(EH11_distribution);
                edgeHistogramVec[nv][12] = p.processEHGroup(EH12_distribution);
                edgeHistogramVec[nv][13] = p.processEHGroup(EH13_distribution);
                edgeHistogramVec[nv][14] = p.processEHGroup(EH14_distribution);
                edgeHistogramVec[nv][15] = p.processEHGroup(EH15_distribution);
                edgeHistogramVec[nv][16] = p.processEHGroup( EH_edges_distribution);


                double medianFocus = totalFocus/(frameCount/samplingFactor);

                if(medianFocus > 6000)
                    medianFocus = 6000;

                focusVec.at(nv)= medianFocus/6000; //cut-off normalization at 6000, very few records over that value

                double facePercent = (double)totalFaces/(frameCount/samplingFactor);
                double eyePercent = (double)totalEyes/(frameCount/samplingFactor);
                double areaPercent = (float)totalFaceArea/(frameCount/samplingFactor);
                double rof3computed = (double)totalRof3/(frameCount/samplingFactor);
                double staticSaliencyComputed = (double)accumStaticSaliency/(frameCount/samplingFactor);

                //eliminate false positives
                if (facePercent < 0.01)
                {
                    facePercent = 0.0;
                    areaPercent = 0.0;
                    rof3computed = 0.0;
                }
                if (facePercent >= 1.0) facePercent = 1.0;


                facesVec.at(nv)= facePercent;
                eyesVec.at(nv)= eyePercent;
                facesAreaVec.at(nv)= areaPercent;
                facesRof3Vec.at(nv) = rof3computed/255; //normalized
                differentHues.at(nv)=(int)totalHues;
                staticSaliencyVec.at(nv)= staticSaliencyComputed;

                extract(nv,frameCount);

                flowxVec.at(nv)=totalFlowX;
                flowyVec.at(nv)=totalFlowY;
                flowxAvgVec.at(nv)=(totalFlowX/(frameCount/samplingFactor));
                flowyAvgVec.at(nv)=(totalFlowY/(frameCount/samplingFactor));
                magFlowVec.at(nv) = (sqrt(pow(flowxAvgVec.at(nv),2)+pow(flowyAvgVec.at(nv),2)))/70000;
                shackinessVec.at(nv) = (double)shakes/frameCount;

                vector<double> frameBgData = vector<double>(5,0);
                frameBgData.at(0) = percentForegorund/(frameCount/samplingFactor);
                frameBgData.at(1) = percentShadow/(frameCount/samplingFactor);
                frameBgData.at(2) = percentBg/(frameCount/samplingFactor);
                frameBgData.at(3) = percentCameraMove/(frameCount/samplingFactor);
                frameBgData.at(4) = percentFocus/(frameCount/samplingFactor)/24;
                bgSubVec.at(nv) = frameBgData;

                if(!quietMode)
                {
                    cout << "\norientation: "<<edgeHistogramVec[nv][16];
                    cout << " "
                         << edgeHistogramVec[nv][0]
                         << edgeHistogramVec[nv][1]
                         << edgeHistogramVec[nv][2]
                         << edgeHistogramVec[nv][3]
                         << edgeHistogramVec[nv][4]
                         << edgeHistogramVec[nv][5]
                         << edgeHistogramVec[nv][6]
                         << edgeHistogramVec[nv][7]
                         << edgeHistogramVec[nv][8]
                         << edgeHistogramVec[nv][9]
                         << edgeHistogramVec[nv][10]
                         << edgeHistogramVec[nv][11]
                         << edgeHistogramVec[nv][12]
                         << edgeHistogramVec[nv][13]
                         << edgeHistogramVec[nv][14]
                         << edgeHistogramVec[nv][15]<< endl;

                    cout <<"entropy: "<<entropyVec.at(nv)<<endl;

                    cout <<"static saliency: "<<staticSaliencyComputed<<endl;
                    cout<<"flow: "<<flowxVec.at(nv)<<" "<< flowyVec.at(nv)<<" "<<flowxAvgVec.at(nv)
                        <<" "<<flowyAvgVec.at(nv)<<" "<< magFlowVec.at(nv)<<endl;
                    cout << "shaking: "<<shakes << " "<< shackinessVec.at(nv)<<endl;
                    cout << "bgsub: "<<  frameBgData.at(0)<<" "<<frameBgData.at(1)<<" "<< frameBgData.at(2)
                         <<" "<<frameBgData.at(3)<<" "<<frameBgData.at(4)<<endl;
                }


                luminance_distribution.clear();
                blue_distribution.clear();
                green_distribution.clear();
                red_distribution.clear();
                totalHues=0;
                totalFaces=0;
                totalFaceArea=0;
                totalRof3=0;
                accumStaticSaliency=0.0;
                EH_edges_distribution.clear();
                EH0_distribution.clear();
                EH1_distribution.clear();
                EH2_distribution.clear();
                EH3_distribution.clear();
                EH4_distribution.clear();
                EH5_distribution.clear();
                EH6_distribution.clear();
                EH7_distribution.clear();
                EH8_distribution.clear();
                EH9_distribution.clear();
                EH10_distribution.clear();
                EH11_distribution.clear();
                EH12_distribution.clear();
                EH13_distribution.clear();
                EH14_distribution.clear();
                EH15_distribution.clear();
                frame_entropy_distribution.clear();

                nv++;
                cap.release();
            }
            else
            {
                //we are processing images!

                Mat image;
                image = imread(fileNames.at(nv).c_str(), CV_LOAD_IMAGE_COLOR);   // Read the iamge from file

                if(! image.data )                              // Check for invalid input
                {
                    cout <<  "Could not open or find the image" << endl ;
                    return -1;
                }

                // int profundidade = image.depth();
                //cout << "bit depth: "<< profundidade<<endl;
                int width=image.cols;
                int height = image.rows;
                fpsVec.at(nv) = 0;
                widthVec.at(nv) = width;
                heightVec.at(nv) = height;

                ruleImage = u.resizeRuleImg(ruleImage,image);

                processColors(image);

                Scalar allEdge = p.processEdges(image,edgeMethod);
                vert_edges_distribution.push_back(allEdge[0]); //Add first channel (Grey-Scale)
                horz_edges_distribution.push_back(allEdge[1]);
                diag_edges_distribution.push_back(allEdge[2]);

                totalHues = p.processHues(image,testOutput);

                vector<double> faceData = p.processHaarCascade(image, face_cascade,aditional_cascade, insideFace,
                                          testOutput, ruleImage);

                totalFaces = faceData[0];
                totalFaceArea = faceData[1];
                totalRof3 = faceData[2];
                totalEyes = faceData[3];

                //compute focus measure and save
                double focusMeasure = p.processFocus(image);
                focusVec.at(nv)= focusMeasure/1000;
                facesVec.at(nv)= totalFaces;
                eyesVec.at(nv)= totalEyes;
                facesAreaVec.at(nv) = (double)totalFaceArea;
                facesRof3Vec.at(nv) = totalRof3/255;  //normalised
                differentHues.at(nv)=totalHues;

                //processStaticSaliency( staticSaliencyAlgorithm, image);
                accumStaticSaliency = processStaticSaliency( staticSaliencyAlgorithm, image);
                staticSaliencyVec.at(nv)= accumStaticSaliency;

                //cout <<"static saliency: "<<accumStaticSaliency;

                gaborVec.at(nv) = p.processFrameGabor(image, testOutput);
                //cout <<"gabor: "<<gaborVec.at(nv)[0]<<endl;

                //lets cut our input image converted to grayscale into a set of rectangular sub-parts
                vector<Mat> cutImage = p.splitMat(image, 0.25, true, testOutput); //0.25(=1/4) <=> split in 16 squares(4x4)

                //for each piece compute edge histogram coefficients
                vector<int> edgeComplete;
                edgeComplete.assign(17,0);

                for(int i=0; i< cutImage.size(); i++)
                {
                    int edgeHistogramExtracted = p.processEdgeHistogram(cutImage[i]);
                    edgeComplete[i]= edgeHistogramExtracted;
                    // cout << edgeHistogramExtracted << endl;
                }

                edgeHistogramVec[nv] = edgeComplete;
                edgeHistogramVec[nv][16] = p.processEHGroup(edgeComplete);

                //cout << "edges "<< edgeHistogramVec[nv][16] << endl;

                //entropy
                Mat src, hist;
                cvtColor( image, src, CV_BGR2GRAY );
                // Establish the number of bins
                int histSize = 256;
                hist = p.myEntropy(src, histSize);
                float entropy = p.entropy(hist,src.size(), histSize);
                frame_entropy_distribution.push_back(entropy);
                // cout<<hist<<endl;
                cout<<"entropy: "<<entropy<<endl;



                extract(nv,1);

                luminance_distribution.clear();
                blue_distribution.clear();
                green_distribution.clear();
                red_distribution.clear();
                frame_entropy_distribution.clear();
                totalHues=0;
                totalFaces=0;
                totalFaceArea=0;
                totalRof3=0;
                accumStaticSaliency=0.0;
                nv++;
            }

        }
        //let's write the metadata to the output csv file
        ofstream myfile (outputPath.c_str());
        ofstream mymlfile (outputMlFile.c_str());

        if (myfile.is_open() && mymlfile.is_open())
        {
            int nv = 0;

            //uncomment to print header
            myfile   << "file_path" << ","
                     << "width" << ","
                     << "height" << ","
                     << "red_ratio" << ","
                     << "red_moment_1"  << ","
                     << "red_moment_2"  << ","
                     << "green_ratio" << ","
                     << "green_moment_1" << ","
                     << "green_moment_2"  << ","
                     << "blue_ratio"  << ","
                     << "blue_moment_1"  << ","
                     << "blue_moment_2" << ","
                     << "focus" << ","
                     << "luminance" << ","
                     << "luminance_std" << ","
                     << "all_edges" << ","
                     << "v_edges"  << ","
                     << "h_edges"  << ","
                     << "d_edges"  << ","
                     << "orientation"  << ","
                     << "simplicity"  << ","
                     << "dif_hues"  << ","
                     << "faces"  << ","
                     << "faces_area"  << ","
                     << "smiles"  << ","
                     << "rule_of_thirds"<< ","
                     << "static_saliency"  << ","
                     << "rank_sum"  << ","
                     << "fps" << ","
                     << "size" << ","
                     << "shackiness" << ","
                     << "motion_mag" << ","
                     << "fg_area" << ","
                     << "shadow_area" << ","
                     << "bg_area" << ","
                     << "camera_move" << ","
                     << "focus_diff" <<  ","
                     << "gabor_0" <<  ","
                     << "gabor_15" <<   ","
                     << "gabor_30" <<  ","
                     << "gabor_45" <<   ","
                     << "gabor_60" <<  ","
                     << "gabor_75" <<   ","
                     << "gabor_90" <<  ","
                     << "gabor_105" <<   ","
                     << "gabor_120" <<  ","
                     << "gabor_135" <<   ","
                     << "gabor_150" <<   ","
                     << "gabor_165" <<  ","

                     << "EH_0" << ","
                     << "EH_1" << ","
                     << "EH_2" << ","
                     << "EH_3" << ","
                     << "EH_4" << ","
                     << "EH_5" << ","
                     << "EH_6" << ","
                     << "EH_7" << ","
                     << "EH_8" << ","
                     << "EH_9" << ","
                     << "EH_10" << ","
                     << "EH_11" << ","
                     << "EH_12" << ","
                     << "EH_13" << ","
                     << "EH_14" << ","
                     << "EH_15" << ","
                     << "EH_16" << ","
                     << "entropy" << "\n";

            while(nv < nFiles)
            {
                double rankSum = computeRanks(nv,videoProcess);
                vector<double> bgsubdata = bgSubVec.at(nv);
                float frameDim = (float)(widthVec.at(nv)*heightVec.at(nv))/2400000;
                float allEdges = (vEdgesVec.at(nv)+hEdgesVec.at(nv))/2;

                string tempName = fileNames.at(nv);
                string destName = tempName.substr(tempName.find_last_of("/")+1, tempName.size());
                string finalName = destName.substr(0, destName.find_last_of("."));
                // cout << finalName << endl;



                float normFps = fpsVec.at(nv)/60;


                myfile << finalName
                       << ","<< widthVec.at(nv)
                       << ","<< heightVec.at(nv)
                       <<","  << redRatioVec.at(nv)
                       << "," << redMomentsVec.at(nv).first
                       << "," << redMomentsVec.at(nv).second
                       << "," << greenRatioVec.at(nv)
                       << "," << greenMomentsVec.at(nv).first
                       << "," << greenMomentsVec.at(nv).second
                       << "," << blueRatioVec.at(nv)
                       << "," << blueMomentsVec.at(nv).first
                       << "," << blueMomentsVec.at(nv).second
                       << "," << focusVec.at(nv)
                       << "," << luminanceVec.at(nv)
                       << "," << luminanceStdVec.at(nv)
                       << "," << allEdges
                       << "," << vEdgesVec.at(nv)
                       << "," << hEdgesVec.at(nv)
                       << "," << dEdgesVec.at(nv)
                       << "," << orientationVec.at(nv)
                       << "," << aestheticsVec.at(nv)
                       << "," << differentHues.at(nv)
                       << "," << facesVec.at(nv)
                       << "," << facesAreaVec.at(nv)
                       << "," << eyesVec.at(nv)
                       << "," << facesRof3Vec.at(nv)
                       << "," << staticSaliencyVec.at(nv)
                       << "," << rankSum
                       << "," << normFps
                       << "," << frameDim
                       << "," << shackinessVec.at(nv)
                       << "," << magFlowVec.at(nv)
                       << "," << bgsubdata.at(0)
                       << "," << bgsubdata.at(1)
                       << "," << bgsubdata.at(2)
                       << "," << bgsubdata.at(3)
                       << "," << bgsubdata.at(4)

                       << "," << gaborVec.at(nv)[0]
                       << "," << gaborVec.at(nv)[1]
                       << "," << gaborVec.at(nv)[2]
                       << "," << gaborVec.at(nv)[3]
                       << "," << gaborVec.at(nv)[4]
                       << "," << gaborVec.at(nv)[5]
                       << "," << gaborVec.at(nv)[6]
                       << "," << gaborVec.at(nv)[7]
                       << "," << gaborVec.at(nv)[8]
                       << "," << gaborVec.at(nv)[9]
                       << "," << gaborVec.at(nv)[10]
                       << "," << gaborVec.at(nv)[11]

                       << "," << edgeHistogramVec.at(nv)[0]
                       << "," << edgeHistogramVec.at(nv)[1]
                       << "," << edgeHistogramVec.at(nv)[2]
                       << "," << edgeHistogramVec.at(nv)[3]
                       << "," << edgeHistogramVec.at(nv)[4]
                       << "," << edgeHistogramVec.at(nv)[5]
                       << "," << edgeHistogramVec.at(nv)[6]
                       << "," << edgeHistogramVec.at(nv)[7]
                       << "," << edgeHistogramVec.at(nv)[8]
                       << "," << edgeHistogramVec.at(nv)[9]
                       << "," << edgeHistogramVec.at(nv)[10]
                       << "," << edgeHistogramVec.at(nv)[11]
                       << "," << edgeHistogramVec.at(nv)[12]
                       << "," << edgeHistogramVec.at(nv)[13]
                       << "," << edgeHistogramVec.at(nv)[14]
                       << "," << edgeHistogramVec.at(nv)[15]
                       << "," << edgeHistogramVec.at(nv)[16]


                       << "," << (float)(entropyVec.at(nv))/12
                       <<"\n";

                /*  //feature vector ready for classification process

                    mymlfile   << redRatioVec.at(nv)
                               << "," << greenRatioVec.at(nv)
                               << "," << blueRatioVec.at(nv)
                               << "," << focusVec.at(nv)
                               << "," << luminanceVec.at(nv)
                               << "," << aestheticsVec.at(nv)
                               << "," << differentHues.at(nv)
                               << "," << facesVec.at(nv)
                               << "," << facesAreaVec.at(nv)
                               << "," << eyesVec.at(nv)
                               << "," << facesRof3Vec.at(nv)
                               << "," << staticSaliencyVec.at(nv)
                               << "," << shackinessVec.at(nv)
                               << "," << magFlowVec.at(nv)
                               << "," << bgsubdata.at(0)
                               << "," << bgsubdata.at(2)
                               << "," << bgsubdata.at(3)
                               << "," << bgsubdata.at(4) <<"\n";*/





                mymlfile   << "," << eyesVec.at(nv)

                           << "," << (float)(edgeHistogramVec.at(nv)[0])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[1])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[2])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[3])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[4])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[5])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[6])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[7])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[8])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[9])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[10])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[11])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[12])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[13])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[14])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[15])/5
                           << "," << (float)(edgeHistogramVec.at(nv)[16])/5

                           << "," << luminanceStdVec.at(nv) <<"\n";




                nv++;
            }
            cout << "\n>> saved output file: "<<outputPath <<"\n";
            cout << ">> saved ml file: "<<outputMlFile <<"\n";

            myfile.close();
            mymlfile.close();
        }
        else cout << "Unable to open output file";

    }
    catch (const exception &e)
    {
        cout << "error: " << e.what() << endl;
        return -1;
    }

    // cout << "\nexiting!\n";
    return 0;

}

