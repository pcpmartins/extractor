#include "processing.h"

using namespace std;
using namespace cv;
using namespace saliency;

processing::processing()
{
    //ctor
}

processing::~processing()
{
    //dtor
}

//Compute Edge Distributions
Scalar processing::processEdges(Mat colorMat, int edgeMethod)
{
    Mat greyMat;
    Scalar allEdges;

    GaussianBlur(colorMat, colorMat, Size(3, 3), 0, 0, BORDER_DEFAULT);
    cvtColor(colorMat, greyMat, CV_BGR2GRAY);

    if(edgeMethod==0)
    {

        //Prewitt operator
        Mat horizontal = (cv::Mat1f(3,3)<<-1,-1,-1,0,0,0,1,1,1);
        Mat vertical = (cv::Mat1f(3,3)<<-1,0,1,-1,0,1,-1,0,1);
        //Mat diagonal = (cv::Mat1f(3,3)<<0,1,0,-1,0,-1,0,-1,1);
        Mat diagonal45 = (cv::Mat1f(3,3)<<1,1,1,-1,-2,1,-1,-1,1);
        Mat diagonal135 = (cv::Mat1f(3,3)<<-1,-1,1,-1,-2,1,1,1,1);

        Mat vEdges, hEdges, d45Edges, d135Edges;
        filter2D(greyMat, hEdges, CV_8U, horizontal);
        filter2D(greyMat, vEdges, CV_8U, vertical);
        filter2D(greyMat, d45Edges, CV_8U, diagonal45);
        filter2D(greyMat, d135Edges, CV_8U, diagonal135);

        Scalar mVEdge = mean(vEdges);
        Scalar mHEdge = mean(hEdges);
        Scalar mD45Edge = mean(d45Edges);
         Scalar mD135Edge = mean(d135Edges);

        allEdges[0]= mVEdge[0];
        allEdges[1]= mHEdge[0];

        if(mD45Edge[0]>mD135Edge[0])
           allEdges[2]= mD45Edge[0];
        else  allEdges[2]= mD135Edge[0];

        // imshow( "edges", vEdges) ;

        // waitKey(0);
    }

    else if(edgeMethod==1)
    {


        Mat hEdges,hEdges2, vEdges, vEdges2, dEdges45, dEdges135, dEdges225, dEdges315, thEdges, tvEdges, dEdges1, dEdges2, tdEdges;

        //Sobel operator
        Mat horizontal = (cv::Mat1f(3,3)<<1,2,1,0,0,0,-1,-2,-1);
        Mat horizontal2 = (cv::Mat1f(3,3)<<-1,-2,-1,0,0,0,1,2,1);
        Mat vertical = (cv::Mat1f(3,3)<<1,0,-1,2,0,-2,1,0,-1);
        Mat vertical2 = (cv::Mat1f(3,3)<<-1,0,1,-2,0,2,-1,0,1);

        Mat Diag_135 = (cv::Mat1f(3,3)<<2,1,0,1,0,-1,0,-2,-1);
        Mat Diag_315 = (cv::Mat1f(3,3)<<-2,-1,0,-1,0,1,0,1,2);

        Mat Diag_225 = (cv::Mat1f(3,3)<<0,-1,-2,1,0,-1,2,1,0);
        Mat Diag_45 = (cv::Mat1f(3,3)<<0,1,2,-1,0,1,-2,-1,0);

        filter2D(greyMat, hEdges, CV_8U, horizontal);
        filter2D(greyMat, hEdges2, CV_8U, horizontal2);
        filter2D(greyMat, vEdges, CV_8U, vertical);
        filter2D(greyMat, vEdges2, CV_8U, vertical2);

        filter2D(greyMat, dEdges45, CV_8U, Diag_45);
        filter2D(greyMat, dEdges225, CV_8U, Diag_225);
        filter2D(greyMat, dEdges135, CV_8U, Diag_135);
        filter2D(greyMat, dEdges315, CV_8U, Diag_315);

        // filter2D(greyMat, dEdges, CV_8U, diagonal);

        /// Total Gradient (approximate)
        addWeighted( hEdges, 0.5, hEdges2, 0.5, 0, thEdges);
        addWeighted( vEdges, 0.5, vEdges2, 0.5, 0, tvEdges);

        addWeighted( dEdges45, 0.5, dEdges225, 0.5, 0, dEdges1);
        addWeighted( dEdges135, 0.5, dEdges315, 0.5, 0, dEdges2);

        addWeighted( dEdges1, 0.25, dEdges2, 0.25, 0, tdEdges);

         imshow( "edges", dEdges45) ;

         waitKey(0);

        Scalar mVEdge = mean(tvEdges);
        Scalar mHEdge = mean(thEdges);
        Scalar mDEdge = mean(tdEdges);


        allEdges[0]= mVEdge[0];
        allEdges[1]= mHEdge[0];
        allEdges[2]= mDEdge[0];
    }
    return allEdges;
}


/**
 *Outputs average, standard deviations pairs
 *Receives a vector with the distribution
 **/
pair<double, double> processing::processMoments(vector <pair<double,double> > vec)
{
    pair<double, double> moments;
    if(vec.size() == 1)  //It's an image
    {
        moments = vec[0];
    }
    else     //It's a video
    {
        //Compute average value
        DistCenter mean(vec.size());
        double avg = std::accumulate(vec.begin(), vec.end(), 0.0, mean);

        //Compute variance
        DistSpread var(vec.size(), avg);
        double stdv = sqrt(std::accumulate(vec.begin(), vec.end(), 0.0, var));

        moments = make_pair(avg,stdv);
    }
    return moments;
}
double processing::processFocus(Mat colorMat)
{

    Mat greyMat;
    cvtColor(colorMat, greyMat, CV_BGR2GRAY);

    Mat lap;
    Laplacian(greyMat, lap, CV_64F);

    Scalar mu, sigma;
    meanStdDev(lap, mu, sigma);

    double focusMeasure = sigma.val[0] * sigma.val[0];

    return focusMeasure;
}
int processing::processHues(Mat src, int testOutput)
{

    Mat dst;
    /// Separate the image in 3 places ( H, S and V )
    vector<Mat> bgr_planes;
    split( src, bgr_planes );

    /// Establish the number of bins
    int histSize = 180;

    /// Set the ranges ( for H,S,V) )
    float range[] = { 0, 180} ;
    const float* histRange = { range };

    bool uniform = true;
    bool accumulate = false;

    Mat h_hist;

    /// Compute the histogram for hue
    calcHist( &bgr_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate );
    //lets count the peaks on the hue histogram, we dont care about the first and last value
    int hues =0;
    for (int i=1; i<h_hist.total()-1; ++i)
    {
        float left  = h_hist.at<float>(i-1);
        float cent  = h_hist.at<float>(i);
        float right = h_hist.at<float>(i+1);

        // we have to set a boundary condition for 'plateaus',
        // so just decide to have the cutoff on the left side
        if ( left < cent  && right <= cent )
        {
            if((cent-left) > 150.0) //we discard too similar hues
                hues++;
        }
    }

    if(testOutput==1)
    {
        utility u;
        u.drawHist(h_hist,histSize );
    }
    return hues;

}
vector<pair<double,double> > processing::processColors(Mat colorMat)
{
    Scalar colAvg, colStds, greyAvg, greyStds;
    vector<pair<double,double> > result;
    result.assign(3,make_pair(0.0,0.0));

    //Compute moments of color for the image
    meanStdDev(colorMat, colAvg, colStds);
    result[0] = make_pair(colAvg[0], colStds[0]);
    result[1]=make_pair(colAvg[1], colStds[1]);
    result[2]=make_pair(colAvg[2], colStds[2]);

    //Compute moments of luminance for the image (grey-scale)
    Mat greyMat;
    cvtColor(colorMat, greyMat, CV_BGR2GRAY);
    meanStdDev(greyMat, greyAvg, greyStds);
    result[3]=make_pair(greyAvg[0], greyStds[0]);

    return result;
}

vector<double> processing::processHaarCascade(Mat colorMat, CascadeClassifier finder, CascadeClassifier aditional_cascade,
        bool insideFace,int testOutput, Mat ruleImage)
{
    vector<Rect> faces;
    vector<Rect> eyes;
    Mat frame_gray;
    int totalFaces=0;
    int totalEyes=0;
    double totalFaceArea=0.0;
    double totalRof3 =0.0;
    int frameRuleValue= 0;
    vector<double> returnVec;
    returnVec.assign(4,0.0);

    cvtColor( colorMat, frame_gray, CV_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    finder.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    totalFaces+=faces.size();
    //-- detect smiles or other things
    aditional_cascade.detectMultiScale( frame_gray, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(20, 20) );



    for (int i = 0; i < faces.size(); i++)
    {
        Point faceCenter( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        //ellipse( colorMat, faceCenter, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

        Mat faceROI = frame_gray( faces.at(i));


        Rect temp = faces.at(i);
        int w = temp.width;
        int h = temp.height;
        totalFaceArea += (float) (w*h)/(frame_gray.cols*frame_gray.rows);

        //compute rule of thirds
        Point pt;
        pt.x =temp.x+w/2;
        pt.y =temp.y+h/2;
        frameRuleValue =  ruleImage.at<unsigned char>(pt.x,pt.y);
        totalRof3 += (double) frameRuleValue/faces.size() ;

        //if smiles center inside face rect count 1
        if(insideFace)
        {
        int eyeCount = 0;

            for( size_t j = 0; j < eyes.size(); j++ )
            {

                Point eyeCenter( eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5 );
                bool fineSmile = false;

                        if(eyeCenter.inside(temp)){
                        if(eyeCenter.y > faceCenter.y && eyes[j].width < temp.width/1.3 && eyes[j].width > temp.width/3) fineSmile = true;

                            if(fineSmile){
                           // ellipse( colorMat, eyeCenter, Size( eyes[j].width*0.5, eyes[j].height*0.5), 0, 0, 360, Scalar( 255, 255, 0 ), 2, 8, 0 );
                                eyeCount++;
                                }else{} //ellipse( colorMat, eyeCenter, Size( eyes[j].width*0.5, eyes[j].height*0.5), 0, 0, 360, Scalar( 0, 255, 0 ), 1, 4, 0 );

                        }else{}// ellipse( colorMat, eyeCenter, Size( eyes[j].width*0.5, eyes[j].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 1, 4, 0 );
            fineSmile = false;
            }
            if(eyeCount > 0 ) totalEyes++;
        }
        else
        {

            for( size_t j = 0; j < eyes.size(); j++ )
            {
                totalEyes++;

                Point center( eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                //circle( colorMat, center, radius, Scalar( 255, 0, 0 ), 2, 6, 0 );
            }

        }


        if(testOutput==6)
        {
            /// Display
            namedWindow("faces", CV_WINDOW_AUTOSIZE );
            imshow("faces", colorMat );
            waitKey(0);
        }

    }

    returnVec[0]=totalFaces;
    returnVec[1]= totalFaceArea;
    returnVec[2]= totalRof3;
    returnVec[3]= totalEyes;
    return returnVec;

}

vector<float> processing::processFrameGabor(Mat gaborMat, int testOutput)
{

    Mat frame_gray;
    Mat filter_result;
    Mat viz;
    //Mat kernel, dest;
    double ksize = 31;//31
    double sigma = 5;//6
    double tetha =0;//0
    double lambda = 4;//4
    double gamma = 0.5;//0.7
    double psi = 0;//0

    double orientations[12] = {0,15,30,45,60,75,90,105,120,135,150,165};
    vector<float> returnVec;
    returnVec.assign(12,0.0);

    cvtColor( gaborMat, frame_gray, CV_BGR2GRAY );
    //equalizeHist( frame_gray, frame_gray );
    int w = frame_gray.cols;
    int h = frame_gray.rows;


    //-- start loop trough 12 orientations
     cout <<endl;

    for (int i = 0; i < 12; i++)
    {

       Mat kernel = getGaborKernel(Size(ksize,ksize), sigma, orientations[i], lambda, gamma, psi, CV_32F);
       Mat dest;
      // Mat src;
       //src.convertTo(dest, CV_32F);
        filter2D(frame_gray, dest, CV_32F, kernel);

       Scalar avgPixelIntensity = mean( dest );
       if(isnan(avgPixelIntensity.val[0])) returnVec[i] = 0.0;
       else  returnVec[i] = abs( avgPixelIntensity.val[0])/50; //this is adhoc!!

       //if (returnVec[i] < 0) returnVec[i] = 0;
      // if (returnVec[i] > 1) returnVec[i] = 1;




        if(testOutput==7)
        {
        /// Display
       // dest.convertTo(viz, CV_8U, 1.0/255.0);
            //namedWindow(" Gabor", CV_WINDOW_AUTOSIZE );
             cout << orientations[i]<<"º - value: " << returnVec[i] <<endl;
            imshow(" Gabor", dest);
            waitKey(0);
            //viz.deallocate();
           ;
        }


    }

    return returnVec;

}
vector<Mat> processing::splitMat(Mat inMat,double divide, bool bgr2gray, int testOutput)
{

        Mat allMat, region_frame;

        if(bgr2gray)
        cvtColor( inMat, allMat, CV_BGR2GRAY );
        else allMat= inMat;

        int block_width = allMat.cols * divide; //0.25
        int block_heigth = allMat.rows * divide; //0.25
        vector<Mat> regions;

        for(int y=0; y<=(allMat.rows - block_heigth); y+=block_heigth)
        {
            for(int x=0; x<=(allMat.cols - block_width); x+=block_width)
            {
                Rect region = Rect(x, y, block_width, block_heigth);
                region_frame = allMat(region);
                regions.push_back(region_frame);
            }
        }

 if(testOutput == 9){
                for(int i=0; i< regions.size(); i++){
                imshow("original", allMat);
                  cout <<"cut Image: " << i<<endl;
                 imshow("cut Image", regions[i]);
                 waitKey(0);
                 }
              }
return regions;


}


//Compute Edge Distributions
int processing::processEdgeHistogram(Mat greyMat)
{

    vector<double> orientations;
    orientations.assign(5,0.0);

    GaussianBlur(greyMat, greyMat, Size(3, 3), 0, 0, BORDER_DEFAULT);


        //5 kernels
        Mat vert_edge_filter = (cv::Mat1f(2,2) << 1,-1,1,-1);
        Mat hor_edge_filter = (cv::Mat1f(2,2) << 1,1,-1,-1);
        Mat diagr_edge_filter = (cv::Mat1f(2,2) << sqrt(2),0,0,-(sqrt(2)));
        Mat diagl_edge_filter = (cv::Mat1f(2,2) << 0,sqrt(2),-(sqrt(2)),0);
        Mat nond_edge_filter = (cv::Mat1f(2,2) << 2,-2,-2,2);

        Mat vEdges, hEdges, diagr_edges, diagl_edges, nondEdges;

        filter2D(greyMat, vEdges, CV_8U, vert_edge_filter);
        filter2D(greyMat, hEdges, CV_8U, hor_edge_filter);
        filter2D(greyMat, diagr_edges, CV_8U,  diagr_edge_filter);
        filter2D(greyMat, diagl_edges, CV_8U, diagl_edge_filter);
        filter2D(greyMat, nondEdges, CV_8U,  nond_edge_filter);

        //mean value for each filter2D result
        Scalar mVEdge = mean(vEdges);
        Scalar mHEdge = mean(hEdges);
        Scalar m45Edge = mean(diagr_edges);
        Scalar m135Edge = mean(diagl_edges);
        Scalar mNondEdge = mean(nondEdges);

        int response = 0;
        double temp_max = 0;

        if( mVEdge[0]> temp_max)
        {
        temp_max = mVEdge[0];
            response = 1;
        }
        if( mHEdge[0]> temp_max)
        {
        temp_max = mHEdge[0];
            response = 2;
        }
        if( m45Edge[0]> temp_max)
        {
        temp_max = m45Edge[0];
            response = 3;
        }
        if( m135Edge[0]> temp_max)
        {
        temp_max = m135Edge[0];
            response = 4;
        }
        if( mNondEdge[0]> temp_max)
            response = 5;

       //cout << "V: "<< mVEdge[0] << " H: "<< mHEdge[0] << " 45: "<< m45Edge[0] << " 135: "<< m135Edge[0]
       // << " nond: "<< mNondEdge[0] <<endl;
        // imshow( "edges", nondEdges) ;
        // waitKey(0);
          //return orientations;
          return response;


}

 int processing::processEHGroup(vector<int> edgeComplete){

  int zero = 0;
  int v = 0;
  int h = 0;
  int d45 = 0;
  int d135 = 0;
  int nond = 0;

     for(int i = 0; i < edgeComplete.size(); i++){

     if(edgeComplete[i] == 0) zero++;
     if(edgeComplete[i] == 1) v++;
     if(edgeComplete[i] == 2) h++;
     if(edgeComplete[i] == 3) d45++;
     if(edgeComplete[i] == 4) d135++;
     if(edgeComplete[i] == 5) nond++;

     }

   int temp = 0;
   int maxi = 0;

   if(nond > zero)
       {
       maxi=nond;
       temp = 5;
       }
   if(d135 > maxi)
       {
        maxi = d135;
        temp = 4;
        }
   if(d45 > maxi)
   {
   maxi = d45;
   temp = 3;
   }
   if(h > maxi) {
   maxi = h;
   temp = 2;

   }
   if(v > maxi){
   temp = 1;
   }

   return temp;

 }

 float processing::entropy(Mat seq, Size size, int index)
{
  int cnt = 0;
  float entr = 0;
  float total_size = size.height * size.width; //total size of all symbols in an image

  for(int i=0;i<index;i++)
  {
    float sym_occur = seq.at<float>(0, i); //the number of times a sybmol has occured
    if(sym_occur>0) //log of zero goes to infinity
      {
        cnt++;
        entr += (sym_occur/total_size)*(log2(total_size/sym_occur));
      }
  }
  //cout<<"cnt: "<<cnt<<endl;
  return entr;

}

// myEntropy calculates relative occurrence of different symbols within given input sequence using histogram
Mat processing::myEntropy(Mat seq, int histSize)
{

  float range[] = { 0, 256 } ;
  const float* histRange = { range };

  bool uniform = true; bool accumulate = false;

  Mat hist;

  /// Compute the histograms:
  calcHist( &seq, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate );

  return hist;
}






