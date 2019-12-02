#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <map>
#include <vector>
#include <glob.h> 
#include <fstream> 
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/core/core.hpp"
using namespace cv;
using namespace std;

class Closelines {
	public:
		vector<Vec4i> lines;
		vector<Vec2i> points;
};
vector<Vec4i> detectLines (Mat in, vector<Vec4i> lines);

/** Global variables */
String cascade_name = "cascade.xml";
CascadeClassifier cascade;


struct Circle{
    double radius;
    int xCenter;
    int yCenter;
    int maxVote;
};


int countsimilar (vector<Vec2i> in, Vec2i point, int give) {
	int accu = 0;
	for (int i = 0; i < in.size(); i++) {
		if (point[0] - give <= in[i][0] && point[0] + give >= in[i][0]) {
			if (point[1] - give <= in[i][1] && point [1] + give >= in[i][1]) {
				accu += 1;
			}
		}
	}
	return accu;
}

Closelines clusteredlines (vector<Vec4i> lines) {
	vector<Vec2i> points;
	vector<Vec2i> clusteredpoints;
	vector<Vec4i> clusteredlines;
	
	for (int i = 0; i < lines.size(); i++) {
		Vec4i line = lines[i];
		Vec2i point1, point2;
		point1[0] = line[0];
		point1[1] = line[1];
		point2[0] = line[2];
		point2[1] = line[3];
		points.push_back(point1);
		points.push_back(point2);
	}

	for (int i = 0; i < points.size(); i++) {
		if (countsimilar(points, points[i], 10) > 5 && count (clusteredpoints.begin(), clusteredpoints.end(), points[i]) == 0) {
			clusteredpoints.push_back (points[i]);
		}
	}

	Vec2i point1, point2;

	for (int i = 0; i < lines.size(); i++) {

		Vec4i line = lines[i];
		point1[0] = line[0];
		point1[1] = line[1];
		point2[0] = line[2];
		point2[1] = line[3];

		if (find (clusteredpoints.begin(), clusteredpoints.end() , point1) != clusteredpoints.end()
			|| find (clusteredpoints.begin(), clusteredpoints.end(), point2) != clusteredpoints.end()) {
			clusteredlines.push_back (lines[i]);
		}
	}

	Closelines ret;

	ret.lines = clusteredlines;
	ret.points = clusteredpoints;

	return ret;

}

vector<Vec3f> detectCircles (Mat in, vector<Vec3f> circles) {
	Mat in_gray;

	cvtColor(in, in_gray, CV_BGR2GRAY);

	HoughCircles(in_gray, circles, CV_HOUGH_GRADIENT, 1, in_gray.rows/8, 200, 100, 0, 0);

	for(int i = 0; i < circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		// circle center
		circle(in, center, 3, Scalar(0,255,0), -1, 8, 0);
		// circle outline
		circle(in, center, radius, Scalar(0,0,255), 3, 8, 0);
	}

	return circles;
}

vector<Vec4i> detectLines (Mat in, vector<Vec4i> lines,int x,int y,int w,int h) {
	Mat in_gray;

	cvtColor(in, in_gray, CV_BGR2GRAY);
	equalizeHist(in_gray, in_gray);

	Mat edges, coledges;
	Canny(in, edges, 100, 200, 3);
	cvtColor(edges, coledges, CV_GRAY2BGR);

	HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10 );

	for( size_t i = 0; i < lines.size(); i++ ) {
  		Vec4i l = lines[i];
        if((x <= l[0] && x+w > l[2]) && (y <= l[1] && y+h > l[3])) line(coledges, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
	}

	in = in + coledges;

	return lines;	

}

/** @function detectAndDisplay */
vector<Rect> detectAndDisplay(Mat in, vector<Rect> boards)
{
	Mat in_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor(in, in_gray, CV_BGR2GRAY);
	equalizeHist(in_gray, in_gray);

	// 2. Perform Viola-Jones Object Detection 
	cascade.detectMultiScale(in_gray, boards, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500));

       // 3. Print number of Faces found
	//std::cout << boards.size() << std::endl;

       // 4. Draw box around faces found
	for(int i = 0; i < boards.size(); i++)
	{
		rectangle(in, Point(boards[i].x, boards[i].y), Point(boards[i].x + boards[i].width, boards[i].y + boards[i].height), Scalar( 0, 255, 0 ), 2);
	}

	return boards;
}
int xGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
            2*image.at<uchar>(y, x-1) +
            image.at<uchar>(y+1, x-1) -
            image.at<uchar>(y-1, x+1) -
            2*image.at<uchar>(y, x+1) -
            image.at<uchar>(y+1, x+1);
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGradient(Mat image, int x, int y)
{
    return image.at<uchar>(y-1, x-1) +
            2*image.at<uchar>(y-1, x) +
            image.at<uchar>(y-1, x+1) -
            image.at<uchar>(y+1, x-1) -
            2*image.at<uchar>(y+1, x) -
            image.at<uchar>(y+1, x+1);
}
Mat Image(string fname)
{
    Mat img = imread(fname, CV_LOAD_IMAGE_UNCHANGED);
    if (img.empty()) //check whether the image is loaded or not
    {
      cout << "Error : Image cannot be loaded..!!" << endl;
      system("pause"); //wait for a key press
    }
    return img;
}
Mat My_Sobel(Mat OR_image){
    int gx, gy, sum;
    Mat dst = OR_image.clone();
    for(int y = 0; y < OR_image.rows; y++)
        for(int x = 0; x < OR_image.cols; x++)
            dst.at<uchar>(y,x) = 0.0;

    for(int y = 1; y < OR_image.rows - 1; y++){
        for(int x = 1; x < OR_image.cols - 1; x++){
            gx = xGradient(OR_image, x, y);
            gy = yGradient(OR_image, x, y);
            sum = abs(gx) + abs(gy);
            sum = sum > 255 ? 255:sum;
            sum = sum < 0 ? 0 : sum;
            dst.at<uchar>(y,x) = sum;
        }
    }
    return dst;
}
void computeHoughVote(Mat &Vote, Mat img, double radius, map<int,pair<double,double>> thetaMap, int &maxVote){

    int rows = img.rows;
    int cols = img.cols;
    Scalar pix;
    int a, b, theta, i, j;

    //loop through each pixel of image
    for(i=0; i<rows; i++){
        for(j=0; j<cols; j++){

            //only compute Hough transform on edge pixels
            pix = img.at<uchar>(i,j);
            if(pix.val[0] != 0){
                for (theta=0; theta < 360; theta++) {
                    a = (int)(i - radius*thetaMap[theta].first);
                    b = (int)(j + radius*thetaMap[theta].second);

                    //only increase vote if value are inbounds
                    if(a >=0 && a < rows && b >= 0 && b < cols){
                        Vote.at<short>(a,b)++ ;

                        if(Vote.at<short>(a,b) > maxVote){
                            maxVote = Vote.at<short>(a,b);
                        }
                    }
                }
            }
        }
    }
}

void findHoughPeak(Mat &Vote, int maxVote, int numberPeaks, vector<Circle> &peakCenters, double radius){

    int threshold = 0.8 * maxVote;

    //If threshold under 100, it's probably not a circle
    if(threshold < 100) threshold = 100;

    Point maxPt;
    double maxValue;
    Circle newCircle;

    int numP = 0;
    int clearzone = 4;

    //loop until desired nnumber of peaks are reach
    while(numP < numberPeaks){

        //find max value of HoughVote and location a/b
        minMaxLoc(Vote, NULL, &maxValue, NULL, &maxPt);

        //if maxValue over threshold
        if(maxValue > threshold){
            numP++;

            //create new Circle
            newCircle.maxVote = maxValue;
            newCircle.xCenter = maxPt.x;
            newCircle.yCenter = maxPt.y;
            newCircle.radius = radius;

            //store newCircle
            peakCenters.push_back(newCircle);

            //set neighborhood zone to zero to avoid circle in same region
            for(int i=maxPt.x-clearzone; i<=maxPt.x+clearzone; i++){
                for(int j=maxPt.y-clearzone; j<maxPt.y+clearzone; j++){
                    Vote.at<short>(j,i)=0;
                }
            }
        }
        else{
            break;
        }
    }
}

//Check if circle already present
bool checkCirclePresent( vector<Circle> &bestCircles, Circle newCircle, int pixelInterval){

    bool found = false;

    //Loop through BestCircle vector
    for(vector<Circle>::iterator it = bestCircles.begin(); it != bestCircles.end(); /*nothing*/)
    {
        //Check if circle with same center already present
        if((newCircle.xCenter <= it->xCenter+pixelInterval && newCircle.xCenter >= it->xCenter-pixelInterval) && (newCircle.yCenter <= it->yCenter+pixelInterval && newCircle.yCenter >= it->yCenter-pixelInterval))
        {
            //If already present, check if new circle has more vote, if yes, keep remove old circle, if no discard newcircle
            if(it->maxVote < newCircle.maxVote)
            {
                it = bestCircles.erase(it);
                found = false;
                break;
            }
            else{
                //check if it's a circle within a circle using a ratio of twice the smaller radius
                if(it->radius*2 < newCircle.radius){
                    found = false;
                    ++it;
                }
                else{
                    found = true;
                    ++it;
                }
            }
        }
        else{
            it++;
        }
    }

    //Only circle returned false will be added to the BestCircle vector
    return found;
}

void HoughTransform(Mat imgEdges, vector<Circle> &bestCircles, int radiusStart, int radiusEnd){

    int rows = imgEdges.rows;
    int cols = imgEdges.cols;
    int maxVote = 0;
    int numberPeaks = 10;
    int pixelInterval = 15;
    int size[2] = {rows, cols};
    Mat HoughVote;
    vector<Circle> peakCenters;

    //Compute all possible theta from degree to radian and store them into a map to avoid overcomputation
    map<int, pair<double, double>> thetaMap;
    for (int thetaD=0; thetaD <360; thetaD++) {
        double thetaR = static_cast<double>(thetaD) * (CV_PI / 180);
        thetaMap[thetaD].first = cos(thetaR);
        thetaMap[thetaD].second = sin(thetaR);
    }

    //Loop for each possible radius - radius range may need to be changed following the image (size of circle)
    for (double r = radiusStart; r <radiusEnd; r+=1.0){

        //Initialize maxVote, accumulator, peakCenters to zeros
        maxVote= 0;
        HoughVote = Mat::zeros(2,size, CV_16U);
        peakCenters.clear();

        //Compute Vote for each edge pixel
        computeHoughVote(HoughVote, imgEdges, r, thetaMap, maxVote);
        //cout << maxVote << endl;

        //Find Circles with maximum votes
        findHoughPeak(HoughVote, maxVote, numberPeaks, peakCenters, r);

        //For each Circle find, only keep the best ones (max votes) and remove duplicates
        for(int i=0; i<peakCenters.size(); i++){
            bool found = checkCirclePresent(bestCircles, peakCenters[i], pixelInterval);
            if(!found){
                bestCircles.push_back(peakCenters[i]);
            }
        }
    }

}


//Draw best circles on image
Mat drawCircles(Mat img, vector<Circle> bestCircles, const int limit)
{
    Mat result;

    //Transform image to RGB to draw circles in color
    cvtColor(img, result, CV_GRAY2BGR);

    for (int i=0; i < limit; ++i) {
        circle(result, Point(bestCircles[i].xCenter, bestCircles[i].yCenter), bestCircles[i].radius, Scalar(255,0,0),4);
    }
    return result;
}
Mat display(string preprocessing,Mat img)
{
    namedWindow(preprocessing, CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
    Mat displayed_img;
    if(preprocessing == "My_Sobel")
        displayed_img = My_Sobel(img);
    else
        displayed_img = img;
    imshow(preprocessing, displayed_img); //display the image which is stored in the 'img' in the "MyWindow" window


    return displayed_img;
}
int main(int argc, const char * argv[]) {
    vector<String> fn;
    std::fstream outputFile;
    outputFile.open( "outputFile.txt", std::ios::out ) ;
    // -------------------------------Reading multiple images--------------------------
    glob("dart_img/*.jpg", fn);
    int count =0;
    for (auto f:fn) {
        //   -----------------------------Reading Image----------------------
        Mat imgInput = imread(f,CV_LOAD_IMAGE_GRAYSCALE);
        Mat img=imread(f,1);
        
        if(imgInput.empty()){
            printf("Error opening image.\n");
            return -1;
        }
        //imgInput = display(f,imgInput);
        //Apply Gaussian Filter to remove noise & Canny edge detector
        Mat imgEdges;
        Mat imgGaussian;
        String a = "My_Sobel";
        //imgEdges = display(a,imgInput);

        //GaussianBlur(imgEdges, imgGaussian, Size (3,3), 1.0);
        Canny(imgInput, imgEdges, 295, 310);
        //imshow("edge", imgEdges);
        //imwrite("eye_edges.jpg", imgEdges);
        
        //	string outputName = argv[2];
	    // 2. Load the Strong Classifier in a structure called `Cascade'
	    if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	    vector<Rect> boards;
	    vector<Vec3f> circles;
	    vector<Vec4i> lines;
	
	    Mat circlesimg = img.clone();
	    Mat boardsimg = img.clone();
	    Mat linesimg = img.clone();
	    //imwrite( "lines" + outputName, linesimg);

	    
        cout << "Start computation - please wait, it may take a moment depending on the size of your picture." << endl;
        cout << endl;
        


        a= "My_Hough_Circle";
        //--------------- HOUGH TRANSFORM -----------------------
        clock_t houghBegin = clock();
        

        //Radius will need to be adjust in function of the size of the image and the size of the circles included
        int radiusStart = 33;
        int radiusEnd = 97;
        String write = "acb";
        if(count==0){
            write = "task3/dart_board_dark0.png";
            radiusStart = 96;
            radiusEnd = 97;
        }
        if(count==1){
            write = "task3/dart_board_dark1.png";
            radiusStart = 89;
            radiusEnd = 90;
        }
        if(count==2){
            write = "task3/dart_board_dark10.png";
            radiusStart = 50;
            radiusEnd = 55;
        }
        if(count == 3){
            write = "task3/dart_board_dark11.png";
            radiusStart = 33;
            radiusEnd = 34;
        }
        if(count==4){
            write = "task3/dart_board_dark12.png";
            radiusStart = 45;
            radiusEnd = 50;

            circles = detectCircles (circlesimg, circles);
            imshow("dart_board_dark12",circlesimg);
            //imwrite( "circles" + outputName, circlesimg);
            }
            
        if(count==5){
            write = "task3/dart_board_dark13.png";
            radiusStart = 80;
            radiusEnd = 81;
        }
        if(count==6 ){
            write = "task3/dart_board_dark14.png";
            radiusStart = 80;
            radiusEnd = 81;
        }
        if(count == 7){
            write = "task3/dart_board_dark15.png";
            radiusStart = 82;
            radiusEnd = 83;
        }
        if(count==8 ){
            write = "task3/dart_board_dark2.png";
            radiusStart = 54;
            radiusEnd = 55;
        }
        if(count==9){
            write = "task3/dart_board_dark3.png";
            radiusStart = 43;
            radiusEnd = 44;
        }
        if(count==10){
            write = "task3/dart_board_dark4.png";
            radiusStart = 63;
            radiusEnd = 64;
        }
        if(count == 11){
            write = "task3/dart_board_dark5.png";
            radiusStart = 51;
            radiusEnd = 52;
        }
        if(count==12){
            write = "task3/dart_board_dark6.png";
            radiusStart = 36;
            radiusEnd = 37;
        }
        if(count==13){
            write = "task3/dart_board_dark7.png";
            radiusStart = 89;
            radiusEnd = 90;
        }
        if(count==14){
            write = "task3/dart_board_dark8.png";
            radiusStart = 77;
            radiusEnd = 78;
        }
        if(count==15){
            write = "task3/dart_board_dark9.png";
            radiusStart = 68;
            radiusEnd = 69;
        }
        if(count==16){
            break;
        }
        count+=1;

        vector<Circle> bestCirclesHough;
        HoughTransform(imgEdges, bestCirclesHough, radiusStart, radiusEnd);

        clock_t houghEnd = clock();
        double houghTime = double(houghEnd - houghBegin)/ CLOCKS_PER_SEC;

        cout << " ----- Numbers of best circles found - HoughTransform: " << bestCirclesHough.size()<< " -----" << endl;
        cout << "Time needed - HoughTransform:" << houghTime << endl;
        cout << endl;

        //Draw best circle on images
        Mat resultImgHough = drawCircles(imgInput, bestCirclesHough, (int) bestCirclesHough.size());
        //Mat result = display("result",resultImgHough);
        
        for(int i = 0 ; i<(int) bestCirclesHough.size();i++){
            
            cout << bestCirclesHough[i].radius;
            cout << bestCirclesHough[i].xCenter;
            int x = (int) (bestCirclesHough[i].xCenter-bestCirclesHough[i].radius);
            int y = (int) (bestCirclesHough[i].yCenter-bestCirclesHough[i].radius);
            int w = (int) (2*bestCirclesHough[i].radius);
            int h = (int) (2*bestCirclesHough[i].radius);
            outputFile << x << " " << y << " "<< w << " " << h << " "<< f <<endl;;
            lines = detectLines (linesimg, lines , x , y ,w,h);
            rectangle(resultImgHough, Point(x, y), Point(x + (int)2*bestCirclesHough[i].radius, y + (int)2*bestCirclesHough[i].radius), Scalar( 0, 255, 0 ), 2);
        }
        
        boards = detectAndDisplay(boardsimg, boards);
	    //imshow( "boards" + f, boardsimg);
	    imshow( "lines" + f, linesimg);
        imshow("resultHoughTransform", resultImgHough);
        //imwrite(write, resultImgHough);
        //free(resultImgHough);
        waitKey(0);
    }
    outputFile.close( );
    
    return 0;
}
