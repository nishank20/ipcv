#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>

using namespace cv;
using namespace std;

Mat src; Mat src_gray;
int thresh = 160;
int max_thresh = 255;
RNG rng(12345);

void thresh_callback(int, void* );

int main( int, char** argv )
{
  src = imread( "dart12.jpg", IMREAD_COLOR );

  cvtColor( src, src_gray, COLOR_BGR2GRAY );
  blur( src_gray, src_gray, Size(3,3) );

  const char* source_window = "Source";
  namedWindow( source_window, WINDOW_AUTOSIZE );
  imshow( source_window, src );

  createTrackbar( " Threshold:", "Source", &thresh, max_thresh, thresh_callback );

  thresh_callback( 0, 0 );

  waitKey(0);

  return(0);
}

void thresh_callback(int, void* )
{
  Mat threshold_output;
  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;
  Mat img = imread( "dart1.jpg", IMREAD_COLOR );
  threshold( src_gray, threshold_output, thresh, 255, THRESH_BINARY );

  findContours( threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );

  vector<vector<Point> > contours_poly( contours.size() );
  vector<Rect> boundRect( contours.size() );
  vector<Point2f>center( contours.size() );
  vector<float>radius( contours.size() );

  for( size_t i = 0; i < contours.size(); i++ )
  {
    if (radius[i]<2){
        approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
        boundRect[i] = boundingRect( Mat(contours_poly[i]) );
        minEnclosingCircle( contours_poly[i], center[i], radius[i] );
    }
  }

  Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );

  for( size_t i = 0; i< contours.size(); i++ )
  {
    if (radius[i]<1 & (int)radius[i]>10){
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours_poly, (int)i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        rectangle( img, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        circle( img, center[i], (int)radius[i], color, 2, 8, 0 );
    }
    
  }

  namedWindow( "Contours", WINDOW_AUTOSIZE );
  imshow( "Contours", img);
}