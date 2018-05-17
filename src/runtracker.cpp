#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <ctime>
#include <chrono>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"
#include "cuda_profiler_api.h"

#include <dirent.h>

using namespace std;
using namespace cv;

double time_interval(clock_t start, clock_t end)
{
	return 1000.0 * (end - start) / CLOCKS_PER_SEC;
}

void init_FR()
{
    PyObject *pName, *pModule, *pDict, *pInitFR;

    const char* module_name = "demo";
    const char* init_frcnn = "init_faster_rcnn";

    Py_Initialize();

    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.argv=['pdm']");
    PyRun_SimpleString("sys.path.append('./')");
    PyRun_SimpleString("sys.path.append('./tools')");

    pName = PyString_FromString(module_name);
}

int main(int argc, char* argv[]){

	if (argc > 6) return -1;

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;
        int no_wait = 1;

	for(int i = 0; i < argc; i++){
		if ( strcmp (argv[i], "hog") == 0 )
			HOG = true;
		if ( strcmp (argv[i], "fixed_window") == 0 )
			FIXEDWINDOW = true;
		if ( strcmp (argv[i], "singlescale") == 0 )
			MULTISCALE = false;
		if ( strcmp (argv[i], "show") == 0 )
			SILENT = false;
		if ( strcmp (argv[i], "lab") == 0 ){
			LAB = true;
			HOG = true;
		}
		if ( strcmp (argv[i], "gray") == 0 )
			HOG = false;
                if ( strcmp (argv[i], "wait") == 0 )
                        no_wait = 0;
	}
	
	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;
	vector<Mat> images;

	// Tracker results
	Rect result;

	// Path to list.txt
	ifstream listFile;
	string fileName = "images.txt";
  	listFile.open(fileName);

  	// Read groundtruth for the 1st frame
  	ifstream groundtruthFile;
	string groundtruth = "region.txt";
  	groundtruthFile.open(groundtruth);
  	string firstLine;
  	getline(groundtruthFile, firstLine);
	groundtruthFile.close();
  	
  	istringstream ss(firstLine);

  	// Read groundtruth like a dumb
  	float x1, y1, x2, y2, x3, y3, x4, y4;
  	char ch;
	ss >> x1;
	ss >> ch;
	ss >> y1;
	ss >> ch;
	ss >> x2;
	ss >> ch;
	ss >> y2;
	ss >> ch;
	ss >> x3;
	ss >> ch;
	ss >> y3;
	ss >> ch;
	ss >> x4;
	ss >> ch;
	ss >> y4; 

	// Using min and max of X and Y for groundtruth rectangle
	float xMin =  min(x1, min(x2, min(x3, x4)));
	float yMin =  min(y1, min(y2, min(y3, y4)));
	float width = max(x1, max(x2, max(x3, x4))) - xMin;
	float height = max(y1, max(y2, max(y3, y4))) - yMin;

	
	// Read Images
	ifstream listFramesFile;
	string listFrames = "images.txt";
	listFramesFile.open(listFrames);
	string frameName;


	// Write Results
	ofstream resultsFile;
	string resultsPath = "output.txt";
	resultsFile.open(resultsPath);

	// read all images into memory
	while(getline(listFramesFile, frameName))
		images.push_back(imread(frameName, CV_LOAD_IMAGE_COLOR));

	// Init and warm up gpu
	//frame = imread(frameName, CV_LOAD_IMAGE_COLOR);
	tracker.init( Rect(xMin, yMin, width, height), images[0] );
	rectangle( images[0], Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );
	resultsFile << xMin << "," << yMin << "," << width << "," << height << endl;

	// Frame counter
	int nFrames = 0;

	cudaProfilerStart();
        auto tic = chrono::high_resolution_clock::now();
	for(int i = 1; i < images.size(); i++){
		// frameName = frameName;

		// Read each frame from the list
		frame = images[i];

		// Update
		result = tracker.update(frame);
		rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );
		resultsFile << result.x << "," << result.y << "," << result.width << "," << result.height << endl;

		nFrames++;

		if (!SILENT){
			imshow("Image", frame);
			waitKey(no_wait);
		}
	}
        auto toc = chrono::high_resolution_clock::now();
        double total_time_ms = chrono::duration<double, std::milli>(toc-tic).count();
	cudaProfilerStop();

	cufhog_finalize();
	
        cout << "total time: " << total_time_ms / 1000.0 << "s" << endl;
	cout << "time per frame: " << total_time_ms / nFrames << "ms" << endl;

	resultsFile.close();

	listFile.close();

}
