
#include <iostream>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <string>
#include <stdlib.h>
#include <fstream>

#include <opencv2\opencv.hpp>

// Configuration file path
#define TRACKER_CONF_PATH ("./config/tracker-conf.ini")

// Windows prefix names and exit key
#define MAIN_WINDOW_NAME ("tracker window")
#define FACE_WINDOW_NAME ("face-window-")
#define EXIT_KEY_CODE (27)

// Maximum number of features inside one ROI.
#define MAX_CORNERS_TO_DETECT_INSIDE_ONE_ROI 40

#define MAX_MUTEXES_BUFFER_SIZE 100

// Default configuration values.
#define DEF_VIDEO_CAPTURE_RESOURCE (0)
#define DEF_HAARCASCADE_FRONTAL_FACE_PATH ("haarcascade_frontalface_alt2.xml")
#define DEF_HAARCASCADE_EYE_PATH ("haarcascade_eye.xml")
#define DEF_HAARCASCADE_NOSE_PATH ("nose.xml")
#define DEF_HAARCASCADE_MOUTH_PATH ("mouth.xml")

// Configuration file fields.
#define CONF_FIELD_IS_CAMERA ("IS_CAMERA")
#define CONF_FIELD_RESOURCE ("RESOURCE")
#define CONF_FIELD_VIDEO_PATH ("VIDEO_PATH")
#define CONF_FIELD_FPS ("FPS")
#define CONF_FIELD_IS_RECORD ("IS_RECORD")
#define CONF_FIELD_OUTPUT_VIDEO_PATH ("OUTPUT_VIDEO_PATH")
#define CONF_FIELD_HAAR_FACE_FEATURES_PATH ("HAAR_FACE_FEATURES_PATH")
#define CONF_FIELD_HAAR_EYE_FEATURES_PATH ("HAAR_EYE_FEATURES_PATH")
#define CONF_FIELD_HAAR_NOSE_FEATURES_PATH ("HAAR_NOSE_FEATURES_PATH")
#define CONF_FIELD_HAAR_MOUTH_FEATURES_PATH ("HAAR_MOUTH_FEATURES_PATH")

/*
 * Configuration fields
 * to the frames sampler thread :
 * 1. isWebcam - Does the video streaming from webcam ?
 * 2. resource - In case of camera, this is the resrouce number.
 * 3. videoPath - In case of video file, this is the video file path
 * 4. fps - The camera/video streaming FPS value.
 * 5. faceHaarFeaturesPath - Haar features .xml file for the face.
 * 6. eyeHaarFeaturesPath - Haar features .xml file for the eye.
 * 7. noseHaarFeaturesPath - Haar features .xml file for the nose.
 * 8. mouthHaarFeaturesPath - Haar features .xml file for the mouth.
 */
typedef struct
{
	bool isWebcam;
	int resource;
	std::string videoPath;
	int fps;
	bool isRecord;
	std::string outputVideoName;
	std::string faceHaarFeaturesPath;
	std::string eyeHaarFeaturesPath;
	std::string noseHaarFeaturesPath;
	std::string mouthHaarFeaturesPath;
} TTrackerConfigurations;

/*
 * Those are shared parameters between the main thread (the tracker)
 * to the frames sampler thread :
 * 1. framesQueue - This is the queue that contains the frames entered by the sampler
 *					thread (the producer) and taken by the main thread (the consumer).
 * 2. samplerLock - Mutex for the frames queue.
 * 3. programRunning - Indicates if the sampler thread still running.
 * 4. fps - Frames per seconds of the current stream (camera or video).
 * 5. trackerConfigurations - Configurations for the tracker (default parameters or from file).
 */
std::queue<cv::Mat> framesQueue;
std::mutex samplerLock;
bool programRunning = true;
TTrackerConfigurations trackerConfigurations;

/*
 * Structure that contains different ROIs for different
 * part of the face.
 * For the eyes we have a vector because there are two (but no always
 * two eyes detected).
 */
typedef struct
{
	cv::Rect face;
	std::vector<cv::Rect> eyes;
	cv::Rect nose;
	cv::Rect mouth;
} TFacialROIs;

/*
 * Structure that represent properties of one face window
 * that the user initiate.
 * active - The user choose to active the window (by double click).
 * created - The window already created.
 * name - The name of the window.
 */
typedef struct
{
	bool active;
	bool created;
	std::string name;
} TFaceWindowParams;

/*
 * Structure that contains the data that need to be passed
 * to the mouse callback :
 * Pointers to the window parameters and pointer to the
 * current ROIs (faces regions) in the video.
 */
typedef struct
{
	std::vector<std::vector<cv::Point2f>>* currROIs;
	std::vector<TFaceWindowParams>* faceWindowsParams;
} TMouseCallbackData;

/*
 * Load the tracker configuration from the ini file.
 */
void loadConfigurations();

/*
 * The function return the source ROI translated by its container ROI.
 */
cv::Rect getTranslatedROI(
	const cv::Rect& srcROI, 
	const cv::Rect& containerROI);

/*
 * Detect facial rectangular ROIs (face,eyes,nose,mouth) and save them.
 * The classifiers are outside the detection function for one time loading.
 */
cv::CascadeClassifier faceClassifier;
cv::CascadeClassifier eyeClassifier;
cv::CascadeClassifier noseClassifier;
cv::CascadeClassifier mouthClassifier;
bool detectFacialROIs(
	const cv::Mat& grayscaleImg, 
	std::vector<TFacialROIs>& facialROIsVector);

/*
 * Detect "good features" (Shi-Tomasi) inside given ROIs.
 */
bool findFeaturesInsideInfaceROIs(
	const cv::Mat& grayscaleImg, 
	const std::vector<TFacialROIs>& facialROIsVector, 
	std::vector<std::vector<cv::Point2f>>& featuresGroups);

/*
 * Build image pyramid that will be used in LK optical flow.
 */
bool buildLKPyr(
	const cv::Mat& grayscaleImg, 
	std::vector<cv::Mat>& pyr);

/*
 * Perform LK optical flow according to given image pyramids and source features set.
 */
bool calcLKOpticalFlowForAllTheFeaturesGroups(
	const std::vector<cv::Mat>& prevPyr, 
	const std::vector<cv::Mat>& currPyr,
	const std::vector<std::vector<cv::Point2f>>& prevFeaturesGroups, 
	std::vector<std::vector<cv::Point2f>>& currFeaturesGroups);

/*
 * Drawing funcions (ROIs, features).
 */
void drawFacialROIsVector(
	const std::vector<TFacialROIs>& facialROIsVector, 
	cv::Mat& img);
void drawFacialFeaturesGroups(
	const std::vector<std::vector<cv::Point2f>>& facialFeaturesGroups, 
	cv::Mat& img);
void drawPoly(
	const std::vector<cv::Point2f>& pts, 
	cv::Mat& img);
void drawROIs(
	const std::vector<std::vector<cv::Point2f>>& ROIs, 
	cv::Mat& img);

/*
* convert rectangle to points set (polygon).
*/
void convertRectToPts(
	const cv::Rect& rect, 
	std::vector<cv::Point2f>& pts);

/*
* Convert initial ROIs to polygons.
*/
void convertRectROIsToPolys(
	const std::vector<TFacialROIs>& facialROIsVector, 
	std::vector<std::vector<cv::Point2f>>& ROIs);

/*
* Get transformation matrices for the faces ROIs transformations.
* Get inverse transformation matrices for video stabilization in the individual face window.
*/
bool getRigidTransformationMatrices(
	const std::vector<std::vector<cv::Point2f>>& currFeaturesGroups, 
	const std::vector<std::vector<cv::Point2f>>& prevFeaturesGroups, 
	const std::vector<std::vector<cv::Point2f>>& initROIs, 
	const std::vector<std::vector<cv::Point2f>>& currROIs, 
	std::vector<cv::Mat>& transMatrices, 
	std::vector<cv::Mat>& transMatricesInv);

/*
* Performing rigid transformation on the faces ROIs.
*/
void performRigidTransformationOnROIs(
	const std::vector<cv::Mat>& transMatrices, 
	std::vector<std::vector<cv::Point2f>>& ROIs,
	cv::Size& imgSize);

/*
* Perform rigid transformation on an image.
*/
void performRigidTransformationImg(
	const cv::Mat& M, 
	cv::Mat& img);

/*
* Acquire one frame from the frames buffer (in case of webcam that performes buffering).
*/
bool acquireFrameFromBuffer(
	cv::Mat& outputFrame);

/*
* Thread function of the frames sampler.
*/
void framesSamplerThreadFunction();

/*
* Check if given point is inside given ROI.
*/
bool isPointInsideROI(
	const cv::Point2f& pt,
	const std::vector<cv::Point2f>& ROI);

/*
* Reduce/Enlarge given ROI by specific amount of percentage.
*/
cv::Rect getReducedROI(
	const cv::Rect& srcROI, double percents);
cv::Rect getEnlargedROI(
	const cv::Rect& srcROI, double percents);

/*
* Callback for the mouse event.
*/
void mainWindowMouseCallback(
	int event, 
	int x, 
	int y, 
	int, 
	void* data);


/*
Structure contains details for single face tracker thread.
*/
typedef struct
{
	cv::Mat inv; // Invers transformation matrix
	cv::Mat img; // Image to manipulate with the invers matrix
} TFaceWindowThreadParams;

// Mutexes buffer for each thread of video stabilization.
std::mutex mtxs[MAX_MUTEXES_BUFFER_SIZE];
// Queues that contain tasks for each one of the video stabilization threads.
std::vector<std::queue<TFaceWindowThreadParams>> faceThreadsBuffers;
// Initial facial ROIs detected by the Haar cascade detector - once detected, never changed during the program.
std::vector<TFacialROIs> currFacialROIsVector;
// Details for the video stabilization windows - once we know the number of faces, it will not change (except window status).
std::vector<TFaceWindowParams> faceWindowsParams;

/*
* Thread function for one video stailizer (number of stabilizers = number of faces).
*/
void faceThreadFunction(int queueIndex);

/*
* Get the appropriate loop delay by the video FPS in the configuration.
*/
int getMainLoopDelayByVideoFPS();


int main(int argc, char** argv)
{
	// Load tracker configurations.
	loadConfigurations();

	// Get loop delay.
	int delay = getMainLoopDelayByVideoFPS();

	cv::VideoCapture cap;
	std::thread streamingJob;

	// In case of real time streaming (camera) we need buffering for
	// the frames.
	// In case of video file, read the frames regulary and set delays
	// according the FPS of the video.
	if (trackerConfigurations.isWebcam)
	{
		streamingJob = std::thread(framesSamplerThreadFunction);
		// First we need to start the streaming thread.
		std::cout << "started video streaming job" << std::endl;
	}
	else
	{
		cap.open(trackerConfigurations.videoPath);
		if (!cap.isOpened())
		{
			std::cout << "video capture failed to open video file " << trackerConfigurations.videoPath << std::endl;
			return 1;
		}
	}

	// Initial detection indication.
	bool initialProcessingNeeded = true;

	// Window creation.
	cv::namedWindow(MAIN_WINDOW_NAME, cv::WINDOW_NORMAL);
	std::cout << "window " << MAIN_WINDOW_NAME << " created" << std::endl;

	// Starting main loop.
	std::cout << "starting tracker main loop" << std::endl;

	
	//---------------------------------------------------------
	cv::Mat currBGRFrame;
	cv::Mat currBGRFrameCopy;
	cv::Mat currGrayscaleFrame;
	cv::Mat prevGrayscaleFrame;
	//---------------------------------------------------------
	std::vector<cv::Mat> currPyr;
	std::vector<cv::Mat> prevPyr;
	//---------------------------------------------------------
	
	std::vector<std::vector<cv::Point2f>> initROIs;
	std::vector<std::vector<cv::Point2f>> currROIs;
	//---------------------------------------------------------
	std::vector<std::vector<cv::Point2f>> currFeaturesGroups;
	std::vector<std::vector<cv::Point2f>> prevFeaturesGroups;
	//---------------------------------------------------------
	std::vector<cv::Mat> transMatrices;
	std::vector<cv::Mat> transMatricesInv;
	//---------------------------------------------------------
	std::vector<std::pair<std::thread, bool>> facesThreads;

	bool windowResized = false;

	// Object used for writing the results into video file.
	cv::VideoWriter outputVideo;
	std::string outputVideoPath = trackerConfigurations.outputVideoName;
	bool videoWriterInitialized = false;

	while (programRunning)
	{
		// Sample frame
		bool goodSampling = true;
		if (trackerConfigurations.isWebcam)
		{
			goodSampling = acquireFrameFromBuffer(currBGRFrame);
		}
		else
		{
			cap >> currBGRFrame;
			if (currBGRFrame.empty())
			{
				goodSampling = false;
			}
		}

		// Process frame only if sampling succeeded.
		if (goodSampling)
		{

			if (!videoWriterInitialized)
			{
				// Initialize the video writer.
				if (trackerConfigurations.isRecord)
				{
					outputVideo.open(outputVideoPath, CV_FOURCC('D', 'I', 'V', 'X'), trackerConfigurations.fps, currBGRFrame.size(), true);
					if (!outputVideo.isOpened())
					{
						std::cout << "Could not open the output video for write: " << outputVideoPath << std::endl;
						programRunning = false;
						break;
					}
					videoWriterInitialized = true;
				}
				
			}

			if (!windowResized)
			{
				cv::resizeWindow(MAIN_WINDOW_NAME, currBGRFrame.size().width, currBGRFrame.size().height);
				windowResized = true;
			}

			// Copy frame and convert it to grayscale.
			currBGRFrameCopy = currBGRFrame.clone();
			cv::cvtColor(currBGRFrameCopy, currGrayscaleFrame, cv::COLOR_RGB2GRAY);

			// Perform histogram equalization (for areas with inconsistent illumination).
			cv::equalizeHist(currGrayscaleFrame, currGrayscaleFrame);
			// ------------------------------------------
			//			Initial processing
			// ------------------------------------------
			if (initialProcessingNeeded)
			{

				// Try to detect facial features.
				bool facialROIsDetectionSucceeded = detectFacialROIs(currGrayscaleFrame, currFacialROIsVector);
				// If facial features detection succeeded.
				if (facialROIsDetectionSucceeded)
				{
					// Try to detect minimum number of good features inside the facial ROIs.
					bool facialFeaturesDetectionSucceeded = findFeaturesInsideInfaceROIs(currGrayscaleFrame, currFacialROIsVector, currFeaturesGroups);
					// If features detection succeeded.
					if (facialFeaturesDetectionSucceeded)
					{
						// No initial processing needed for the rest of the cycles.
						initialProcessingNeeded = false;

						// Convert the initial facial ROIs to the transformed ROIs represented by polygons.
						convertRectROIsToPolys(currFacialROIsVector, currROIs);

						// Save initial features groups.
						initROIs = currROIs;

						// Set empty transformation matrices for the followed processing cycles.
						transMatrices.clear();
						transMatricesInv.clear();
						for (int i = 0; i < currFacialROIsVector.size(); i++)
						{
							transMatrices.push_back(cv::Mat());
							transMatricesInv.push_back(cv::Mat());
						}

						// Set potential windows parameters.
						for (int i = 0; i < currFacialROIsVector.size(); i++)
						{
							TFaceWindowParams windowParams;
							// Not created and not active yet.
							windowParams.active = windowParams.created = false;
							std::string windowName = FACE_WINDOW_NAME;
							char buff[3];
							_itoa(i + 1, buff, 10);
							windowName += buff;
							// Constant name + index number.
							windowParams.name = windowName;
							// Save it.
							faceWindowsParams.push_back(windowParams);
							// Add queue for each one of the video stabilizers.
							faceThreadsBuffers.push_back(std::queue<TFaceWindowThreadParams>());
							
						}

						// Load mouse callback data.
						TMouseCallbackData mouseCallbackData;
						mouseCallbackData.faceWindowsParams = &faceWindowsParams;
						mouseCallbackData.currROIs = &currROIs;
						// Set mouse callback to the main tacker window.
						cv::setMouseCallback(MAIN_WINDOW_NAME, mainWindowMouseCallback, &mouseCallbackData);

						// Initialize sabilizers threads.
						for (int i = 0; i < currFacialROIsVector.size(); i++)
						{
							facesThreads.push_back(std::pair<std::thread, bool>(std::thread(faceThreadFunction, i), true));
						}
					}
				}
			}
			// ------------------------------------------
			//			Processing for the rest of the frames
			// ------------------------------------------
			else
			{
				// Build images pyramids for optical flow.
				buildLKPyr(prevGrayscaleFrame, prevPyr);
				buildLKPyr(currGrayscaleFrame, currPyr);

				// Perform optical flow calculation.
				calcLKOpticalFlowForAllTheFeaturesGroups(prevPyr, currPyr, prevFeaturesGroups, currFeaturesGroups);

				// Get transformation matrices for the ROIs tracking and for the video stabilization of individual faces.
				if (getRigidTransformationMatrices(currFeaturesGroups, prevFeaturesGroups, initROIs, currROIs, transMatrices, transMatricesInv))
				{
					// Perform the transformation for the ROIs.
					performRigidTransformationOnROIs(transMatrices, currROIs, currBGRFrameCopy.size());
				}

				

				// Move over each one of the active face tracking windows and update parameters appropriatelly.
				for (int i = 0; i < faceWindowsParams.size(); i++)
				{
					if (faceWindowsParams[i].active && faceWindowsParams[i].created)
					{
						// Prepare record for the video stabilizer.
						TFaceWindowThreadParams fwtp;
						fwtp.img = currBGRFrameCopy.clone();
						fwtp.inv = transMatricesInv[i];
						// Enter the record to the approprite queue.
						mtxs[i].lock();
						faceThreadsBuffers[i].push(fwtp);
						mtxs[i].unlock();
					}
				}
			}

			// Draw features groups and ROIs.
			drawFacialFeaturesGroups(currFeaturesGroups, currBGRFrameCopy);
			drawROIs(currROIs, currBGRFrameCopy);

			// Show the image in the tracker window.
			cv::imshow(MAIN_WINDOW_NAME, currBGRFrameCopy);

			// Save current data as previous data.
			// Grayscale image
			cv::swap(prevGrayscaleFrame, currGrayscaleFrame);
			// Features groups
			prevFeaturesGroups.clear();
			for (std::vector<cv::Point2f>& cg : currFeaturesGroups)
			{
				std::vector<cv::Point2f> pg;
				std::swap(pg, cg);
				prevFeaturesGroups.push_back(pg);
			}
			// Images pyramid
			prevPyr.clear();
			for (cv::Mat& cm : currPyr)
			{
				cv::Mat pm;
				cv::swap(pm, cm);
				prevPyr.push_back(pm);
			}
		
			// If recording write the current frame to the file.
			if (videoWriterInitialized)
			{
				outputVideo << currBGRFrameCopy;
			}

		}
		// If no good sampling or video is over, stop the program.
		else
		{
			if (!trackerConfigurations.isWebcam)
			{
				programRunning = false;
			}
		}

		// Check for user request to finish the program.
		int key = cv::waitKey(delay);
		if (key == EXIT_KEY_CODE)
		{
			std::cout << "user stoped main loop" << std::endl;
			programRunning = false;
			break;
		}
		
	}
	
	std::cout << "main loop ended" << std::endl;
	cv::destroyAllWindows();
	std::cout << "resources released" << std::endl;
	std::cout << "program ended successfully" << std::endl;

	if (trackerConfigurations.isWebcam)
	{
		streamingJob.join();
	}

	for (int i = 0; i < facesThreads.size(); i++)
	{
		if (facesThreads[i].second)
		{
			facesThreads[i].first.join();
		}
	}

	return 0;
}

void loadConfigurations()
{
	
	// Create input stream to the file.
	std::ifstream fis;
	fis.open(TRACKER_CONF_PATH);

	// Check if stream opened successfully
	if (!fis.good())
	{
		std::cout << "program failed to open configuration file : " << TRACKER_CONF_PATH << std::endl;
		std::cout << "loading default configuration" << std::endl;

		// For now load only default configurations.
		trackerConfigurations.isWebcam = true;
		trackerConfigurations.resource = DEF_VIDEO_CAPTURE_RESOURCE;
		trackerConfigurations.videoPath = "";
		trackerConfigurations.fps = 30;
		trackerConfigurations.isRecord = false;
		trackerConfigurations.outputVideoName = "";
		trackerConfigurations.faceHaarFeaturesPath = DEF_HAARCASCADE_FRONTAL_FACE_PATH;
		trackerConfigurations.eyeHaarFeaturesPath = DEF_HAARCASCADE_EYE_PATH;
		trackerConfigurations.noseHaarFeaturesPath = DEF_HAARCASCADE_NOSE_PATH;
		trackerConfigurations.mouthHaarFeaturesPath = DEF_HAARCASCADE_MOUTH_PATH;
	}

	std::string line;
	char delimeter = '=';
	// Read all lines
	while (std::getline(fis, line))
	{
		// Read the field name.
		std::string field = line.substr(0, line.find(delimeter));
		// Get string value.
		std::string sval = line.substr(line.find(delimeter) + 1);

		std::cout << line << std::endl;

		if (field == CONF_FIELD_IS_CAMERA)
		{
			std::istringstream iss(sval);
			iss >> trackerConfigurations.isWebcam;
		}
		else if (field == CONF_FIELD_RESOURCE)
		{
			std::istringstream iss(sval);
			iss >> trackerConfigurations.resource;
		}
		else if (field == CONF_FIELD_VIDEO_PATH)
		{
			trackerConfigurations.videoPath = sval;
		}
		else if (field == CONF_FIELD_FPS)
		{
			std::istringstream iss(sval);
			iss >> trackerConfigurations.fps;
		}
		else if (field == CONF_FIELD_HAAR_FACE_FEATURES_PATH)
		{
			trackerConfigurations.faceHaarFeaturesPath = sval;
		}
		else if (field == CONF_FIELD_HAAR_EYE_FEATURES_PATH)
		{
			trackerConfigurations.eyeHaarFeaturesPath = sval;
		}
		else if (field == CONF_FIELD_HAAR_NOSE_FEATURES_PATH)
		{
			trackerConfigurations.noseHaarFeaturesPath = sval;
		}
		else if (field == CONF_FIELD_HAAR_MOUTH_FEATURES_PATH)
		{
			trackerConfigurations.mouthHaarFeaturesPath = sval;
		}
		else if (field == CONF_FIELD_IS_RECORD)
		{
			std::istringstream iss(sval);
			iss >> trackerConfigurations.isRecord;
		}
		else if (field == CONF_FIELD_OUTPUT_VIDEO_PATH)
		{
			trackerConfigurations.outputVideoName = sval;
		}
	}
	
}


bool isPointInsideROI(const cv::Point2f& pt, const std::vector<cv::Point2f>& ROI)
{
	// The "rectangle" isn't really a rectangle so we need to check that
	// the point is inside the rectangle with the maximum area that inside
	// the ROI polygon.
	return (pt.x > cv::max(ROI[0].x, ROI[3].x) && pt.x < cv::min(ROI[1].x, ROI[2].x)
		&&
		pt.y > cv::max(ROI[0].y, ROI[1].y) && pt.y < cv::min(ROI[2].y, ROI[3].y));
}

void mainWindowMouseCallback(int event, int x, int y, int, void* data)
{
	if (event == cv::EVENT_LBUTTONDBLCLK)
	{
		TMouseCallbackData* mouseCallbackData = (TMouseCallbackData*)(data);

		// Check if the mouse clicked inside ROI, and if so set the suitable
		// ROI as an active tracking window.
		
		const std::vector<std::vector<cv::Point2f>>& ROIs = *(mouseCallbackData->currROIs);
		std::vector<TFaceWindowParams>& windowsParams = *(mouseCallbackData->faceWindowsParams);

		cv::Point2i testPoint(x, y);

		for (int i = 0; i < windowsParams.size(); i++)
		{
			if (!windowsParams[i].active && !windowsParams[i].created)
			{
				if (isPointInsideROI(testPoint, ROIs[i]))
				{
					std::cout << "user choose to activate tracker no. " << i << " no ROI " << ROIs[i] << std::endl;
					windowsParams[i].active = true;

					cv::namedWindow(windowsParams[i].name, cv::WINDOW_AUTOSIZE);
					windowsParams[i].created = true;
				}
			}
		}
	}
}

bool getRigidTransformationMatrices(const std::vector<std::vector<cv::Point2f>>& currFeaturesGroups, const std::vector<std::vector<cv::Point2f>>& prevFeaturesGroups, const std::vector<std::vector<cv::Point2f>>& initROIs, const std::vector<std::vector<cv::Point2f>>& currROIs, std::vector<cv::Mat>& transMatrices, std::vector<cv::Mat>& transMatricesInv)
{

	if (currFeaturesGroups.size() != prevFeaturesGroups.size() || initROIs.size() != currROIs.size())
	{
		std::cout << "error - different number between the current number of features groups and the previous one" << std::endl;
		return false;
	}

	// Loop over all the features groups

	// No affine transformation, just rigid.
	const bool fullAffine = false;

	for (int i = 0; i < currFeaturesGroups.size(); i++)
	{
		const std::vector<cv::Point2f>& prevGroup = prevFeaturesGroups[i];
		const std::vector<cv::Point2f>& currGroup = currFeaturesGroups[i];
		const std::vector<cv::Point2f>& initROI = initROIs[i];
		const std::vector<cv::Point2f>& currROI = currROIs[i];

		// Estimate transformation matrix to the groups pair.

		if (prevGroup.size() != currGroup.size() || prevGroup.empty() || currGroup.empty())
		{
			std::cout << "warning - the something wrong with the sizes of the " << i << " features groups" << std::endl;

			cv::Mat emptyR;
			transMatrices[i] = emptyR;
		}
		else
		{
			cv::Mat R = cv::estimateRigidTransform(prevGroup, currGroup, fullAffine);
			transMatrices[i] = R;
		}

		if (initROI.size() != currROI.size() || initROI.empty() || currROI.empty())
		{
			std::cout << "warning - the something wrong with the sizes of the " << i << " features groups" << std::endl;

			cv::Mat emptyR;
			transMatricesInv[i] = emptyR;
		}
		else
		{
			cv::Mat R = cv::estimateRigidTransform(currROI, initROI, fullAffine);
			transMatricesInv[i] = R;
		}

	}

	return true;
}

void performRigidTransformationImg(const cv::Mat& M, cv::Mat& img)
{
	const int flag = 1;
	const int borderMode = 1;
	const cv::Scalar borderValue = cv::Scalar();
	if (!M.empty() && !img.empty())
	{
		cv::warpAffine(img, img, M, img.size(), flag, borderMode, borderValue);
	}
}

void performRigidTransformationOnROIs(const std::vector<cv::Mat>& transMatrices, std::vector<std::vector<cv::Point2f>>& ROIs, cv::Size& imgSize)
{
	if (transMatrices.size() != ROIs.size())
	{
		std::cout << "number of transformation matrices is not suitable for number of ROIs" << std::endl;
	}

	for (int i = 0; i < ROIs.size(); i++)
	{
		const cv::Mat& M = transMatrices[i];
		std::vector<cv::Point2f>& ROI = ROIs[i];

		if (!M.empty())
		{
			//cv::transform(ROI, ROI, M);
			std::vector<cv::Point2f> tmpResROI;
			cv::transform(ROI, tmpResROI, M);
			bool outOfImg = false;
			for (cv::Point2f pt : tmpResROI)
			{
				if (pt.x < 0 || pt.y < 0 || pt.x > imgSize.width || pt.y > imgSize.height)
				{
					outOfImg = true;
				}
			}
			if (!outOfImg)
			{
				ROI = tmpResROI;
			}
		}
	}
}

void convertRectToPts(const cv::Rect& rect, std::vector<cv::Point2f>& pts)
{
	// Take integers as floating point numbers.
	float x = (float)rect.x;
	float y = (float)rect.y;
	float w = (float)rect.width;
	float h = (float)rect.height;
	// Clear polygon and set it with the rectangle vertives.
	pts.clear();
	pts.push_back(cv::Point2f(x, y));
	pts.push_back(cv::Point2f(x + w, y));
	pts.push_back(cv::Point2f(x + w, y + h));
	pts.push_back(cv::Point2f(x, y + h));
}

void drawPoly(const std::vector<cv::Point2f>& pts, cv::Mat& img)
{
	// Drawing parameters.
	const bool isClosed = true;
	const cv::Scalar color(255, 255, 255);
	const int thickness = 1;
	const int lineType = cv::LINE_8;
	const int shift = 0;

	// Convert the floating point vertices to integer vertices.
	std::vector<cv::Point2i> ptsi;
	for (cv::Point2f ptf : pts)
	{
		ptsi.push_back(cv::Point2i((int)ptf.x, (int)ptf.y));
	}

	// Draw the polygon.
	cv::polylines(img, ptsi, isClosed, color, thickness, lineType, shift);
}

void drawROIs(const std::vector<std::vector<cv::Point2f>>& ROIs, cv::Mat& img)
{
	// Move over all the ROIs and draw each one.
	for (const std::vector<cv::Point2f>& ROI : ROIs)
	{
		drawPoly(ROI, img);
	}
}

bool calcLKOpticalFlowForAllTheFeaturesGroups(const std::vector<cv::Mat>& prevPyr, const std::vector<cv::Mat>& currPyr, const std::vector<std::vector<cv::Point2f>>& prevFeaturesGroups, std::vector<std::vector<cv::Point2f>>& currFeaturesGroups)
{
	if (prevFeaturesGroups.empty())
	{
		std::cout << "no features groups to move from in optical flow" << std::endl;
		return false;
	}

	// Clear current features groups.
	currFeaturesGroups.clear();

	// unite all the features groups for more efficient optical flow.
	std::vector<cv::Point2f> fullPrevGroup;
	for (const std::vector<cv::Point2f>& prevGroup : prevFeaturesGroups)
	{
		fullPrevGroup.insert(fullPrevGroup.end(), prevGroup.begin(), prevGroup.end());
	}

	// now its time for bi-directional optical flow estimation
	// parameters
	const cv::Size optFlowWinSize = cv::Size(21, 21);
	const int maxLevel = 3;
	std::vector<cv::Point2f> currCorners, fullPrevGroupInv;
	std::vector<float> firstSideErr, secondSideErr;
	std::vector<uchar> firstSideStatus, secondSideStatus;
	cv::TermCriteria optFlowCriteria = cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 30, 0.01);
	const int flags = 0;
	const double minEigThreshold = 1e-4;
	// optical flow calculation
	// first the calculation will be the previous image agains the current one
	// and then the oposite, but with the new estimated points already taken from
	// the first calculation
	calcOpticalFlowPyrLK(
		prevPyr, currPyr, fullPrevGroup, currCorners, firstSideStatus, firstSideErr,
		optFlowWinSize, maxLevel, optFlowCriteria, flags, minEigThreshold);
	calcOpticalFlowPyrLK(
		currPyr, prevPyr, currCorners, fullPrevGroupInv, secondSideStatus, secondSideErr,
		optFlowWinSize, maxLevel, optFlowCriteria, flags, minEigThreshold);
	// performing comparison between the two sets of previous corners
	// in order to take the "error free" current ones.
	// Also don't take features outside the face ROI
	std::vector<cv::Point2f> finalCorners;
	for (int i = 0; i < currCorners.size(); i++)
	{
		cv::Point2f diff = fullPrevGroupInv[i] - fullPrevGroup[i];
		if (firstSideStatus[i] && secondSideStatus[i] && diff.x <= 0.5 && diff.y <= 0.5)
		{
			finalCorners.push_back(currCorners[i]);
		}
		else
		{
			finalCorners.push_back(fullPrevGroup[i]);
		}
	}

	size_t currOff = 0;
	for (int i = 0; i < prevFeaturesGroups.size(); i++)
	{
		std::vector<cv::Point2f> currGroup;
		currGroup.insert(currGroup.begin(), finalCorners.begin() + currOff, finalCorners.begin() + currOff + prevFeaturesGroups[i].size());
		currFeaturesGroups.push_back(currGroup);
		currOff += prevFeaturesGroups[i].size();
	}

	return true;
}



bool buildLKPyr(const cv::Mat& grayscaleImg, std::vector<cv::Mat>& pyr)
{
	// Cant build pyramid from empty image.
	if (grayscaleImg.empty())
	{
		std::cout << "can't build pyramid from empty image" << std::endl;
		return false;
	}

	// Pyramid builder parameters.
	const cv::Size winSize = cv::Size(21, 21);
	const int maxLevel = 3;
	const bool withDerivatives = true;
	const int pyrBorder = cv::BORDER_REFLECT_101;
	const int derivBorder = cv::BORDER_CONSTANT;
	const bool tryReuseInputImage = true;

	// Clear given pyramid.
	pyr.clear();

	// Build new pyramid.
	buildOpticalFlowPyramid(
		grayscaleImg, pyr, winSize, maxLevel,
		withDerivatives, pyrBorder, derivBorder, tryReuseInputImage);

	return true;
}


bool acquireFrameFromBuffer(cv::Mat& outputFrame)
{
	if (!framesQueue.empty())
	{
		// Lock the queue, sample the frame and remove it from the queue.
		samplerLock.lock();
		outputFrame = framesQueue.front();
		framesQueue.pop();
		// Release the queue.
		samplerLock.unlock();

		if (!outputFrame.empty())
		{
			// Sampling succeeded.
			return true;
		}
	}

	// Sampling failed - the queue is empty.
	return false;
}


void framesSamplerThreadFunction()
{

	// Initialize the video stream and check its validity.
	cv::VideoCapture cap;
	if (trackerConfigurations.isWebcam)
	{
		std::cout << "video capture opened with resource " << trackerConfigurations.resource << std::endl;
		cap.open(trackerConfigurations.resource);
	}
	else
	{
		std::cout << "video capture opened with video file " << trackerConfigurations.videoPath << std::endl;
		cap.open(trackerConfigurations.videoPath);
	}


	if (!cap.isOpened())
	{
		std::cout << "sampling thread failed to open video capture" << std::endl;
		return;
	}

	cv::Mat frame;
	while (programRunning)
	{
		// Get the frame and push it to the buffer.
		cap >> frame;
		samplerLock.lock();
		framesQueue.push(frame);
		samplerLock.unlock();
	}

	// Release resources when no more sampling needed.
	cap.release();
	std::cout << "sampling thread ended" << std::endl;
}

cv::Rect getTranslatedROI(const cv::Rect& srcROI, const cv::Rect& containerROI)
{
	// Convert source ROI relatively to its container.
	return cv::Rect(
		containerROI.x + srcROI.x,
		containerROI.y + srcROI.y,
		srcROI.width,
		srcROI.height);
}


bool detectFacialROIs(const cv::Mat& grayscaleImg, std::vector<TFacialROIs>& facialROIsVector)
{
	// Reset the ROIs.

	facialROIsVector.clear();

	// Classifiers initialization.
	static bool alreadyLoaded = false;

	if (!alreadyLoaded)
	{

		if (!faceClassifier.load(trackerConfigurations.faceHaarFeaturesPath))
		{
			std::cout << "failed to load cascade classifier with file " << trackerConfigurations.faceHaarFeaturesPath << std::endl;
			return false;
		}
		if (!eyeClassifier.load(trackerConfigurations.eyeHaarFeaturesPath))
		{
			std::cout << "failed to load cascade classifier with file " << trackerConfigurations.eyeHaarFeaturesPath << std::endl;
			return false;
		}
		if (!noseClassifier.load(trackerConfigurations.noseHaarFeaturesPath))
		{
			std::cout << "failed to load cascade classifier with file " << trackerConfigurations.noseHaarFeaturesPath << std::endl;
			return false;
		}
		if (!mouthClassifier.load(trackerConfigurations.mouthHaarFeaturesPath))
		{
			std::cout << "failed to load cascade classifier with file " << trackerConfigurations.mouthHaarFeaturesPath << std::endl;
			return false;
		}
	}

	// Detect faces.

	std::vector<cv::Rect> facesROIs;

	const double scaleFactor = 1.1;
	const int minNeighbors = 3;
	const int flags = 0;
	const cv::Size minSize = cv::Size();
	const cv::Size maxSize = cv::Size();
	facesROIs.clear();
	faceClassifier.detectMultiScale(
		grayscaleImg, facesROIs,
		scaleFactor, minNeighbors, flags, minSize, maxSize);

	if (facesROIs.empty())
	{
		std::cout << "no faces detected in image" << std::endl;
		return false;
	}

	const double percentsOfReduction = 30.0;

	for (const cv::Rect& faceROI : facesROIs)
	{
		// Detect eyes, nose & mouth inside the current ROI.

		TFacialROIs facialROIs;

		facialROIs.face = faceROI;

		cv::Mat ROIImg = grayscaleImg(faceROI);
		cv::Rect eyesROI(faceROI.x, faceROI.y, faceROI.width, faceROI.height / 2);
		cv::Rect mouthROI(faceROI.x, faceROI.y + faceROI.height / 2, faceROI.width, faceROI.height / 2);
		cv::Mat lowerHalfOfTheFace = grayscaleImg(mouthROI);
		cv::Mat higherHalfOfTheFace = grayscaleImg(eyesROI);

		// --- Eyes

		std::vector<cv::Rect> eyesROIs;
		eyeClassifier.detectMultiScale(
			higherHalfOfTheFace, eyesROIs,
			scaleFactor, minNeighbors, flags, minSize, maxSize);
		if (eyesROIs.size() == 1)
		{
			facialROIs.eyes.push_back(getReducedROI(getTranslatedROI(eyesROIs[0], eyesROI), percentsOfReduction));
		}
		else if (eyesROIs.size() == 2)
		{
			facialROIs.eyes.push_back(getReducedROI(getTranslatedROI(eyesROIs[0], eyesROI), percentsOfReduction));
			facialROIs.eyes.push_back(getReducedROI(getTranslatedROI(eyesROIs[1], eyesROI), percentsOfReduction));
		}
		else
		{
			//std::cout << "classifier failed in eye detection : " << eyesROIs.size() << " eyes found inside " << faceROI << std::endl;
			//return false;
			facialROIs.eyes.push_back(getReducedROI(eyesROI, percentsOfReduction));
		}

		// --- Nose
		std::vector<cv::Rect> noseROIs;
		noseClassifier.detectMultiScale(
			ROIImg, noseROIs,
			scaleFactor, minNeighbors, flags, minSize, maxSize);
		if (noseROIs.size() == 1)
		{
			facialROIs.nose = getReducedROI(getTranslatedROI(noseROIs[0], faceROI), percentsOfReduction);
		}
		else
		{
			//std::cout << "classifier failed in nose detection : " << noseROIs.size() << " noses found inside " << faceROI << std::endl;
			//return false;
			facialROIs.nose = getReducedROI(faceROI, percentsOfReduction * 2.0);
		}

		// --- Mouth
		std::vector<cv::Rect> mouthROIs;
		mouthClassifier.detectMultiScale(
			lowerHalfOfTheFace, mouthROIs,
			scaleFactor, minNeighbors, flags, minSize, maxSize);
		if (mouthROIs.size() == 1)
		{
			facialROIs.mouth = getReducedROI(getTranslatedROI(mouthROIs[0], mouthROI), percentsOfReduction);
		}
		else
		{
			//std::cout << "classifier failed in mouth detection : " << mouthROIs.size() << " mouthes found inside " << faceROI << std::endl;
			//return false;
			facialROIs.mouth = getReducedROI(mouthROI, percentsOfReduction);
		}

		// Saving record.
		facialROIsVector.push_back(facialROIs);
	}

	std::cout << "facial features fully detected successfully for at least one face in the image" << std::endl;
	return true;
}

void convertRectROIsToPolys(const std::vector<TFacialROIs>& facialROIsVector, std::vector<std::vector<cv::Point2f>>& ROIs)
{
	ROIs.clear();

	for (const TFacialROIs& facialROIs : facialROIsVector)
	{
		std::vector<cv::Point2f> ROI;
		convertRectToPts(facialROIs.face, ROI);
		ROIs.push_back(ROI);
	}
}

void drawFacialROIsVector(const std::vector<TFacialROIs>& facialROIsVector, cv::Mat& img)
{
	const cv::Scalar color(255, 255, 255);
	const int thickness = 1;
	const int lineType = cv::LINE_8; 
	const int shift = 0;

	for (const TFacialROIs& facialROIs : facialROIsVector)
	{
		cv::rectangle(img, facialROIs.face, color, thickness, lineType, shift);
		for (const cv::Rect& eyeROI: facialROIs.eyes)
		{
			cv::rectangle(img, eyeROI, color, thickness, lineType, shift);
		}
		cv::rectangle(img, facialROIs.nose, color, thickness, lineType, shift);
		cv::rectangle(img, facialROIs.mouth, color, thickness, lineType, shift);
	}
}

void drawFacialFeaturesGroups(const std::vector<std::vector<cv::Point2f>>& facialFeaturesGroups, cv::Mat& img)
{
	const int radius = 2;
	const cv::Scalar color(0, 255, 0);
	const int thickness = 1;
	const int lineType = cv::LINE_8;
	const int shift = 0;

	for (const std::vector<cv::Point2f>& group : facialFeaturesGroups)
	{
		for (const cv::Point2f& pt : group)
		{
			cv::circle(img, pt, radius, color, thickness, lineType, shift);
		}
	}
}


cv::Rect getReducedROI(const cv::Rect& srcROI, double percents)
{
	if (percents <= 0 || percents >= 100)
	{
		return srcROI;
	}

	double dx = srcROI.x, dy = srcROI.y;
	double dw = srcROI.width, dh = srcROI.height;

	double wred = (percents * dw / 100.0) * 0.5;
	double hred = (percents * dh / 100.0) * 0.5;

	return cv::Rect(
		(int)(dx + wred),
		(int)(dy + hred),
		(int)(dw - 2 * wred),
		(int)(dh - 2 * hred));
}

cv::Rect getEnlargedROI(const cv::Rect& srcROI, double percents)
{
	if (percents <= 0 || percents >= 100)
	{
		return srcROI;
	}

	double dx = srcROI.x, dy = srcROI.y;
	double dw = srcROI.width, dh = srcROI.height;

	double wred = (percents * dw / 100.0) * 0.5;
	double hred = (percents * dh / 100.0) * 0.5;

	return cv::Rect(
		(int)(dx - wred),
		(int)(dy - hred),
		(int)(dw + 2 * wred),
		(int)(dh + 2 * hred));
}

bool findFeaturesInsideInfaceROIs(const cv::Mat& grayscaleImg, const std::vector<TFacialROIs>& facialROIsVector, std::vector<std::vector<cv::Point2f>>& featuresGroups)
{
	// Reset the features groups.

	featuresGroups.clear();

	// Loop over the facial ROIs vector.

	// Features detector parameters :
	const int maxCorners = MAX_CORNERS_TO_DETECT_INSIDE_ONE_ROI;
	const double qualityLevel = 0.01;
	const double minDistance = 10;
	const int blockSize = 3;
	const bool useHarrisDetector = false;
	const double k = 0.04;

	// Corners sub pixels parameters :
	cv::Size winSize = cv::Size(5, 5);
	cv::Size zeroZone = cv::Size(-1, -1);
	cv::TermCriteria criteria = cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 40, 0.001);

	for (const TFacialROIs& facialROIs : facialROIsVector)
	{
		// Find good features by Shi-Tomasi inside the eyes, nose and mouth ROIs.

		// Preparing mask for the facial ROIs.
		cv::Mat mask = cv::Mat::zeros(grayscaleImg.size(), CV_8UC1);
		for (const cv::Rect& eyeROI : facialROIs.eyes)
		{
			mask(eyeROI) = 255;
		}
		mask(facialROIs.nose) = 255;
		mask(facialROIs.mouth) = 255;

		// Detect the features.
		std::vector<cv::Point2f> featuresGroup;
		cv::goodFeaturesToTrack(grayscaleImg, featuresGroup, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);	

		if (featuresGroup.size() <= 0)
		{
			std::cout << "the features detector failed to detect features inside " << facialROIs.face << std::endl;
			return false;
		}

		// Subpixeling
		cv::cornerSubPix(grayscaleImg, featuresGroup, winSize, zeroZone, criteria);

		featuresGroups.push_back(featuresGroup);
	
	}

	std::cout << "features detection succeeded" << std::endl;
	return true;
}

void faceThreadFunction(int queueIndex)
{
	// Take the appropriate queue.
	std::queue<TFaceWindowThreadParams>& queue = faceThreadsBuffers[queueIndex];

	int delay = getMainLoopDelayByVideoFPS();

	cv::VideoWriter outputVideo;
	bool videoWriterInitialized = false;

	while (programRunning)
	{

		if (faceWindowsParams[queueIndex].active && faceWindowsParams[queueIndex].created)
		{
			if (!queue.empty())
			{
				

				mtxs[queueIndex].lock();
				TFaceWindowThreadParams& fwtp = queue.front(); // Copy the fields.
				mtxs[queueIndex].unlock();

				// Perform the inverse transformation and present in the window only the ROI part.
				performRigidTransformationImg(fwtp.inv, fwtp.img);
				cv::imshow(faceWindowsParams[queueIndex].name, fwtp.img(currFacialROIsVector[queueIndex].face));

				if (!videoWriterInitialized)
				{
					// Initialize the video writer.
					if (trackerConfigurations.isRecord)
					{
						std::string baseName = trackerConfigurations.outputVideoName;
						std::string outputVideoPath = baseName.substr(0, baseName.find_last_of(".")) + "-" + faceWindowsParams[queueIndex].name + ".avi";
						outputVideo.open(outputVideoPath, CV_FOURCC('D', 'I', 'V', 'X'), trackerConfigurations.fps, currFacialROIsVector[queueIndex].face.size(), true);
						if (!outputVideo.isOpened())
						{
							std::cout << "Could not open the output video for write: " << outputVideoPath << std::endl;
							programRunning = false;
							break;
						}
						videoWriterInitialized = true;
					}
				}

				// If recording write the current frame to the file.
				if (videoWriterInitialized)
				{
					outputVideo << fwtp.img(currFacialROIsVector[queueIndex].face);
				}

				queue.pop(); // Remove the task from the queue.
			}	
		}

		// Work in the pace of the video FPS.
		int key = cv::waitKey(delay);
		if (key == EXIT_KEY_CODE)
		{
			std::cout << "user stoped main loop" << std::endl;
			programRunning = false;
			break;
		}
	}
}

int getMainLoopDelayByVideoFPS()
{
	int fps = trackerConfigurations.fps;
	if (!trackerConfigurations.isWebcam)
	{
		fps = 1000 / fps;
	}
	return fps;
}