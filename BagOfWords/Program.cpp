#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <math.h>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace xfeatures2d;

const string DATASET_PATH = "...\\AplicatieProiect\\Semnaturi\\"; // Imagine directory
const string IMAGE_EXT = ".jpg"; // Extensie
const int TESTING_PERCENT_PER = 20; // Imagine de test
const int DICTIONARY_SIZE = 230;	// 80 word per class
int allDescPerImgNum = 0;

Mat allDescriptors, inputData, inputDataLables, kCenters, kLabels;
Ptr<SVM> svm;
vector<Mat> allDescPerImg;
vector<int> allClassPerImg;

// Histogram display function
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	// Computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i < hist_cols; i++)
		if (hist[i] > max_hist)
			max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(0, 0, 0)); // Histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat canny(Mat src)
{
	/*
	Pasii metodei Canny:
	1. Filtrarea imaginii cu un filtru Gaussian pentru eliminarea zgomotelor
	2. Calculul modulului si directiei gradientului
	3. Suprimarea non-maximelor modulului gradientului
	~4. Binarizarea adaptiva a punctelor de muchie si prelungirea muchiilor prin histereza
	*/

	//imshow("Image", src); // Afisare imagine

	int height = src.rows;
	int width = src.cols;

	// Prewitt
	int Px[3][3] = { { -1, 0, 1 }, { -1, 0, 1 }, { -1,  0,  1 } };
	int Py[3][3] = { {  1, 1, 1 }, {  0, 0, 0 }, { -1, -1, -1 } };

	// Sobel [Try This One!]
	int Sx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1,  0,  1 } };
	int Sy[3][3] = { {  1, 2, 1 }, {  0, 0, 0 }, { -1, -2, -1 } };

	// Roberts (cross)
	int Rx[2][2] = { { 1,  0 }, { 0, -1 } };
	int Ry[2][2] = { { 0, -1 }, { 1,  0 } };

	Mat srcPart = src.clone();
	std::vector<uchar> v;
	int w = 3; // 3, 5 sau 7

	// Filtrarea imaginii cu un filtru Gaussian pentru eliminarea zgomotelor
	for (int i = w / 2; i < height - w / 2; i++)
		for (int j = w / 2; j < width - w / 2; j++)
		{
			int Gx = 0;
			int Gy = 0;
			for (int k = -w / 2; k <= w / 2; k++)
				for (int l = -w / 2; l <= w / 2; l++) {
					Gx += src.at<uchar>(i + k, j + l) * Sx[k + 1][l + 1];
					Gy += src.at<uchar>(i + k, j + l) * Sy[k + 1][l + 1];
				}

			// Calculul modulului si directiei gradientului
			float gradient = sqrt(Gx * Gx + Gy * Gy) / (4 * sqrt(2)); // Gradient
			float directia = atan2(Gy, Gx); // Directia

			srcPart.at<uchar>(i, j) = (uchar)gradient;
			v.clear();
		}


	Mat srcRes = srcPart.clone();

	// Suprimarea non-maximelor modulului gradientului
	for (int i = w / 2; i < height - w / 2; i++)
	{
		for (int j = w / 2; j < width - w / 2; j++)
		{
			int Gx = 0;
			int Gy = 0;
			for (int k = -w / 2; k <= w / 2; k++)
				for (int l = -w / 2; l <= w / 2; l++) {
					Gx += src.at<uchar>(i + k, j + l) * Sx[k + 1][l + 1];
					Gy += src.at<uchar>(i + k, j + l) * Sy[k + 1][l + 1];
				}

			// Calculul modulului si directiei gradientului
			float gradient = sqrt(Gx * Gx + Gy * Gy) / (4 * sqrt(2)); // Gradient
			float directia = atan2(Gy, Gx); // Directia

			// Cuantificarea directiilor gradientului
			if ((directia<CV_PI / 8 && directia> -CV_PI / 8) || (directia < CV_PI && directia > 7 * CV_PI / 8) || (directia > -CV_PI && directia < -7 * CV_PI / 8)) // 0 (90` 270`)
			{
				if (srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i, j + 1) && srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i, j - 1))
					srcRes.at<uchar>(i, j) = srcPart.at<uchar>(i, j);
				else
					srcRes.at<uchar>(i, j) = 0;
			}

			if ((directia > CV_PI / 8 && directia < 3 * CV_PI / 8) || (directia > -7 * CV_PI / 8 && directia < -5 * CV_PI / 8)) // 1 (45` 225`)
			{
				if (srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i - 1, j + 1) && srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i + 1, j - 1))
					srcRes.at<uchar>(i, j) = srcPart.at<uchar>(i, j);
				else
					srcRes.at<uchar>(i, j) = 0;
			}

			if ((directia > 3 * CV_PI / 8 && directia < 5 * CV_PI / 8) || (directia<-3 * CV_PI / 8 && directia>-5 * CV_PI / 8)) // 2 (0` 180`)
			{
				if (srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i - 1, j) && srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i + 1, j))
					srcRes.at<uchar>(i, j) = srcPart.at<uchar>(i, j);
				else
					srcRes.at<uchar>(i, j) = 0;
			}

			if ((directia > 5 * CV_PI / 8 && directia < 7 * CV_PI / 8) || (directia<-CV_PI / 8 && directia>-3 * CV_PI / 8)) // 3 (135` 315`)
			{
				if (srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i + 1, j + 1) && srcPart.at<uchar>(i, j) >= srcPart.at<uchar>(i - 1, j - 1))
					srcRes.at<uchar>(i, j) = srcPart.at<uchar>(i, j);
				else
					srcRes.at<uchar>(i, j) = 0;
			}

		}
	}

	//imshow("Partial Canny", srcPart);
	//imshow("Final Canny", srcRes);

	//waitKey();
	return srcRes;
}

Mat binAdaptiv(Mat image)
{
	/*
	Pasii metodei Canny:
	1,2,3. Filtrarea imaginii, calculul modulului si directiei gradientului, suprimarea non-maximelor modulului gradientului
	4. Binarizarea adaptiva a punctelor de muchie si prelungirea muchiilor prin histereza
	*/

	Mat src = canny(image);
	int height = src.rows;
	int width = src.cols;

	int histValues[256] = { 0 };
	int numarPixeliCuModulGradientNenul = 0;

	for (int i = 1; i < height - 1; i++)
		for (int j = 1; j < width - 1; j++)
		{
			histValues[src.at<uchar>(i, j)]++;
			if (src.at<uchar>(i, j) != 0)
				numarPixeliCuModulGradientNenul++;
		}

	float p = 0.1; // Valori intre 0.01 - 0.1
	int nrPuncteMuchie = p * numarPixeliCuModulGradientNenul;
	//printf("Numar puncte muchie: %d\n", nrPuncteMuchie);

	int PragAdaptiv = 0;
	for (int i = 255; i >= 0; i--)
	{
		if (histValues[i] != 0)
			nrPuncteMuchie -= histValues[i];

		if (nrPuncteMuchie <= 0)
		{
			PragAdaptiv = i;
			i = 0;
		}
	}

	int Prag_inalt = PragAdaptiv;
	int Prag_coborat = 0.4 * PragAdaptiv;
	//printf("Prag inalt: %d\nPrag coborat: %d\n", Prag_inalt, Prag_coborat);

	for (int i = 1; i < height - 1; i++)
	{
		for (int j = 1; j < width - 1; j++)
		{
			if (src.at<uchar>(i, j) <= Prag_coborat)
				src.at<uchar>(i, j) = 0;
			else
			{
				if (src.at<uchar>(i, j) >= Prag_coborat && src.at<uchar>(i, j) <= Prag_inalt)
					src.at<uchar>(i, j) = 127;
				else
				{
					if (src.at<uchar>(i, j) >= Prag_inalt)
						src.at<uchar>(i, j) = 255;
				}
			}
		}
	}

	//imshow("PragAdaptiv", src);

	bool stop = false;
	while (!stop)
	{
		stop = true;
		for (int i = 1; i < height - 1; i++)
			for (int j = 1; j < width - 1; j++)
				if (src.at<uchar>(i, j) == 255)
					for (int k = -1; k <= 1; k++)
						for (int l = -1; l <= 1; l++)
							if (src.at<uchar>(i + k, j + l) == 127)
							{
								src.at<uchar>(i + k, j + l) = 255;
								stop = false;
							}
	}

	for (int i = 1; i < height - 1; i++)
		for (int j = 1; j < width - 1; j++)
			if (src.at<uchar>(i, j) == 127)
				src.at<uchar>(i, j) = 0;

	//imshow("Final PragAdaptiv", src);

	//waitKey();

	return src;
}

Mat conversieRGBtoGrayscale(Mat src)
{

	Mat gray(src.rows, src.cols, CV_8UC1); // Create image

	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			gray.at<uchar>(i, j) = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[2]) / 3;
		}
	}

	return gray;
}

Mat conversieGrayscaleToBinary(Mat src, int threshold)
{
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) < threshold)
				src.at<uchar>(i, j) = 0;
			else
				src.at<uchar>(i, j) = 255;
		}
	}

	return src;
}

float histogramaPixeli(Mat src, const int imp, int hist_vC[], int hist_oC[])
{
	int height = src.rows;
	int width = src.cols;
	//printf("%d, %d", height, width);

	int hist_verticala[1000] = { 0 };
	int hist_orizontala[1000] = { 0 };

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (src.at<uchar>(i, j) == 0)
			{
				hist_verticala[j / (int)(256 / imp)] ++;
				hist_orizontala[i / (int)(256 / imp)] ++;
			}
		}
	}

	float total_vert = 0.0f;
	for (int i = 0; i < width / (int)(256 / imp); ++i)
	{
		float procent = (float) hist_verticala[i] / hist_vC[i];
		if (procent > 1.00f)
			procent = (float) hist_vC[i] / hist_verticala[i];
		total_vert += procent;
		//printf("Vert: %f\n", procent);
	}
	total_vert = (total_vert / (width / (int)(256 / imp)) ) * 100;
	//printf("%f\n", total_vert);

	
	float total_oriz = 0.0f;
	for (int i = 0; i < height / (int)(256 / imp); ++i)
	{
		float procent = hist_orizontala[i] / hist_oC[i];
		if (procent > 1.00f)
			procent = (float)hist_oC[i] / hist_orizontala[i];
		total_oriz += procent;
	}
	total_oriz = (total_oriz / (height / (int)(256 / imp))) * 100;
	//printf("%f\n", total_oriz);
	
	float total = (total_vert + total_oriz) / 2;
	//printf("Total: %f\n", total);
	
	//showHistogram("Verticala", hist_verticala, width / (int)(256 / imp), 200); // Afisare histograma
	//showHistogram("Orizontala", hist_orizontala, height / (int)(256 / imp), 200); // Afisare histograma

	return total;

}

Mat resizeImage(Mat src, int size)
{
	Mat dst1, dst2;
	//without interpolation
	//resizeImg(src, dst1, size, false);

	cv::resize(src, dst2, cv::Size(300,200));
	//cv::resize(src, dst2, cv::Size(), (float) size / src.cols, (float) size / src.cols);
	//with interpolation
	//resizeImg(src, dst2, size, true);
	//imshow("Resize image (with interpolation)", dst2);

	return dst2;
	//imshow("Resize Image (without interpolation)", dst1);
}



bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void readGetImageDescriptors(const string& className, int numberOfImages, int classLable) {

	for (int i = 1; i <= numberOfImages; i++) {
		// If the image is chosen as a test image, do not use it in training
		//if (i % TESTING_PERCENT_PER == 0)
		//	continue;

		Ptr<SIFT> sift;
		sift = SIFT::create();

		string imageNumber = to_string(i);
		while (imageNumber.length() < 4)
			imageNumber = '0' + imageNumber;

		// Read image, detect and compute features
		if (fileExists(DATASET_PATH + className + "\\image_" + imageNumber + IMAGE_EXT)) 
		{
			Mat grayImg, descriptors;
			Mat src = imread(DATASET_PATH + className + "\\image_" + imageNumber + IMAGE_EXT, IMREAD_GRAYSCALE);

			int height = src.rows;
			int width = src.cols;

			//Mat srcGray(height, width, CV_8UC1); // Create gray image
			//srcGray = conversieRGBtoGrayscale(src); // RGB to Grayscale conversion

			Mat srcBinary(height, width, CV_8UC1); // Create gray image
			srcBinary = conversieGrayscaleToBinary(src, 100); // Grayscale to Binary conversion

			// Crop Image
			int minR = srcBinary.rows, maxR = 0;
			int minC = srcBinary.cols, maxC = 0;

			for (int r = 0; r < srcBinary.rows; r++) {
				for (int c = 0; c < srcBinary.cols; c++) {
					if (srcBinary.at<uchar>(r, c) == 0)
					{
						if (r < minR)
							minR = r;
						if (r > maxR)
							maxR = r;
						if (c < minC)
							minC = c;
						if (c > maxC)
							maxC = c;
					}
				}
			}

			Mat cropImage(maxR - minR, maxC - minC, CV_8UC1);
			for (int r = 0; r < maxR - minR; r++)
				for (int c = 0; c < maxC - minC; c++)
					cropImage.at<uchar>(r, c) = srcBinary.at<uchar>(r + minR, c + minC);

			//histogramaPixeli(cropImage, 32);

			Mat resizedImage = resizeImage(cropImage, 500);

			//imshow("Image", src); // Afisare imagine
			//imshow("Crop", cropImage); // Afisare imagine
			//imshow("Photo", resizedImage);

			grayImg = binAdaptiv(resizedImage);

			vector<KeyPoint> keypoints;

			sift->detectAndCompute(grayImg, noArray(), keypoints, descriptors);

			allDescriptors.push_back(descriptors);
			allDescPerImg.push_back(descriptors);
			allClassPerImg.push_back(classLable);
			allDescPerImgNum++;

			Mat outImg;
			drawKeypoints(grayImg, keypoints, outImg, Scalar::all(-1), DrawMatchesFlags::DEFAULT); //DrawMatchesFlags::DEFAULT //DrawMatchesFlags::DRAW_RICH_KEYPOINTS
			//imshow("Image " + className + "[" + imageNumber + "]", outImg);
			//waitKey(0);
		}
		else
			break;
	}
}

Mat getDataMat(Mat descriptors) {
	BFMatcher matcher;
	vector<DMatch> matches;

	// descriptors => query descriptors
	// kCenters => train descriptors
	// matches => add descriptors in this vector if kCenters match with descriptors variable
	matcher.match(descriptors, kCenters, matches);

	// Make a Histogram of Visual Words
	Mat datai(1, DICTIONARY_SIZE, CV_32FC1);
	int index = 0;
	for (vector<DMatch>::iterator j = matches.begin(); j < matches.end(); j++)
	{
		datai.at<float>(0, matches.at(index).trainIdx)++;
		index++;
	}
	return datai;
}

void getHistogram(const string& className, int numberOfImages, int classLable) {

	for (int i = 1; i <= numberOfImages; i++) {
		// If the image is chosen as a test image, do not use it in training
		if (i % TESTING_PERCENT_PER == 0)
			continue;

		Ptr<SIFT> sift;
		sift = SIFT::create();

		string imageNumber = to_string(i);
		while (imageNumber.length() < 4)
			imageNumber = '0' + imageNumber;

		// Read image, detect and compute features, create histogram Mats
		if (fileExists(DATASET_PATH + className + "\\image_" + imageNumber + IMAGE_EXT)) {
			Mat grayImg, descriptors;
			grayImg = imread(DATASET_PATH + className + "\\image_" + imageNumber + IMAGE_EXT, IMREAD_GRAYSCALE);
			vector<KeyPoint> keypoints;
			sift->detectAndCompute(grayImg, noArray(), keypoints, descriptors);

			inputData.push_back(getDataMat(descriptors));
			inputDataLables.push_back(Mat(1, 1, CV_32SC1, classLable));
		}
		else
			break;
	}
}

void getHistogramFast() {

	for (int i = 0; i < allDescPerImgNum; i++) 
	{
		Mat dvec = getDataMat(allDescPerImg[i]);

		inputData.push_back(dvec);
		inputDataLables.push_back(Mat(1, 1, CV_32SC1, allClassPerImg[i]));
	}
}

double testData(const string& className, int numberOfImages, int classLable) {
	int allTests = 0;
	int correctTests = 0;

	for (int i = TESTING_PERCENT_PER; i <= numberOfImages; i += TESTING_PERCENT_PER) {

		Ptr<SIFT> sift;
		float r = 0;
		sift = SIFT::create();

		string imageNumber = to_string(i);
		while (imageNumber.length() < 4)
			imageNumber = '0' + imageNumber;

		// Load image, Detect and Describe features
		if (fileExists(DATASET_PATH + className + "\\image_" + imageNumber + IMAGE_EXT))
		{
			Mat grayImg, descriptors;
			grayImg = imread(DATASET_PATH + className + "\\image_" + imageNumber + IMAGE_EXT, IMREAD_GRAYSCALE);
			vector<KeyPoint> keypoints;

			sift->detectAndCompute(grayImg, noArray(), keypoints, descriptors);
			Mat dvector = getDataMat(descriptors);

			allTests++;
			// svm->predict is used to classify an input sample using a trained SVM
			if (svm->predict(dvector) == classLable)
				correctTests++;
		}
		else
			break;
	}
	return (double)correctTests / allTests;
}

void algoritmKmeans()
{
	clock_t sTime = clock();
	printf("Citire imagini de test...\n");

	readGetImageDescriptors("test1", 10, 1);
	readGetImageDescriptors("test2", 10, 2);
	readGetImageDescriptors("test3", 12, 3);

	printf("-> Citire, Detectie si Transformare imagini de test in %.3f secunde.\n", (clock() - sTime) / double(CLOCKS_PER_SEC));

	
	int clusterCount = DICTIONARY_SIZE, attempts = 5, iterationNumber = 1e4;
	sTime = clock();
	printf("Rulare algoritm kmeans...\n");

	// kCenter is the output matrix of the cluster centers, one row per each cluster center
	// kLabels is the input/output integer array that stores the cluster indices for every sample
	kmeans(allDescriptors, clusterCount, kLabels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, iterationNumber, 1e-4), attempts, KMEANS_PP_CENTERS, kCenters); //KMEANS_RANDOM_CENTERS, KMEANS_USE_INITIAL_LABELS
	printf("-> Kmeans rulat in %.3f secunde.\n", (clock() - sTime) / double(CLOCKS_PER_SEC));

	
	sTime = clock();
	printf("Generare histograme...\n");
	getHistogramFast();
	printf("-> Histograme realizate in %.3f secunde\n", (clock() - sTime) / double(CLOCKS_PER_SEC));

	sTime = clock();
	printf("Antrenare SVM...\n");

	// Set up SVM's parameters
	svm = SVM::create();
	svm->setType(SVM::C_SVC);
	svm->setKernel(SVM::LINEAR);
	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));

	// Train the SVM with given parameters
	svm->train(inputData, ROW_SAMPLE, inputDataLables);
	printf("-> SVM antrenata in %.3f secunde\n", (clock() - sTime) / double(CLOCKS_PER_SEC));

	sTime = clock();
	printf("Testare imagini...\n");
	printf("-> %.0f%c acuratete in semnatura din categoria 'test1'.\n", (float)(testData("input", 22, 1) * 100), '%');
	printf("-> %.0f%c acuratete in semnatura  din categoria 'test2'.\n", (float)(testData("input", 22, 2) * 100), '%');
	printf("-> %.0f%c acuratete in semnatura  din categoria 'test3'.\n", (float)(testData("input", 22, 3) * 100), '%');

	printf("-> Test realizat in %.3f secunde", (clock() - sTime) / double(CLOCKS_PER_SEC));
	
	cin.get();
}

Mat processedImage(string cale)
{
	Mat src = imread(cale, IMREAD_GRAYSCALE);

	int height = src.rows;
	int width = src.cols;

	//Mat srcGray(height, width, CV_8UC1); // Create gray image
	//srcGray = conversieRGBtoGrayscale(src); // RGB to Grayscale conversion

	Mat srcBinary(height, width, CV_8UC1); // Create gray image
	srcBinary = conversieGrayscaleToBinary(src, 100); // Grayscale to Binary conversion

	// Crop Image
	int minR = srcBinary.rows, maxR = 0;
	int minC = srcBinary.cols, maxC = 0;

	for (int r = 0; r < srcBinary.rows; r++) {
		for (int c = 0; c < srcBinary.cols; c++) {
			if (srcBinary.at<uchar>(r, c) == 0)
			{
				if (r < minR)
					minR = r;
				if (r > maxR)
					maxR = r;
				if (c < minC)
					minC = c;
				if (c > maxC)
					maxC = c;
			}
		}
	}

	Mat cropImage(maxR - minR, maxC - minC, CV_8UC1);
	for (int r = 0; r < maxR - minR; r++)
		for (int c = 0; c < maxC - minC; c++)
			cropImage.at<uchar>(r, c) = srcBinary.at<uchar>(r + minR, c + minC);


	Mat resizedImage = resizeImage(cropImage, 500);

	return resizedImage;
}

void algoritmHist()
{

	Mat srcInput = processedImage("...\\AplicatieProiect\\Semnaturi\\input\\image_0030.jpg");

	int hist_verticala[1000] = { 0 };
	int hist_orizontala[1000] = { 0 };

	for (int i = 0; i < srcInput.rows; i++)
	{
		for (int j = 0; j < srcInput.cols; j++)
		{
			if (srcInput.at<uchar>(i, j) == 0)
			{
				hist_verticala[j / (int)(256 / 32)] ++;
				hist_orizontala[i / (int)(256 / 32)] ++;
			}
		}
	}

	//imshow("Input", srcInput);
	//waitKey();


	string testImg[3];
	testImg[0] = "test1";
	testImg[1] = "test2";
	testImg[2] = "test3";

	for (int k = 0; k < 3; ++k)
	{
		float totalProcent = 0.0f;
		for (int i = 1; i <= 11; ++i)
		{
			string imageNumber = to_string(i);
			while (imageNumber.length() < 4)
				imageNumber = '0' + imageNumber;
		
			Mat imgTest = processedImage("...\\AplicatieProiect\\Semnaturi\\" + testImg[k] + "\\image_" + imageNumber + IMAGE_EXT);

			//imshow("Imagine", imgTest);
			//waitKey();

			float procent = histogramaPixeli(imgTest, 32, hist_verticala, hist_orizontala);
			totalProcent += procent;

			//printf("Procent[%d]=%f\n", i, procent);
		}

		totalProcent /= 10;
		printf("Procentaj %s: %f\n", testImg[k].c_str(), totalProcent);
	}

	imshow("Input", srcInput);
	waitKey(0);
	//histogramaPixeli(resizedImage, 32);
	

}

int main(int argc, char **argv)
{
	//algoritmKmeans();
	//algoritmHist();
	
	return(0);
}