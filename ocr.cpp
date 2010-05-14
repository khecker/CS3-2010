#ifdef _CH_
#pragma package <opencv>
#endif

#ifndef _EiC
#include "cv.h"
#include "ml.h"
#include "highgui.h"
#endif

#include <iostream.h>
#include <fstream.h>

int train_samples = 500;
int test_samples = 500;
int num_classes = 10;
int max_k = 32;
int size = 28;
CvMat* trainClasses = cvCreateMat(train_samples * num_classes, 1, CV_32FC1);
CvMat* trainData = cvCreateMat(train_samples * num_classes, 2, CV_32FC1);
CvKNearest knn;

// I used image files from http://cis.jhu.edu/~sachin/digit/digit.html. There 
// are ten files each containing 1000 28x28 sample images. Each pixel value is 
// stored as an unsigned byte and can range from 0 to 255. The images are 
// derived from the MNIST database.

IplImage preprocessing(IplImage* imgSrc, int new_width, int new_height) {
     return *imgSrc;
}

// This function extracts the pixel data from the files, processes the images 
// and stores the images in the data matrices.
void getData() {
     int x, y, i, j;
     char filename [50];
     unsigned char pixel; 
     IplImage* source;
     IplImage processed;
     CvMat row, data;
     CvScalar value;
          
     // Read the image files data0 - data9.
     for(i = 0; i < num_classes; i++)
     {
           ifstream read_file;

           sprintf(filename, "images/data%i", i);
           read_file.open(filename);
           
           if(!read_file) // file couldn't be opened
                 printf("Error: file %s could not be opened\n", filename);
           else
                 printf("Reading file %s\n", filename);
           
           for(j = 0; j < train_samples; j++)
           {
                 // Create a new image of size 28x28. Each pixel is 8 bits 
                 // with one channel.
                 source = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U,1);
     
                 for(x = 0; x < size; x++)
                 {
                       for(y = 0; y < size; y++)
                       {
                             // Read in the next pixel value from the file, 
                             // and set the corresponding pixel in the image 
                             // to that value.
                             read_file >> pixel;
                             value = cvRealScalar((int)pixel);
                             cvSet2D(source, x, y, value);
                       }
                 }
                 
                 read_file.close();
                 
                 // Process the image
                 processed = preprocessing(source, size, size);
                 
                 // Set classes.
                 cvGetRow(trainClasses, &row, i*train_samples + j);
                 cvSet(&row, cvRealScalar(i));
                 
                 // Set data.
                 cvGetRow(trainData, &row, i*train_samples + j);
                 
                 // Convert our 8-bit image into a 32-float image.
                 IplImage* floatImage = cvCreateImage(cvSize(size, size), 
                      IPL_DEPTH_32F, 1);
                 cvConvertScale(&processed, floatImage, 0.0039215, 0);
                 
                 // Moves rectangle from source into data
                 cvGetSubRect(floatImage, &data, cvRect(0, 0, size, size));

                 CvMat row_header, *row1;
                 row1 = cvReshape(&data, &row_header, 0, 1);
                 //cvCopy(row1, &row);
           }
     }
}

// Takes an image file and uses CvKNearest.find_nearest to convert the number 
// in the image into a float.
float classify(IplImage* img) {
      IplImage processed;
      CvMat data;
      CvMat* nearest_neighbors = cvCreateMat(1, max_k, CV_32FC1);
      float result;
      
      processed = preprocessing(img, size, size);
      
      // Convert our 8-bit image into a 32-float image.
      IplImage* floatImage = cvCreateImage(cvSize(size, size), 
           IPL_DEPTH_32F, 1);
      cvConvertScale(&processed, floatImage, 0.0039215, 0);
                 
      // Moves rectange from source into data
      cvGetSubRect(floatImage, &data, cvRect(0, 0, size, size));

      CvMat row_header, *row1;
      row1 = cvReshape(&data, &row_header, 0, 1);
      
      // Get the float corresponding to this image.
      result = knn.find_nearest(row1, max_k, 0, 0, nearest_neighbors, 0);
      return result;
}

// Runs the OCR on the test data (last 500 images of each data file) and prints 
// out the percentage of incorrect classifications.
void test() {
     int x, y, i, j, errors;
     float result, errorPercent;
     char filename [50];
     unsigned char pixel; 
     IplImage* source;
     IplImage processed;
     CvScalar value;
          
     // Read the image files data0 - data9.
     for(i = 0; i < num_classes; i++)
     {
           ifstream read_file;

           sprintf(filename, "images/data%i", i);
           read_file.open(filename);
           
           if(!read_file) // file couldn't be opened
                 printf("Error: file %s could not be opened\n", filename);
           else
                 printf("Reading file %s\n", filename);
           
           for(j = 0; j < train_samples; j++)
           {
                 read_file >> pixel;
           }
           
           for(j = 0; j < test_samples; j++)
           {
                 // Create a new image of size 28x28. Each pixel is 8 bits 
                 // with one channel.
                 source = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U,1);
     
                 for(x = 0; x < size; x++)
                 {
                       for(y = 0; y < size; y++)
                       {
                             // Read in the next pixel value from the file, 
                             // and set the corresponding pixel in the image 
                             // to that value.
                             read_file >> pixel;
                             value = cvRealScalar((int)pixel);
                             cvSet2D(source, x, y, value);
                       }
                 }
                 
                 read_file.close();
                 
                 // Process the image
                 processed = preprocessing(source, size, size);
                 
                 result = classify(&processed);
                 
                 if((int)result != i)
                       errors++;
           }
     }
     errorPercent = 100 * (float)errors / (float)test_samples;
     printf("Error percentage: %4.2f\n", errorPercent);
}
      
int main( int argc, char** argv ) {
    getData();
    knn(trainData, trainClasses, 0, false, max_k);
    test();
}
