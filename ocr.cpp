#include "cv.h"
#include "ml.h"
#include "highgui.h"

#include <iostream.h>
#include <fstream.h>

int train_samples = 5;
int test_samples = 50;
int num_classes = 10;
int max_k = 32;
int size = 28;
IplImage zeroimg;
CvMat* trainClasses = cvCreateMat(train_samples * num_classes, 1, CV_32FC1);
CvMat* trainData = cvCreateMat(train_samples * num_classes, size*size, CV_32FC1);

/* 
 * This function takes an image and to pointers to ints and sets
 * the ints equal to the minimum and maximum x-coordinates of the actual
 * image
 */
void findX(IplImage* imgSrc,int* min, int* max){
    int i;
    int minFound=0;
    CvMat data;
    CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
    CvScalar val=cvRealScalar(0);

   /*
    * For each col sum, if sum < width*255 then we find the min
    * then continue to end to search the max, if sum< width*255 
    * then is new max
    */
    for (i=0; i< imgSrc->width; i++){
        cvGetCol(imgSrc, &data, i);
        val= cvSum(&data);
        if(val.val[0] < maxVal.val[0]){
            *max= i;
            if(!minFound){
                *min= i;
                minFound= 1;
            }
        }
    }
}


/* 
 * This function takes an image and to pointers to ints and sets
 * the ints equal to the minimum and maximum x-coordinates of the actual
 * image
 */
void findY(IplImage* imgSrc,int* min, int* max){
    int i;
    int minFound=0;
    CvMat data;
    CvScalar maxVal=cvRealScalar(imgSrc->width * 255);
    CvScalar val=cvRealScalar(0);

   /*
    * For each col sum, if sum < width*255 then we find the min
    * then continue to end to search the max, if sum< width*255 
    * then is new max
    */
    for (i=0; i< imgSrc->height; i++){
        cvGetRow(imgSrc, &data, i);
        val= cvSum(&data);
        if(val.val[0] < maxVal.val[0]){
            *max=i;
            if(!minFound){
                *min= i;
                minFound= 1;
            }
        }
    }
}


/*
 * This function takes an image and finds returns a rectangle that contains
 * the actual image after cropping the blank white space.
 */
CvRect findBB(IplImage* imgSrc){
    CvRect aux;
    int xmin, xmax, ymin, ymax;
    xmin=xmax=ymin=ymax=0;
 
    findX(imgSrc, &xmin, &xmax);
    findY(imgSrc, &ymin, &ymax);
 
    aux=cvRect(xmin, ymin, xmax-xmin, ymax-ymin);
 
    //printf("BB: %d,%d - %d,%d\n", aux.x, aux.y, aux.width, aux.height);
 
    return aux;
 
}


/*
 * This function takes an image and the width and height we want to scale
 * it to, then crops the blank white space and scales the actual image
 * to those parameters.
 */
IplImage preprocessing(IplImage* imgSrc, int new_width, int new_height){
    IplImage* result;
    IplImage* scaledResult;
 
    CvMat data;
    CvMat dataA;
    CvRect bb; //bounding box
    CvRect bba; //boundinb box maintain aspect ratio
 
    //Find bounding box
    bb=findBB(imgSrc);
 
   /*
    * Get bounding box data and no with aspect ratio, 
    * the x and y can be corrupted
    */
    cvGetSubRect(imgSrc, &data, cvRect(bb.x, bb.y, bb.width, bb.height));
   /* 
    * Create image with this data with width and height with aspect ratio 1
    * then we get highest size betwen width and height of our bounding box
    */
    int size=(bb.width>bb.height)?bb.width:bb.height;
    result=cvCreateImage( cvSize( size, size ), 8, 1 );
    cvSet(result,CV_RGB(255,255,255),NULL);
    //Copy de data in center of image
    int x=(int)floor((float)(size-bb.width)/2.0f);
    int y=(int)floor((float)(size-bb.height)/2.0f);
    cvGetSubRect(result, &dataA, cvRect(x,y,bb.width, bb.height));
    cvCopy(&data, &dataA, NULL);
    //Scale result
    scaledResult=cvCreateImage( cvSize( new_width, new_height ), 8, 1 );
    cvResize(result, scaledResult, CV_INTER_NN);
 
    //Return processed data
    return *scaledResult;
}

// I used image files from http://cis.jhu.edu/~sachin/digit/digit.html. There 
// are ten files each containing 1000 28x28 sample images. Each pixel value is 
// stored as an unsigned byte and can range from 0 to 255. The images are 
// derived from the MNIST database.
//
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
           sprintf(filename, "images/data%i", i);
           ifstream read_file(filename);
           
           if(!read_file) // file couldn't be opened
                 printf("Error: file %s could not be opened\n", filename);
           else
                 printf("Reading file %s\n", filename);
           
           for(j = 0; j < train_samples; j++)
           {
                 // Create a new image of size 28x28. Each pixel is 8 bits 
                 // with one channel.
                 source = cvCreateImage(cvSize(size, size), IPL_DEPTH_8U,1);
                 
                 //for(x = 0; x < 1*size*size; x++) {read_file >> pixel;}
     
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
                             if((int)pixel != 0)
                                           printf("  ");
                             else
                                 printf("0 ");
                       }
                       printf("\n");
                 }
                 
                 //while(1);
                 // Process the image
                 processed = preprocessing(source, size, size);
                 
                 /*
                 // Testing the preprocessing code.
                 for(x = 0; x < size; x++)
                 {
                       for(y = 0; y < size; y++)
                       {
                             pixel2 = processed.imageData[size*x + y];
                             if((int)pixel2 != 0)
                                  printf("  ");
                             else
                                  printf("%i ", 0);
                       }
                       printf("\n");
                 }
                 
                 while(1);
                 */
                 
                 // Convert our 8-bit image into a 32-float image.
                 IplImage* floatImage = cvCreateImage(cvSize(size, size), 
                      IPL_DEPTH_32F, 1);
                 cvConvertScale(&processed, floatImage, 1, 0);//0.0039215, 0);
                 
                 if(i == 5 && j == 0) 
                      zeroimg = processed;
                 // Get reference to the i*train_samples + j row, and set
                 // it to i.
                 cvGetRow(trainClasses, &row, i*train_samples + j);
                 cvSet(&row, cvRealScalar(i));
                 
                 // Get reference to the i*train_samples + j row in trainData.
                 cvGetRow(trainData, &row, i*train_samples + j);
                 //printf("row rows: %i, cols: %i\n", row.rows, row.cols);
                 // Moves rectangle from source into data
                 cvGetSubRect(floatImage, &data, cvRect(0, 0, size, size));
                 //printf("data rows: %i, cols: %i\n", data.rows, data.cols);
                 CvMat row_header, *row1;
                 row1 = cvReshape(&data, &row_header, 0, 1);//(size*size/2));
                 //printf("row1 rows: %i, cols: %i\n", row1->rows, row1->cols);
                 cvCopy(row1, &row);
           }
           read_file.close();
     }
}

/*
float find_nearest(CvMat* img, int max_k) {
      float neighbors[max_k][2];
      float result;
      int x, y;
      double myval, otherval, clas, temp;
      float temp_dist, temp_class;
      float dist = 0;
      float top = 0;
      float bottom = 0;
      
      for(x = 0; x < max_k; x++)
      {
            neighbors[x][0] = -1;
            neighbors[x][1] = -1;
      }
      
      for(x = 0; x < train_samples*num_classes; x++)
      {
            clas = cvGet1D(trainClasses, x).val[0];
            for(y = 0; y < size*size; y++)
            {
                  otherval = cvGet2D(trainData, x, y).val[0];
                  myval = cvGet1D(img, y).val[0];
                  temp = otherval - myval;
                  if(temp < 0)
                        temp = -1 * temp;
                  dist = dist + temp;
            }
            for(y = 0; y < max_k; y++)
            {
                  if(neighbors[y][1] == -1 || neighbors[y][1] > dist)
                  {
                        temp_dist = neighbors[y][1];
                        temp_class = (int)neighbors[y][0];
                        neighbors[y][1] = dist;
                        neighbors[y][0] = clas;
                        dist = temp_dist;
                        clas = temp_class;
                  }
            }
            dist = 0;
      }
      for(x = 0; x < max_k; x++)
      {
            printf("class: %i, dist: %f\n", (int)neighbors[x][0], neighbors[x][1]);
            top = top + neighbors[x][0] * (1 / (neighbors[x][1] * neighbors[x][1]));
            bottom = bottom + (1 / (neighbors[x][1] * neighbors[x][1]));
      }
      
      result = top/bottom;
      return result;
}
*/

// Takes an image file and uses CvKNearest.find_nearest to convert the number 
// in the image into a float.
float classify(IplImage* img, CvKNearest* knn) {
      IplImage processed;
      CvMat data;
      CvMat* nearest_neighbors = cvCreateMat(1, max_k, CV_32FC1);
      float result;
      
      processed = preprocessing(img, size, size);
      
      // Convert our 8-bit image into a 32-float image.
      IplImage* floatImage = cvCreateImage(cvSize(size, size), 
           IPL_DEPTH_32F, 1);
      cvConvertScale(&processed, floatImage, 1, 0);//0.0039215, 0);
                 
      // Moves rectange from source into data
      cvGetSubRect(floatImage, &data, cvRect(0, 0, size, size));

      //printf("data rows: %i, data cols: %i\n", &data->rows, &data->cols);

      CvMat row_header, *row1, *row2;
      row1 = cvReshape(&data, &row_header, 0, 1);
      
      // Get the float corresponding to this image.
      result = knn->find_nearest(row1, max_k, 0, 0, 0, 0);
      //result = find_nearest(row1, max_k);
      return result;
}

// Runs the OCR on the test data (last 500 images of each data file) and prints 
// out the percentage of incorrect classifications.
void test(CvKNearest* knn) {
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
           
           for(j = 0; j < train_samples*size*size; j++)
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

                 result = classify(source, knn);
                 
                 printf("Reading the %ith image in this file, classified as: %i\n", j, (int)result);
                 if((int)result != i)
                       errors++;
           }
           read_file.close();
     }
     errorPercent = 100 * (float)errors / (float)(test_samples*num_classes);
     printf("Error percentage: %4.2f\n", errorPercent);
}
      
int main( int argc, char** argv ) {
    getData();  
    //CvKNearest* knn = new CvKNearest(trainData, trainClasses, 0, false, max_k);  
    //test(knn);
    //printf("result: %i\n", (int)classify(&zeroimg, knn));
    while(1);
}
