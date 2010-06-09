#include "cv.h"
#include "ml.h"
#include "highgui.h"
#include "cvaux.h"
#include "cxcore.h"

#include <iostream.h>
#include <fstream.h>
#include <math.h>

int noisy_percent = 50;
int filter, radius = 0;
int train_samples = 500;
int test_samples = 50;
int num_samples = 1000;
int num_classes = 10;
int max_k = 8;
int size = 28;
IplImage testImg;
CvMat* trainClasses = cvCreateMat(train_samples * num_classes, 1, CV_32FC1);
CvMat* trainData = cvCreateMat(train_samples * num_classes, size*size, 
      CV_32FC1);
IplImage* img0 = 0, *img = 0;
CvPoint prev_pt = {-1,-1};

/* 
 * This function takes an image and two pointers to ints and sets
 * the ints equal to the minimum and maximum x-coordinates of the actual
 * image
 */
void findX(IplImage* imgSrc,int* min, int* max){
    int i;
    int minFound = 0;
    CvMat data;
    CvScalar maxVal = cvRealScalar(imgSrc->width * 255);
    CvScalar val = cvRealScalar(0);

   /*
    * For each col sum, if sum < width*255 then we find the min
    * then continue to end to search the max, if sum< width*255 
    * then it is the new max
    */
    for (i=0; i< imgSrc->width; i++){
        cvGetCol(imgSrc, &data, i);
        val = cvSum(&data);
        if(val.val[0] < maxVal.val[0]){
            *max = i;
            if(!minFound){
                *min = i;
                minFound = 1;
            }
        }
    }
}


/* 
 * This function takes an image and two pointers to ints and sets
 * the ints equal to the minimum and maximum x-coordinates of the actual
 * image
 */
void findY(IplImage* imgSrc,int* min, int* max){
    int i;
    int minFound = 0;
    CvMat data;
    CvScalar maxVal = cvRealScalar(imgSrc->width * 255);
    CvScalar val = cvRealScalar(0);

   /*
    * For each col sum, if sum < width*255 then we find the min
    * then continue to end to search the max, if sum< width*255 
    * then it is the new max
    */
    for (i=0; i< imgSrc->height; i++){
        cvGetRow(imgSrc, &data, i);
        val = cvSum(&data);
        if(val.val[0] < maxVal.val[0]){
            *max = i;
            if(!minFound){
                *min = i;
                minFound = 1;
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
 * This function takes an image and the width and height and computes an
 * average value for the darkness of each pixel then binarizes the image
 */

IplImage binarize(IplImage* imgSrc, int new_width, int new_height) {
	int x, y;
	int average = 0;
	unsigned char pixel;

	/* Get the sum of all the pixel values */
	for (x=0; x<new_width; x++) {
		for (y=0; y<new_height; y++) {
			pixel = imgSrc->imageData[size*x + y];
			average += (int) pixel;
		}
	}

	/* average them */
	average = average / (new_width * new_height);

	/* Then compare to each pixel */
	for (x=0; x<new_width; x++) {
		for (y=0; y<new_height; y++) {
			pixel = imgSrc->imageData[size*x + y];
			if (average > (int) pixel) {
				/* If lighter than the average, set to 0(white) */
				pixel = 0;
			} else {
				/* If darker than the average set to 255(black) */
				pixel = 255;
			}
		}
	}
}

/*
 * This function takes an image and the width and height we want to scale
 * it to, then crops the blank white space and scales the actual image
 * to those parameters.
 */
IplImage preprocessing(IplImage* imgSrc, int new_width, int new_height, 
    int filter, int radius){
          
    IplImage* result;
    IplImage* scaledResult;
 
    CvMat data;
    CvMat dataA;
    CvRect bb; //bounding box
    CvRect bba; //boundinb box maintain aspect ratio
 
    if(filter != 0)
	/*
	 * Apply filter to the image on a radius x radius range; this should reduce 
     * noise as blurring the image would reduce the impact of random pixels of 
     * the wrong color. 
	 */
    cvSmooth(imgSrc, imgSrc, filter, radius, radius);

    //Find bounding box
    bb = findBB(imgSrc);
 
   /*
    * Get bounding box data and no with aspect ratio, 
    * the x and y can be corrupted
    */
    cvGetSubRect(imgSrc, &data, cvRect(bb.x, bb.y, bb.width, bb.height));
   /* 
    * Create image with this data with width and height with aspect ratio 1
    * then we get highest size betwen width and height of our bounding box
    */
    int size = (bb.width>bb.height)?bb.width:bb.height;
    result = cvCreateImage( cvSize( size, size ), 8, 1 );
    cvSet(result,CV_RGB(255,255,255),NULL);
    //Copy de data in center of image
    int x = (int)floor((float)(size-bb.width)/2.0f);
    int y = (int)floor((float)(size-bb.height)/2.0f);
    cvGetSubRect(result, &dataA, cvRect(x,y,bb.width, bb.height));
    cvCopy(&data, &dataA, NULL);
    //Scale result
    scaledResult = cvCreateImage( cvSize( new_width, new_height ), 8, 1 );
    cvResize(result, scaledResult, CV_INTER_NN);
 
	// We might want to binarize the result
	binarize(scaledResult, new_width, new_height);

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
     IplImage* source;
     IplImage processed;
     CvMat row, data;
     CvScalar value;
          
     // Read the image files data0 - data9.
     for(i = 0; i < num_classes; i++)
     {
           sprintf(filename, "images/data%i", i);
           FILE *read_file = fopen(filename, "rb");
           
           if(!read_file)
                 printf("Failed to read file %s\n", filename);
           else
                 printf("Training with data from %s\n", filename);
           
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
                             unsigned char pixel = fgetc(read_file);
                             value = cvRealScalar((int)pixel);
                             cvSet2D(source, x, y, value);
                       }
                 }

                 // Process the image
                 processed = preprocessing(source, size, size, filter, radius);
                 
                 // Convert our 8-bit image into a 32-float image.
                 IplImage* floatImage = cvCreateImage(cvSize(size, size), 
                      IPL_DEPTH_32F, 1);
                 cvConvertScale(&processed, floatImage, 0.0039215, 0);
                 
                 // Get reference to the i*train_samples + j row, and set
                 // it to i.
                 cvGetRow(trainClasses, &row, i*train_samples + j);
                 cvSet(&row, cvRealScalar(i));
                 
                 // Get reference to the i*train_samples + j row in trainData.
                 cvGetRow(trainData, &row, i*train_samples + j);

                 // Moves rectangle from source into data
                 cvGetSubRect(floatImage, &data, cvRect(0, 0, size, size));
                 
                 // Converts the 2D date matrix into a 1D array.
                 CvMat row_header, *row1;
                 row1 = cvReshape(&data, &row_header, 0, 1);

                 cvCopy(row1, &row);
           }
           fclose(read_file);
     }
}

// Takes an image file and uses CvKNearest.find_nearest to convert the number 
// in the image into a float.
float classify(IplImage* img, CvKNearest* knn) {
      IplImage processed;
      CvMat data;
      CvMat* nearest_neighbors = cvCreateMat(1, max_k, CV_32FC1);
      float result;
      
      processed = preprocessing(img, size, size, filter, radius);
      
      // Convert our 8-bit image into a 32-float image.
      IplImage* floatImage = cvCreateImage(cvSize(size, size), 
           IPL_DEPTH_32F, 1);
      cvConvertScale(&processed, floatImage, 0.0039215, 0);
      
                 
      // Moves rectange from source into data
      cvGetSubRect(floatImage, &data, cvRect(0, 0, size, size));

      // Converts the 2D data matrix into a 1D array.
      CvMat row_header, *row1, *row2;
      row1 = cvReshape(&data, &row_header, 0, 1);
      
      // Get the float corresponding to this image.
      result = knn->find_nearest(row1, max_k, 0, 0, 0, 0);
      return result;
}

// Runs the OCR on the test data (last 500 images of each data file) and prints 
// out the percentage of incorrect classifications.
void test(CvKNearest* knn) {
     int x, y, i, j, errors = 0;
     float result, errorPercent;
     char filename [50];
     unsigned char pixel; 
     IplImage* source;
     IplImage processed;
     CvScalar value;
          
     // Read the image files data0 - data9.
     for(i = 0; i < num_classes; i++)
     {
           sprintf(filename, "imagesnoisy/%i/data%i", noisy_percent, i);
           FILE *read_file = fopen(filename, "rb");
           
           if(!read_file)
                 printf("Failed to read file %s\n", filename);
           
           // Read the training images so that we can get to the test images.
           for(j = 0; j < train_samples*size*size; j++)
                 unsigned char pixel = fgetc(read_file);
           
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
                             unsigned char pixel = fgetc(read_file);
                             value = cvRealScalar((int)pixel);
                             cvSet2D(source, x, y, value);
                       }
                 }

                 result = classify(source, knn);
                 printf("Image classified as %i.\n", (int)result);
                 
                 if((int)result != i)
                       errors++;
           }
           fclose(read_file);
     }
     errorPercent = 100 * (float)errors / (float)(test_samples*num_classes);
     printf("Correct percentage: %4.2f\n", 100 - errorPercent);
}

void on_mouse( int event, int x, int y, int flags, void* )
{
    if( !img )
        return;

    if( event == CV_EVENT_LBUTTONUP || !(flags & CV_EVENT_FLAG_LBUTTON) )
        prev_pt = cvPoint(-1,-1);
    else if( event == CV_EVENT_LBUTTONDOWN )
        prev_pt = cvPoint(x,y);
    else if( event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON) )
    {
        CvPoint pt = cvPoint(x,y);
        if( prev_pt.x < 0 )
            prev_pt = pt;
        cvLine( img, prev_pt, pt, cvScalarAll(255), 75, CV_AA, 0 );
        prev_pt = pt;
        cvShowImage( "image", img );
    }
}

// Reads the image files and adds salt and pepper noise to them. noisy_percent 
// is the percentage of pixels that will have their values changed. ie, for 25% 
// noise set noisy_percent = 25.
void makeNoisy(int noisy_percent) {
     int x, y, i, j;
     double randnum;
     char filename [50], writename[50];
     
     srand((unsigned)time(0));
     
     // Read the image files data0 - data9.
     for(i = 0; i < num_classes; i++)
     {
           sprintf(filename, "images/data%i", i);
           sprintf(writename, "imagesnoisy/%i/data%i", noisy_percent, i);
           FILE *read_file = fopen(filename, "rb");
           FILE *write_file = fopen(writename, "wb");
           
           if(!read_file)
                 printf("Failed to read file %s.\n", filename);
           else
                 printf("Making noisy data from %s.\n", filename);
                 
           if(!write_file)
                 printf("Failed to open write file %s.\n", writename);
           else
                 printf("Writing noisy data to file %s.\n", writename);
           
           for(j = 0; j < num_samples; j++)
           {
                 for(x = 0; x < size; x++)
                 {
                       for(y = 0; y < size; y++)
                       {
                             // Read in the next pixel value from the file.
                             unsigned char pixel = fgetc(read_file);
                             randnum = (double) rand() / RAND_MAX;
                             // If randnum is above a certain percent, then 
                             // replace it with a random pixel value.
                             if(randnum < (double)noisy_percent/100)
                             {
                                   int newpixel = rand() % 255;
                                   pixel = (unsigned char)newpixel;
                             }
                             fputc(pixel, write_file);
                       }
                 }
           }
           fclose(read_file);
           fclose(write_file);
     }
}
      
int main( int argc, char** argv ) {
    int testing = -1;
    char input;
    
    while(1) {
         printf("Would you like the OCR to run in testing mode? (y/n)\n\n");
         cin >> input;
         if(input == 'y') {
              testing = 1;
              break;
         }
         else if (input == 'n') {
              testing = 0;
              break;
         }
    }
    
    printf("\nWhat type of noise reduction filter would you like to use?\n"
           "Choices are: N (None), G (Gaussian), M (Median), "
           "B (Bilateral).\n\n");
           
    while(1)
    {
         char c;
         cin >> c;
         
         if(c == 'n' || c == 'N')
         {
              filter = 0;
              printf("\nNo noise reduction will be used.\n\n");
              break;
         }
         if(c == 'g' || c == 'G')
         {
              filter = CV_GAUSSIAN;
              printf("\nGaussian noise reduction will be used.\n\n");
              break;
         }
         if(c == 'b' || c == 'B')
         {
              filter = CV_BILATERAL;
              printf("\nBilateral reduction will be used.\n\n");
              break;
         }
         if(c == 'm' || c == 'M')
         {
              filter = CV_MEDIAN;
              printf("\nMedian noise reduction will be used.\n\n");
              break;
         }
         else
         {
             printf("\nNot a valid choice. Please try again\n\n");
         }
    }
    
    while(filter != 0)
    {
         printf("\nPlease enter a radius for the noise reduction filter (positive odd number).\n\n");
         cin >> radius;
         if(radius > 0 && radius%2 == 1)
              break;
    }    
    
    while(testing == 1) {
         printf("\nPlease enter a noise percentage to test (0 - 100, multiples of 10).\n\n"); 
         cin >> noisy_percent;
         if(noisy_percent >= 0 && noisy_percent%10 == 0)
              break;
    } 
    
    getData();  
    CvKNearest* knn = new CvKNearest(trainData, trainClasses, 0, false, max_k); 
    
    if(testing == 1) {
            test(knn);
            while(1);
            }
    else {
    printf( "\nPress ENTER to classify the image.\nPress 'e' to calculate error"
    "\nPress ESC to quit.\n\n");
    
    cvNamedWindow( "image", 1 );

    img = cvCreateImage(cvSize(400, 400), 8, 1);
    int count = 0;
    int errors = 0;

    cvZero(img);
    cvShowImage( "image", img );
    cvSetMouseCallback( "image", on_mouse, 0 );
    while(1)
    {
        int c = cvWaitKey(0);
        
        if(c == 27)
            break;
        
        if(c == 'e')
            printf("Correct percentage: %4.2f\n", 100 - 100 * (float)errors/(float)count);

        if(c == '\r')
        {
            float result = classify(img, knn);
            printf("Image classified as %i.\nIs this correct? (y/n)\n\n", (int)result);
            while(1)
            {
                    int d = cvWaitKey(0);
                    if(d == 'n')
                    {
                         errors++;
                         break;
                    }
                    if(d == 'y')
                         break;
            }
            count++;   
            cvZero(img);
            cvShowImage( "image", img );
        }
    }
    }
    return 1;
}
