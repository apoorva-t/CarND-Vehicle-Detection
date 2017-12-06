## Writeup 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_notcar.png
[image2]: ./examples/hog_image.png
[image3]: ./examples/slide_window.png
[image4]: ./examples/detection.png
[image5]: ./examples/frame_heat.png
[image6]: ./examples/labeled_frames.png
[image7]: ./examples/img_with_bbox.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third and fourth code cell of the IPython notebook under 'Functions for HOG, spatial and color histogram feature extraction'.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I started off with the same parameter values as used in the lecture modules. The combination that gave the highest test accuracy and that gave the best predictions on the test images was chosen. The final parameter values I chose were: orientations = 9, cells_per_block=(2,2), pixel_per_cell=(8,8) and hog_channel = 'ALL'.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The features for car and noncar image data are extracted in the block 'Feature Extraction for car and noncar images'. I found that using HOG with all channels alongwith spatial binning and color histogram features gave better results than using just the HOG features. Using a colorspace of 'YCrCb' gives a better prediction than using 'RGB'. The features for car and noncar images are stacked together, standardized to zero mean and unit variance, and randomly split into training and test data sets.

Then I a linear SVM is trained using the LinearSVC classifier from sklearn. With the default value of the C parameter, the classifier gave an accuracy of ~98% on the test data split from the input dataset. However, using this on the test images provided resulted in quite a few stray boxes being generated (false positives). This meant the classifier was not performing as well. To improve the performance of the linear svm classifier, I then used the GridSearchCV function provided by sklearn to tune the value of parameter 'C' for the best possible estimation.

The code for this is under the block 'Run classification'.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the function 'find_cars' under the block titled 'Sliding window search'. The basic structure of this code is taken from the implementation given in the lecture modules. The values 'y_start', 'y_stop', 'x_start', 'x_stop' together determine the patch of the image to run the sliding window search on. Since cars are expected to be found in the lower part of the image frame, y_start mostly begins from ~360 onwards. Multiple searches are run with different scales (window sizes) - smaller scales in the middle patch of the frame (since cars detected here will appear smaller), and larger scales for the lower part of the frame (where cars will appear larger). Initially, I search with scales ranging from 1.0 to 3.5 with steps of 0.5. However, since search with scales beyond 2.5 did not give substantially better results, I limited it to 2.5. The cells_per_step was set to (2,2) giving an overlap of 75%, which gave good results with a reasonable processing time.

As explained in the lectures, the HOG features are extracted for the entire image frame of interest initially and then the relevant features are extracted for the search window under consideration.

The output of the find_cars function is a list of bounding rectangles around car detections.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  After running the sliding window search and detection on the region of interest, we get a number of overlapping rectangles for positive car detections. I have used a heatmap, as recommended in the lessons, to find regions of high overlap. Overlapping/adjoining high heat areas are extracted using the labels function of scipy. The final bounding box is defined by the edges that satisfy the heatmap threshold. Here is an example of how the pipeline works on a test image:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections (cars found) in each frame of the video.  From the positive detections I created a heatmap. Since consecutive frames in a video are very closely spaced, we don't expect to see a large movement in the position of the car between a few consecutive frames. So to keep the bounding box of car detection relatively smooth and filter out false positives, I am adding up the heatmaps across a few frames of the video and then applying a threshold to that map to identify vehicle positions. The number of frames to sum across and the threshold to use for a good detection were found by trial and error. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I started off by using the computer vision approach for defining and extracting features, and training a Linear SVM to fit these as described in the lecture modules. The choice of hyper-parameters for the classifier and kind of classifier to use have a significant impact on accuracy of the detection. Just by using a GridSearchCV function to tune the 'C' parameter for the SVM instead of using the default, I was able to filter out false positives much more efficiently. The choice of region of interest to use for the sliding window search impacted the accuracy and processing time of the detection pipeline. Using a range of scales between 1.0 to 2.5 (effectively leading to variable search window sizes) was very effective - this is the part which involved a lot of trial and error to come up with the values of 'y_start' and 'y_stop'.

In the current pipeline, I have hard-coded the 'x_start' and 'x_stop' for the region of intereset assuming the car is driving in the leftmost lane (as is the case in the test images and video). To make it work for any lane, this would need to work in sync with the rest of the autonomous driving system, where the car will know which lane it is driving in (center or extreme left/right) and switch the 'x start/stop' values accordingly. 

Also, since this pipeline is using a computer vision approach for vehicle detection, it performs very slowly - not suitable for real-time detection. I believe using an ML approach would lead to much better processing time which is something I will explore next.

To make the pipeline work under various conditions, the training data will need to be augmented with different lighting conditions, frames under different driving conditions (for example: hairpin bends on a hill), and of course other types of vehicles.  

