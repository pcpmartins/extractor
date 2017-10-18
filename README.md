
'''Paper CBMI: Semi-automatic Video Assessment System''' [[File:Semiautomatic_video_assessment_system.pdf]]


==Visual feature extraction tool==

A command line utility that can be used to generate metadata for visual quality assessment, it relies on opencv 3.2 library of algorithms to compute features that describe images and videos in respect to visual properties carefully selected from state-of-the-art literature. An extensive review and testing was made to find a group of visual features good for the task of visual discrimination.
A top-down aproach for extraction results in a very compact feature vector. A graphical interface (demo available [https://drive.google.com/file/d/0Bzelhrdw43rCc0JkOGdSNnYtclE/view?usp=sharing here]) was used to check overall usability and performance of the features. A subset was used to train classifiers for aesthetic and interestingness and some of the features were also indexed and used to compute a similarity metric.

=== Usage ===

The utility accepts images or videos as input. The files can be loaded from a text file list or by providing a folder path.
The output feature vector is saved in CSV format. 

{{hc|opencv-vq -h|2=<nowiki>
Usage: execfile <inputFile> <outputFile> [arguments]

Arguments:
  -v, example usage: -v=false
      Set video processing off (for image files)
  -q, example usage: -q=false
      Quiet mode set off.
  -t, example usage: -t=0
      test histogram output off.
      0-none, 1-hue histogram, 2-static saliency, 3- optical flow 
      4-dynamic saliency, 5-background subtraction
      6-faces, 7-Gabor filter, 8-edge histograms
  -d, example usage: -d=true
      process files from a folder.
  -s, example usage: -s=5
      set sampling factor to 5.
  -r, example usage: -r=1
      set resize mode to 320x240.
      0-no resize,1-320x240, 2-480x360, 3-640x480.
  -b, example usage: -b=0
      background subtraction off.
      0-none, 1-knn, 2-mog2.
  -f, example usage: -f=false
      optical flow off.
  -i, example usage: -i=true
      use Haar classifiers intersection.
  -g, example usage: -g=true
      extract Gabor features.
  -e, example usage: -e=true
      extract edge histograms.
  -h, example usage: -h=true
      this help message.
      </nowiki>
}}

example 1: Extract features from image files in bin/data/images folder to the CSV file images_output.csv

{{hc|<nowiki>opencv-vq bin/data/images images_output.csv -v=false -d=true</nowiki> |2=<nowiki>

>> parsing input: bin/data/images folder

> processing: bin/data/images/2hues.jpg 
> processing: bin/data/images/5hues.jpg 
> processing: bin/data/images/8hues.jpg 
> processing: bin/data/images/black.jpg 
> processing: bin/data/images/blue.jpg 
> processing: bin/data/images/face-4.jpg 
> processing: bin/data/images/face-8.jpg 
> processing: bin/data/images/face-gaza.jpg 
> processing: bin/data/images/face-kids.jpg 
> processing: bin/data/images/face-obama.jpg 
> processing: bin/data/images/facts-about-eyesight-600x375.jpg 
> processing: bin/data/images/frame1.jpg 
> processing: bin/data/images/frame2.jpg 
> processing: bin/data/images/lena.jpg 
> processing: bin/data/images/paprika.jpg 
> processing: bin/data/images/red.jpg 
> processing: bin/data/images/rule.jpg 
> processing: bin/data/images/spots2.png 
> processing: bin/data/images/test_edge_nond.jpg 
> processing: bin/data/images/test_edges_135.jpg 
> processing: bin/data/images/test_edges_45.jpg 
> processing: bin/data/images/test_edges_h.jpg 
> processing: bin/data/images/test_edges_v.jpg 
> processing: bin/data/images/x-135.jpg 
> processing: bin/data/images/x-45.jpg 
> processing: bin/data/images/x-nond.jpg 
>> saved output file: images_output.csv
>> saved ml file: mloutput.csv

      </nowiki>
}}
The resulting output file can be found [https://drive.google.com/file/d/0Bzelhrdw43rCWWc5QWQtQmp6Y0E/view?usp=sharing here].

example 2: Extract features from video files in videos.txt to the CSV file video_output.csv
{{hc|<nowiki>opencv-vq videos.txt video_output.csv -v=true -d=false</nowiki> |2=<nowiki>
>> parsing input: videos.txt text file

> processing: bin/data/videos/1_fingers.mov 30.099fps 320x240 
> processing: bin/data/videos/4_arabic_news.mov 25fps 320x240 |
> processing: bin/data/videos/5_mov_480x360.mov 14.8247fps 480x360 
>> saved output file: video_output.csv
>> saved ml file: mloutput.csv

      </nowiki>
}}

[https://drive.google.com/file/d/0Bzelhrdw43rCQURPaHYwaUoyMFk/view?usp=sharing Here] is the resulting feature vector in CSV format.

=== Feature description ===

In this section we made a list of features,for each one we explain its relevancy and show the range of acceptable values

*file_path - path to media file
The path of folder, text file or url to the media.
note: -d argument should be set acordingly.

*width - [1-1920]
Width of the image or video.

*height - [1-1080]
Height of image or video.

====Focus====
*focus - [0.0-1.0]
Based on a survey implementation of many focus measurement algorithms in [https://drive.google.com/file/d/0B6UHr3GQEkQwYnlDY2dKNTdudjg/view "Analysis of focus measure operators for shape from focus"]

We selected the OpenCV port of 'LAPV' Variance of Laplacian algorithm ([http://decsai.ugr.es/vip/files/conferences/Autofocusing2000.pdf Pech2000]).

This kind of algorithms are used in camera auto-focus, and rely on edge detection since we can detect more edges in a focused image then in a blurred version of it. Returns  the maximum sharpness detected, which is a pretty good indicator of if a camera is in focus or not. Not surprisingly, normal values are scene dependent but much less so than other methods like FFT which has too high of a false positive rate to be as useful.

====Color moments====
The basis of color moments lays in the assumption that the distribution of color in an image  can be interpreted as a probability distribution. Probability distributions are characterized by a number of unique moments (e.g. Normal distributions are differentiated by their mean and variance). It therefore follows that if the color in an image follows a certain probability distribution, the moments of that distribution can then be used as features to identify that image based on color. Color moments are scaling and rotation invariant, they are also a good feature to use under changing lighting conditions.
We use the first two central moments of color distribution. They are the mean and standard deviation computed as explained in [http://spie.org/Publications/Proceedings/Paper/10.1117/12.205308 "M. Stricker and M. Orengo Similarity of color images."]

We compute moments for each RGB channel.

Mean - The first color moment can be interpreted as the average color in the image.
*red_moment_1 - [0-255]
*green_moment_1 - [0-255]
*blue_moment_1 - [0-255]

Standard deviation - The second color moment which is obtained by taking the square root of the variance of the color distribution.
*red_moment_2 - [0-255]
*green_moment_2 - [0-255]
*blue_moment_2 - [0-255]

====Color ratio====
Color moments are great as a measure to judge similarity between images or videos, but it's a concept hard to be directly interpreted by a human user, instead we devised a better feature based on the color average given by the first moment, the color ratio. For a human is much easier to understand and use this concept.

*red_ratio - [0.0-1.0]
*blue_ratio - [0.0-1.0]
*green_ratio - [0.0-1.0]

====Luminance====
Luminance is arguable the most important component of color, humans are so much more sensitive to light than color information. We compute the first luminance moment.

*luminance - [0.0-1.0]

We also compute the luminance standard deviation. In terms of the human visual perception system the variance of luminance may be more informative than the mean or median.

*luminance_std - [0.0-1.0]
====edges====
Edges may indicate the boundaries of objects in the scene. Detecting sharp changes in brigthness between each pixel and its surroundings allow us to detect edges.We start by first applying Gaussian Blur (3x3) to add some robustness to eventual presence of noise. Afterwards we apply Prewitt or Sobel kernel convolution to differentiate edge pixels and give some perspective about the general orientation of edges found in the scene (horizontal, vertical or diagonal predominance).

*v_edges - [0.0-1.0].
*h_edges - [0.0-1.0].
*d_edges - [0.0-1.0].
*orientation - [Horz-0, Vert-1, Diag-2].

note: This features are being replaced by the edge histogram algorithm + grid support tool

*EH_0 - [0,1,2,3,4,5]
(...) 
*EH_15 - [0,1,2,3,4,5]

We implented a simplification of the method presented in [https://www.dcc.fc.up.pt/~mcoimbra/lectures/VC_1415/VC_1415_P8_LEH.pdf Efficient Use of Local Edge Histogram Descriptor] where edge histograms MPEG-7 compliant features are extracted.We used a global and a local representation, for the local one a support tool was used to split each frame by a 4x4 grid, resulting in 16 sub-images. Each one of this sub-images was convolved with 5 different oriented kernels (0-zero, 1-vertical, 2-horizontal, 3-45ª, 4-135ª, 5-non-directional) this process results in 16 local features representing the orientation of each sub-image.

*EH_16 - [0,1,2,3,4,5]
The global feature is computed using the full image.

====Color diversity====

Color diversity is a property related to visual aesthetics.
*dif_hues - [0-360].
We count different groups of hues. On the hue histogram (computed from the H channel of HSV) we count any variation of hue bigger than a certain threshold.

using the argument -t=1 we turn on a visualization were we can see the hue histograms computed for each image or frame.

*simplicity - [0.0-1.0]
At the moment this feature relies on two factors:

-We count groups of different hues present on the H channel of HSV.
-Edge composition - it's a measure of edges detected on the image.
We compute the harmonic mean of this two factors.

====Object detection====
We apply object Detection using Haar feature-based cascade classifiers, this method was originally proposed by Paul Viola and Michael J. Jones. in the International Journal of Computer Vision, 57(2):137–154, 2004.
We instantiate two cascade classifiers loading data from existing .xml classifier files, the first one is used for face detection, and the second can be used to load any of the other existing classifiers,
there are around 30 different human features classifiers in the folder (/bin/data/haar), we can find features like eyes, mouth, nose and smile among others.

Using the argument -t=6 everytime a feature is detected we have acess in a window to its visualization.

*faces - images[0,1] , videos[0.0-1.0]
Faces capture attention, to the detriment of other visual stimuli, in a visual cueing paradigm, observers responded faster to a target probe appearing in the location of a face cue than of a competing object cue, [http://jov.arvojournals.org/article.aspx?articleid=2192946 see].
This measure have a boolean value for images or a double value for videos, in the first case it signs if it was found or not faces in the image, in the second it represents the average number of video frames where faces were detected.

*faces_area - [0.0-1.0]
Ratio of face area(face bounding box) to full image area.
If faces are found we measure their areas, we consider that preponderancy of attention faces can arouse is directly related to their visible area.

*smiles -  images[0,1] , videos[0.0-1.0]
When someone smiles he's normally seen as a person with positive intentions. Generally people tend to be attracted by smiling subjects.We do smile detection with cascade classifier to compute a smile measure.It represents average frames where smiles were found, for videos, and smiles found for images.
Depending on the boolean parameter "insideFace" smiles are detected inside or outside faces.

*rule_of_thirds - [0.0-1.0]
The rule of thirds is one of the usual guidelines used to assess quality of visual image composition, The image is divided in nine equal parts by four equal spaced lines(the power lines), two oriented horizontally and the other two vertically, important compositional elements should be placed along these lines or their intersections.
Like in [https://ai2-s2-pdfs.s3.amazonaws.com/ab62/68d290b23fee36c3faf1ae77f0fd900a225f.pdf “Evaluating Visual Aesthetics in Photographic Portraiture”] we use a Spatial composition template to score images according to a variation of the rule of thirds where more power points were added, the template is in grayscale format, afterwards we compare face centroids from the previous step of the pipeline with this template and assign a score ranging from 0-255, this value is afterward normalized to the [0.1-1.0] range.

====Saliency====
Topics related to saliency were adapted from the opencv [http://docs.opencv.org/3.0-beta/modules/saliency/doc/saliency.html documentation].

Many computer vision applications may benefit from understanding where humans focus given a scene. Other than cognitively understanding the way human perceive images and scenes, finding salient regions and objects in the images helps various tasks such as speeding up object detection, object recognition, object tracking and content-aware image editing.

*static_saliency[0.0-1.0]
Algorithms belonging to this category, exploit different image features that allow to detect salient objects in a non dynamic scenario. We experiment with both approaches for static saliency offered by the new opencv 3.2 Saliency API:

[SpectralResidual] - Starting from the principle of natural image statistics, this method simulate the behavior of pre-attentive visual search. The algorithm analyzes the log spectrum of each image and obtain the spectral residual. Then transform the spectral residual to spatial domain to obtain the saliency map, which suggests the positions of proto-objects.
based on [https://www.researchgate.net/publication/221364530_Saliency_Detection_A_Spectral_Residual_Approach “Saliency detection: A spectral residual approach.”]

[FineGrained] - This method calculates saliency based on center-surround differences. High resolution saliency maps are generated in real time by using integral images. Based on [http://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=4CC3F597B5F4F9EC4E4D71817098C3E2?doi=10.1.1.700.4706&rep=rep1&type=pdf Human detection using a mobile platform and novel features derived from a visual saliency mechanism].

After computing a saliency map with one of the above methods, we calculate the mean pixel value of this saliency map to give a rough measure of its strength, the value is normalized dividing it by 255.
Using the argument -t=2 we have acess in a window to the original image, saliency map and binary map.

*rank_sum [0.0-1.0]
We experimented to combine some objective measures in a effort to compute a general objective quality measure.

*fps - [1-60]
Frames per second

====Optical flow====
Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement of object or camera. It is a 2D vector field where each vector is a displacement vector showing the movement of points from first frame to second.
Optical flow works on two main assumptions: the pixel intensities of an object do not change between consecutive frames and neighboring pixels have similar motion.
OpenCV provides a algorithm to find dense optical flow. It computes the optical flow for all the points in the frame. It is based on Gunner Farneback's algorithm which is explained in "Two-Frame Motion Estimation Based on Polynomial Expansion" by Gunner Farneback in 2003.
For the sake of performance we compute motion vectors in a 5 by 5 grid, this way we only have to deal with a fraction  of around 1/25 of total pixels present in the image.
After we compute the absolute total flow and signed total flow per frame used to compute the total video flow and average flow per frame.
We also measure optical flow only on a border of the frame so the main subject movement dont interfere with our angle measures to increase accuracy of shake detection.

Using the argument -t=3 we have acess in a window to the motion field were we can see superimposed the average motion vector computed using all motion vectors or only the ones inside a border.

*shackiness [0.0-1.0]
We compare the angle between two subsequent motion vectors(computed from three consecutive video frames), if the angle is above a certain threshold(90º) we mark the transition between frames as "shaky", then compute the ratio between this "shaky" transitions and the total frames of the video.

*motion_mag [0.0-1.0]
Motion magnitude is the length of the average motion vector, we can use it as a global measure of change introduced by motion  between two frames of video.

====Background subtraction====
Background subtraction or Foreground Detection  is a technique where an image's foreground is extracted for further processing. It is widely used for detecting moving objects in videos or cameras. The rationale in the approach is that of detecting the moving objects from the difference between the current frame and a reference frame, often called “background image”, or “background model”.
Background subtraction is generally based on a static background hypothesis which is often not applicable in real environments. With indoor scenes, reflections or animated images on screens lead to background changes. In a same way, due to wind, rain or illumination changes brought by weather, static backgrounds methods have difficulties with outdoor scenes.

From the video analysis [http://docs.opencv.org/3.0-beta/modules/video/doc/video.html section] of OpenCV, we use the class BackgroundSubtractor, included in the Motion Analysis and Object Tracking algorithms. The class is only used to define the common interface for the whole family of background/foreground segmentation algorithms from wich we use two specific algorithms that take shadows in account, a shadow is detected if a pixel is a darker version of the background. See [Prati, Mikic, Trivedi and Cucchiarra, Detecting Moving Shadows..., IEEE PAMI,2003]

[BackgroundSubtractorKNN] - The class implements the K-nearest neigbours background subtraction described in [ Z.Zivkovic, F. van der Heijden. “Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction”, Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006] . Very efficient if number of foreground pixels is low.
Link: [http://www.zoranz.net/Publications/zivkovicPRL2006.pdf] - [http://escholarship.org/uc/item/2kj2g2f7]

[BackgroundSubtractorMOG2] - The class implements the Gaussian mixture model background subtraction described in [Zivkovic. “Improved adaptive Gausian mixture model for background subtraction”, International Conference Pattern Recognition, UK, August, 2004] - [http://www.zoranz.net/Publications/zivkovic2004ICPR.pdf Link], and Z.Zivkovic, F. van der Heijden. “Efficient Adaptive Density Estimation per Image Pixel for the Task of Background Subtraction”, Pattern Recognition Letters, vol. 27, no. 7, pages 773-780, 2006.- [http://www.zoranz.net/Publications/zivkovicPRL2006.pdf Link ]

Using the argument -t=5 we have acess in a window to the computed foreground, background and shadows.
Using the argument -b we can 0 - turn background subtraction off, or choose algorithm 1-knn, 2-mog2.

*fg_area
Foreground area

*shadow_area
From [http://www.hitl.washington.edu/research/knowledge_base/virtual-worlds/EVE/III.A.1.c.DepthCues.html]
When we know the location of a light source and see objects casting shadows on other objects, we learn that the object shadowing the other is closer to the light source. As most illumination comes downward we tend to resolve ambiguities using this information.

*bg_area
background area

*camera_move
Percentage of frames where foreground area is bigger then 80% of total area available, this indicates high probability of camera movement occurance.

*focus_diff
Difference of focus between foreground and background. We try to detect low depth of field, the region of interest should be in sharp focus and the background blurred, this is a attribute that can boost aesthetic perceived from media or attention arouse.

*entropy [0.0 – 1.0]
Entropy is a quantity normally used to describe the business of an image, in other words, the volume of data that needs to be encoded by a compression algorithm. A completly flat image, without any proeminent pixel have zero entropy and can be compressed to a very small size. On the other side an image with high entropy, full of details and pixel contrast will result in a very large encoded file size.
