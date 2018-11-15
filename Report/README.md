# Abstract:

Semantic segmentation is a pixel-wise classification problem where a convolutional neural network (CNN) is trained to assign a class to every pixel in an input image. The output produced is a 2D segmentation map through which we can infer what objects are present in the image, their corresponding pixel locations, and object boundaries. The need to classify every pixel calls for dense annotation of every label image. Consequently, creating manual annotations for a new dataset is expensive. Nevertheless, this project creates a limited dataset consisting of 540 images with manual annotations. In order to improve the diversity and robustness of this limited dataset, it is augmented with artificial images. Four different variants of the dataset called "atWork_full", "atWork_size_invariant", "atWork_similar_shapes", and "atWork_binary" are created based on the similarity in size, shape and color of objects. We show that the dataset variants which combine similar objects lead to higher Mean Intersection Over Union(mIOU). 

The DeepLabv3+ segmentation model, in addition to its focus on improving mIOU, also uses MobileNetv2 and Xception networks as network backbones (encoders) to improve resource efficiency. The MobileNetv2 network backbone requires an inference time of 0.98 seconds per image, occupies 8.7 MB disk memory, and achieves a mIOU of 77.47 % on the "atWork_full" dataset variant. In contrast, the Xception network backbone achieves 89.63 % percent on the same dataset variant but is less resource efficient as it requires 5.53 seconds inference time and 165.6 MB disk memory. The quantized versions of the two network backbones are shown to be more efficient in terms of disk memory occupied. However, roughly 9 % and 2 % average drop in mIOU across all four dataset variants is observed for the quantized versions of MobileNetv2 and Xception network backbones respectively.


The table of contents:
* Introduction
    * Motivation
        * Potential applications
    * Challenges and difficulties
        * Labeling cost
        * Context knowledge
    * Problem statement
* State of the Art
    * Improving accuracy
        * Fully Convolutional Networks
    * Accuracy and resource efficiency
        * SegNet
        * ENet
        * ICNet
    * Compressing DCNNs 
        * Pruning CNNs
        * Quantizing CNNs
* Convolutional Neural Networks and Semantic Segmentation
    * Artificial Neural Networks 
    * Convolutional Neural Networks
        * CNN Architecture 
    *  CNNs for Semantic Segmentation
*  Methodology
    * DeepLab
    * DeepLabv2
    * DeepLabv3
    * DeepLabv3+
    * MobileNetv2
    * Xception
    * Quantization
*  Dataset creation
    * Overview of the dataset
    *  Artificial image generation algorithm 
        * Motivation
        * Process of artificial image generation 
        * Generator options 
        * Sample results
        * Downloading background images
        * Notable features of the artificial image generator 
        * Artificial images for each dataset split
    * Creation of dataset variants
        * Motivation
        * Dataset variants 
        * White backgrounds dataset
    * Data analysis
        * Surface area of the objects
        * Percentage of pixels and class count
    * Meta-data of the dataset
    * Possible directions of improvement
* Experimental Evaluation
    * About the metrics 
    * Comparing dataset variants
    * Comparing DeepLabv3+ backbones
    * Training with different data
    * Comparing individual classes
        * Confusion matrix
        * Class IOUs
    * Comparing learning rate policies
    * Effects of class balancing
    * Effects of quantizing the inference graph
    * Discussions
* Conclusions
    * Contributions
    * Lessons learned 
    * Future work
* Appendix A Further details regarding the Dataset
    * Annotation tools
    * Description of the labeling process 
    * Search keywords for background images 
    * Generator option details
* Appendix B Sample predictions
* Appendix C Hyperparameters
* References
