# What the artificial image generator does?
The artificial image generator makes use of real world images of the objects in the dataset, their corresponding semantic annotations and background images downloaded from the internet to generate artificial images to augment the dataset. The generated artificial images contain objects at different scales and at random locations imrproving the variety of the dataset.

# How does the artificial image generator work?
The process of artificial image generation is described in the [final research report](..../Report/GurulinganNK-RnD-Report.pdf) in the subsection called "Process of artificial image generation".

# Notable features:
* Both semantic segmentation and object detection labels.
* The labels can be visualized in three different ways called 1. mask, 2. overlay, 3. preview.
1. Mask:
![sample result 1](..../Report/images/eg_mask.png)
2. Overlay:
![sample result 1](..../Report/images/eg_overlay.png)
3. Preview:
![sample result 1](..../Report/images/sample_white_1.png)

# Generator options:
The various arguments which can be tweaked to control the way in which artificial images are generated.

     **Generator options**                            Description                                         
  --------------------------- ------------------------------------------------------------ -- -- -- -- -- --
           **mode**                 1: Generate artificial images; 2: Save visuals.                       
     **image_dimension**                    Dimension of the real images.                                
        **num_scales**            Number of scales including original object scale.                      
     **backgrounds_path**     Path to directory where the background images are located.                 
        **image_path**             Path to directory where real images are located.                      
        **label_path**               Path to directory where labels are located.                         
    **obj_det_label_path**          Path to directory where the object detection csv labels are located.                  
      **real_img_type**                    The format of the real image.                                
      **min_obj_area**                 Minimum area in percentage allowed for an object in image space.                      
    **max_obj_area**                 Maximum area in percentage allowed for an object in image space.
    **save_label_preview**         Save image+label in a single image for preview.                       
    **save_obj_det_label**           Save object detection labels in csv files.                         
        **save_mask**                 Save images showing the segmentation mask.                         
       **save_overlay**               Save segmentation label overlaid on image.                         
     **overlay_opacity**             Opacity of the label on the overlaid image.                         
     **image_save_path**          Path where the generated artificial image needs to be saved.                  
     **label_save_path**         Path where the generated segmentation label needs to be saved.                 
    **preview_save_path**           Path where preview image needs to be saved.                         
    **obj_det_save_path**      Path where object detection labels needs to be saved.                    
     **mask_save_path**           Path where segmentation masks needs to be saved.                      
    **overlay_save_path**          Path where overlaid images needs to be saved.                        
       **start_index**              from which image and label names should start.                       
       **name_format**                     The format for image file names.                              
      **remove_clutter**                Remove images cluttered with objects.                            
        **num_images**                 Number of artificial images to generate.                          
       **max_objects**              Maximum number of objects allowed in an image.                       
      **num_regenerate**      Number of regeneration attempts of removed object details.                 
       **min_distance**          Minimum pixel distance required between two objects.                    
    **max_occupied_area**              Maximum object occupancy area allowed.                           
       **scale_ranges**       Can be used to change the zoom range of specific objects. 

# Sample results:
* ![sample result 1](..../Report/images/sample_result_1.png)
* ![sample result 2](..../Report/images/sample_result_2.png)
* ![sample result 3](..../Report/images/sample_result_3.png)