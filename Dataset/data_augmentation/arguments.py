import argparse
import collections

parser = argparse.ArgumentParser(
    description='Arguments to control artificial image generation.')

parser.add_argument('--image_dimension', default=[480,640], type=list, required=False,
                    help='Dimension of the real images.')

parser.add_argument('--num_scales', default='randomize', type=str, required=False,
                    help='Number of scales including original object scale.')

parser.add_argument('backgrounds_path', type=str,
                    help='Path to directory where the background images are located.')

parser.add_argument('image_path', type=str,
                    help='Path to directory where real images are located.')

parser.add_argument('label_path', type=str,
                    help='Path to directory where labels are located.')

parser.add_argument('--real_img_type', default='.jpg', type=str, required=False,
                    help='The format of the real image.')

parser.add_argument('--min_obj_area', default=20, type=int, required=False,
                    help='Minimum area in percentage allowed for an object in image space.')

parser.add_argument('--max_obj_area', default=70, type=int, required=False,
                    help='Minimum area in percentage allowed for an object in image space.')

parser.add_argument('--save_label_preview', default=False, type=bool, required=False,
                    help='Save image+label in single image for preview.')

parser.add_argument('--save_obj_det_label', default=False, type=bool, required=False,
                    help='Save object detection labels in csv files.')

parser.add_argument('--save_mask', default=False, type=bool, required=False,
                    help='Save images showing the segmentation mask.')

parser.add_argument('image_save_path', type=str,
                    help='Path where the generated artificial image needs to be saved.')

parser.add_argument('label_save_path', type=str,
                    help='Path where the generated segmentation label needs to be saved.')

parser.add_argument('--obj_det_save_path', default=None, type=str, required=False,
                    help='Path where object detection labels needs to be saved')

parser.add_argument('--mask_save_path', default=None, type=str, required=False,
                    help='Path where segmentation masks needs to be saved')

parser.add_argument('--start_index', default=0, type=int, required=False,
                    help='Index from which image and label names should start.')

parser.add_argument('--name_format', default='%05d', type=str, required=False,
                    help='The format for image file names.')

parser.add_argument('--remove_clutter', default=True, type=bool, required=False,
                    help='Remove images cluttered with objects.')

parser.add_argument('--num_images', default=20, type=int, required=False,
                    help='Number of artificial images to generate.')

parser.add_argument('--max_objects', default=10, type=int, required=False,
                    help='Maximum number of objects allowed in an image.')

parser.add_argument('--num_regenerate', default=100, type=int, required=False,
                    help='Number of regeneration attempts of removed details dict.')

parser.add_argument('--min_distance', default=70, type=int, required=False,
                    help='Minimum pixel distance required between two objects.')

parser.add_argument('--max_occupied_area', default=0.8, type=float, required=False,
                    help='Maximum object occupancy area allowed.')


LABEL_DEF_MATLAB={'f20_20_B': 1, 's40_40_B': 2, 'f20_20_G': 3,
                  's40_40_G': 4,  'm20_100': 5, 'm20': 6, 'm30': 7,
                  'r20': 8, 'bearing_box_ax01': 9, 'bearing': 10, 'axis': 11,
                  'distance_tube': 12, 'motor': 13, 'container_box_blue': 14,
                  'container_box_red': 15, 'bearing_box_ax16': 16,
                  'em_01': 17, 'em_02': 18, 'background': 19}

args = parser.parse_args()

if args.save_obj_det_label and args.obj_det_save_path is None:
    parser.error('Path to save object detection labels is also required.')

if args.save_mask and args.mask_save_path is None:
    parser.error('Path to save segmentation masks is also required.')


class GeneratorOptions(
    collections.namedtuple('GeneratorOptions', [
        'image_dimension',
        'num_scales',
        'backgrounds_path',
        'image_path',
        'label_path',
        'real_img_type',
        'min_obj_area',
        'max_obj_area',
        'save_label_preview',
        'save_obj_det_label',
        'save_mask',
        'image_save_path',
        'label_save_path',
        'obj_det_save_path',
        'mask_save_path',
        'start_index',
        'name_format',
        'remove_clutter',
        'num_images',
        'max_objects',
        'num_regenerate',
        'min_distance',
        'max_occupied_area',
    ])):
    """Immutable class to hold artificial image generation options."""

    __slots__ = ()

    def __new__(cls):

        return super(GeneratorOptions, cls).__new__(
            cls, args.image_dimension, args.num_scales, args.backgrounds_path,
            args.image_path, args.label_path, args.real_img_type, args.min_obj_area,
            args.max_obj_area, args.save_label_preview, args.save_obj_det_label,
            args.save_mask, args.image_save_path, args.label_save_path,
            args.obj_det_save_path, args.mask_save_path, args.start_index,
            args.name_format, args.remove_clutter, args.num_images, args.max_objects,
            args.num_regenerate, args.min_distance, args.max_occupied_area)