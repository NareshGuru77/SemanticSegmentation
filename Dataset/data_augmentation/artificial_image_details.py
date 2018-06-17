from data_augmentation import arguments

if __name__ == '__main__':

    generator_options = arguments.GeneratorOptions()

    print(generator_options.image_dimension)
    print(arguments.LABEL_DEF_MATLAB)
    print(generator_options.backgrounds_path)
    print(type(generator_options.save_mask))

