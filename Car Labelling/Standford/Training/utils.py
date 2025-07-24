from resize_with_pad import ResizeWithPad
from torchvision import datasets, models, transforms
    
class Utils():

    def get_transformations(self, params):
        train_transforms = self.__get_transformation(params['train_transforms'])
        val_transforms = self.__get_transformation(params['val_transforms'])

        return train_transforms, val_transforms
    
    
    def __get_transformation(self, transforms_params):
        transforms_l = []

        if transforms_params['resized']:
            if transforms_params['padding']:
                transforms_l.append(ResizeWithPad(target_size=(transforms_params['new_image_size'], transforms_params['new_image_size'])))
            else:
                transforms_l.append(transforms.Resize((transforms_params['new_image_size'],transforms_params['new_image_size']), interpolation=int(transforms_params['interpolation'])))

        transforms_l.append(transforms.ToTensor())

        if transforms_params['normalized']:
            transforms_l.append(transforms.Normalize(transforms_params['normalized_mean'], transforms_params['normalized_std']))

        print(transforms_l)
        return transforms.Compose(transforms_l)
    

    

