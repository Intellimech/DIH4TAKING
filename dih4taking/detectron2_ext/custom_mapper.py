# def custom_mapper(dataset_dict):
#     dataset_dict = copy.deepcopy(dataset_dict)
#     image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")

#     augs = [
#         T.RandomBrightness(0.9, 1.1),
#         T.RandomSaturation(intensity_min=0.9, intensity_max=1.1),
#         T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
#         T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
#         RandomToGray(prob=0.5),
#     ]

#     image, transforms = T.apply_transform_gens(augs, image)
#     dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

#     annos = [
#         detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dict.pop("annotations")
#         if obj.get("iscrowd", 0) == 0
#     ]
#     instances = detection_utils.annotations_to_instances(annos, image.shape[:2], mask_format="bitmask")
#     dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

#     return dataset_dict