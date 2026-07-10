public = {
    'num_classes': 2,
    'num_keypoints': 17,
    'eval_spatial_size': [640, 640],
    'dataset_type': 'COCO',
    'iou_types': ['keypoints'],
}

private = {
    'model': {
        'num_levels': 3,
        'hidden_dim': 256,
        'num_select': 60,
        'num_queries': 60,
        'num_classes': "${public.num_classes}",
        'feat_strides': [8, 16, 32],
        'num_keypoints': '${public.num_keypoints}',
        'eval_spatial_size': '${public.eval_spatial_size}',
    }
}