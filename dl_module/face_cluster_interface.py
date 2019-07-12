import logging

import numpy as np
from sklearn.cluster import DBSCAN


def custom_metric(x1, x2):
    # return np.sum(np.square(np.subtract(x1, x2)))
    return 1. - np.dot(x1, x2.T)


def cluster_face_func(face_data, user_id, face_id_label_dict):
    result = []
    try:
        features = [i["face_feature"] for i in face_data]
        nrof_images = len(features)
        db = DBSCAN(eps=0.44, min_samples=5, metric=custom_metric)
        db.fit(features)
        labels = db.labels_
        not_find_index_set = set()
        for index in range(nrof_images):
            if {index} & not_find_index_set:
                continue
            current_label = int(labels[index])
            current_index_set = set(np.nonzero(labels == current_label)[0])
            not_find_index_set = not_find_index_set | current_index_set
            if current_label == -1:
                face_res = []
                for img_index in current_index_set:
                    old_group = face_id_label_dict.get(str(img_index), -1)
                    if old_group == current_label:
                        continue
                    face_id_label_dict[str(img_index)] = -1
                    face_info = face_data[img_index]
                    face_res.append({
                        'fileName': face_info['face_id'].rsplit("_", 1)[0],
                        'emotionStr': face_info['emotionStr'],
                        'faceImgCoordinate': face_info['face_box'],
                        'oldGroupId': old_group
                    })
                if face_res:
                    result.append({
                        'groupId': -1,
                        'imgInfo': face_res,
                        'faceImgUrl': "face_cluster_data/{}/face_images/{}.jpg".format(user_id, face_data[0]['face_id'])
                    })
            else:
                group_id = int(min(current_index_set))
                face_tmp = []
                img_set = set()
                for img_index in current_index_set:
                    face_info = face_data[img_index]
                    face_tmp.append([img_index, {
                        'fileName': face_info['face_id'].rsplit("_", 1)[0],
                        'emotionStr': face_info['emotionStr'],
                        'faceImgCoordinate': face_info['face_box'],
                    }])
                    img_set.add(face_info['face_id'])
                if len(img_set) > 3:
                    face_res = []
                    for face in face_tmp:
                        old_group = face_id_label_dict.get(str(face[0]), "")
                        if old_group != group_id:
                            face_id_label_dict[str(face[0])] = group_id
                            face_content = face[1]
                            face_content['oldGroupId'] = old_group
                            face_res.append(face_content)
                    if face_res:
                        result.append({
                            'groupId': group_id,
                            'imgInfo': face_res,
                            'faceImgUrl': "face_cluster_data/{}/face_images/{}.jpg".format(
                                user_id, face_data[group_id]['face_id'])
                        })
    except Exception as e:
        logging.exception(e)
    finally:
        return result, face_id_label_dict
