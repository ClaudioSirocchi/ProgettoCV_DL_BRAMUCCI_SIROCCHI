import os
import random

import pandas as pd
import json

# Dataset split
TRAIN_PERC = 0.75


def get_pat_id(file_name):
    pat_id = file_name[file_name.find('_') + 1:]
    pat_id = pat_id.replace('.csv', '')
    return int(pat_id.replace('pz', ''))


def get_image_id(pat_id, image):
    str_val = str(pat_id) + image.split('_')[0].rjust(5, '0')

    return (int(str_val))


def write_json_file(dataframe, filename, rotated):
    with open('../data/babypose/json/bbox.json', 'r') as bbox_file:
        bbox_data = json.load(bbox_file)


    keypoints_names = [
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "head",
        "neck"
    ]
    coco_anno = {
        "info": {
            "description": "CrowdPose annotations modified by Matteo Lorenzo Bramucci.",
            "year": 2023,
            "date_created": "2023/06/01"
        },
        "categories": [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": [
                    "left_shoulder",
                    "right_shoulder",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                    "head",
                    "neck"
                ],
                "skeleton": [
                    [
                        12,
                        13
                    ],
                    [
                        13,
                        0
                    ],
                    [
                        13,
                        1
                    ],
                    [
                        0,
                        2
                    ],
                    [
                        2,
                        4
                    ],
                    [
                        1,
                        3
                    ],
                    [
                        3,
                        5
                    ],
                    [
                        13,
                        7
                    ],
                    [
                        13,
                        6
                    ],
                    [
                        7,
                        9
                    ],
                    [
                        9,
                        11
                    ],
                    [
                        6,
                        8
                    ],
                    [
                        8,
                        10
                    ]
                ]
            }
        ],
        "images": [],
        "annotations": [],
    }
    counter = 0
    dataframe.reset_index()
    for index, row in dataframe.iterrows():
        counter += 1
        coco_anno['images'].append({
            "file_name": 'pz' + str(row['patient']) + '/' + row['image'],
            "id": get_image_id(row['patient'], row['image']),
            "height": 480,
            "width": 640,
            "crowdIndex": 0.0
        })
        the_keypoints = []
        num_keypoints = 0
        for name in keypoints_names:
            if name == "left_wrist":
                name = "left_hand"
            if name == "right_wrist":
                name = "right_hand"
            if name == "left_ankle":
                name = "left_foot"
            if name == "right_ankle":
                name = "right_foot"
            vals = list(map(int, row[name].split(',')))
            if vals == [-1, -1]:
                the_keypoints.append(0)
                the_keypoints.append(0)
                the_keypoints.append(0)
            else:
                if rotated:
                    the_keypoints.append(640 - vals[0])
                    the_keypoints.append(480 - vals[1])
                else:
                    the_keypoints.append(vals[0])
                    the_keypoints.append(vals[1])
                the_keypoints.append(2)
                num_keypoints += 1
        bbox = bbox_data['pz' + str(row['patient'])]
        if rotated:
            the_bbox = [640 - (bbox[0] + bbox[2]), 480 - (bbox[1] + bbox[3]), bbox[2], bbox[3]]
        else:
            the_bbox = bbox
        coco_anno['annotations'].append({
            "num_keypoints": num_keypoints,
            "iscrowd": 0,
            "keypoints": the_keypoints,
            "image_id": get_image_id(row['patient'], row['image']),
            "category_id": 1,
            "id": counter,
            "bbox": the_bbox
        })


    # convert into JSON:
    y = json.dumps(coco_anno)

    with open(f'../data/babypose/json/{filename}', 'w') as f:
        f.write(y)

def image_exists(patient, image):
    return os.path.exists(f"../data/babypose/images/pz{patient}/{image}")


def convert_image_split(annotations_path, rotated=False, reduced_to=0):
    print('convert_image_split')
    is_trainval = 'trainval' in annotations_path
    annotations = pd.DataFrame()

    # carico tutti i file .csv in un unico dataframe ad aggiungo una colonna che per ogni riga indichi il riferimento al paziente
    first = True
    for csvfile in sorted(os.listdir(annotations_path)):
        if not csvfile.startswith('.'):
            # Read the CSV file
            if first:
                annotations = pd.read_csv(annotations_path + '/' + csvfile, delimiter=';', encoding='utf-8')
                annotations.insert(0, 'patient', get_pat_id(csvfile))
                first = False
            else:
                dati = pd.read_csv(annotations_path + '/' + csvfile, delimiter=';', encoding='utf-8')
                dati.insert(0, 'patient', get_pat_id(csvfile))
                annotations = annotations._append(dati)

    annotations.reset_index(drop=True, inplace=True)

    # controllo che le righe del dataframe corrispondano a immagini esistenti
    # altrimenti le rimuovo
    # aggiungo una colonna che contiene True o False dipendentemente dal fatto che l'immagine esiste o meno
    annotations['image_exists'] = annotations.apply(lambda x: image_exists(x['patient'], x['image']), axis=1)


    print(f'Elementi totali delle annotazioni: {len(annotations)}')
    # elimino dal dataframe le righe con image_exists == False
    annotations.drop(annotations[annotations['image_exists'] == False].index, inplace=True)
    print(f'Elementi accettabili delle annotazioni: {len(annotations)}')

    # ora sono sicuro di operare su di un dataframe consistente

    # randomizzo la posizione delle righe con un seed noto così da rendere riproducibile l'esperimento
    shuffled = annotations.sample(frac=1, random_state=123).reset_index(drop=True)

    if reduced_to != 0:
        shuffled = shuffled.iloc[:reduced_to]
        shuffled.reset_index(drop=True, inplace=True)
        print(f'Richiesta riduzione per prove a {len(shuffled)} elementi.')

    samples_num = len(shuffled)
    if is_trainval:
        train_size = int(samples_num * TRAIN_PERC)          # calcolo il numero di righe da includere nel train set
        valid_size = samples_num - train_size        # calcolo il numero di righe da includere nel valid set
        rows_train = shuffled.iloc[:train_size]  # estraggo il train set
        rows_valid = shuffled.iloc[train_size:train_size + valid_size]  # estraggo il valid set
        write_json_file(rows_train, 'babypose_train.json', rotated)
        write_json_file(rows_valid, 'babypose_val.json', rotated)
        print('Annotazioni per trainval')
        print(f'Train size: {train_size}, Valid size: {valid_size}')
        print(f'Total:  {train_size + valid_size}')
    else:
        test_size = samples_num
        rows_test = shuffled
        write_json_file(rows_test, 'babypose_test.json', rotated)
        print('Annotazioni per test')
        print(f'Test size: {test_size}')
        print(f'Total:  {test_size}')

    print('\n')



def convert_patient_split(annotations_path, rotated=False, reduced_to=0):
    print('convert_patient_split')
    is_trainval = 'trainval' in annotations_path

    the_patient_list = get_patient_list(annotations_path)
    patients_num = len(the_patient_list)

    if is_trainval:
        train_size = int((patients_num * TRAIN_PERC) + 0.5)  # calcolo il numero di righe da includere nel train set
        valid_size = patients_num - train_size  # calcolo il numero di righe da includere nel valid set
        train_patients = the_patient_list[:train_size]  # estraggo il train set
        valid_patients = the_patient_list[train_size:train_size + valid_size]  # estraggo il valid set
        print('Generazione annotazioni per train')
        convert_pro(annotations_path, train_patients, type='train', rotated=rotated, reduced_to=reduced_to)
        print('Generazione annotazioni per validazione')
        convert_pro(annotations_path, valid_patients, type='val', rotated=rotated, reduced_to=reduced_to)
        print(f'Patient Train size: {train_size}, Valid size: {valid_size}')
        print(f'Total:  {train_size + valid_size}')
    else:
        test_size = patients_num  # calcolo il numero di righe da includere nel test set
        test_patients = the_patient_list  # estraggo il test set
        print('Generazione annotazioni per test')
        convert_pro(annotations_path, test_patients, type='test', rotated=rotated, reduced_to=reduced_to)
        print(f'Patient Test size: {test_size}')
        print(f'Total:  {test_size}')

    print('\n')



def convert_pro(annotations_path, annotations_list, type='train', rotated=False, reduced_to=0):
    annotations = pd.DataFrame()

    # carico tutti i file .csv in un unico dataframe ad aggiungo una colonna che per ogni riga indichi il riferimento al paziente
    first = True
    for csvfile in annotations_list:
        if not csvfile.startswith('.'):
            # Read the CSV file
            if first:
                annotations = pd.read_csv(annotations_path + '/' + csvfile, delimiter=';', encoding='utf-8')
                annotations.insert(0, 'patient', get_pat_id(csvfile))
                first = False
            else:
                dati = pd.read_csv(annotations_path + '/' + csvfile, delimiter=';', encoding='utf-8')
                dati.insert(0, 'patient', get_pat_id(csvfile))
                annotations = annotations._append(dati)

    annotations.reset_index(drop=True, inplace=True)

    # controllo che le righe del dataframe corrispondano a immagini esistenti
    # altrimenti le rimuovo
    # aggiungo una colonna che contiene True o False dipendentemente dal fatto che l'immagine esiste o meno
    annotations['image_exists'] = annotations.apply(lambda x: image_exists(x['patient'], x['image']), axis=1)


    print(f'Elementi totali delle annotazioni: {len(annotations)}')
    # elimino dal dataframe le righe con image_exists == False
    annotations.drop(annotations[annotations['image_exists'] == False].index, inplace=True)
    print(f'Elementi accettabili delle annotazioni: {len(annotations)}')

    # ora sono sicuro di operare su di un dataframe consistente

    # randomizzo la posizione delle righe con un seed noto così da rendere riproducibile l'esperimento
    shuffled = annotations.sample(frac=1, random_state=123).reset_index(drop=True)

    if reduced_to != 0:
        shuffled = shuffled.iloc[:reduced_to]
        shuffled.reset_index(drop=True, inplace=True)
        print(f'Richiesta riduzione per prove a {len(shuffled)} elementi.')

    samples_num = len(shuffled)
    print(f'{type} size: {samples_num}')

    # scrittura delle annotazioni nella cartella prevista e nel formato richiesto
    write_json_file(shuffled, f'babypose_{type}.json', rotated)


"""
Nel nuovo concetto di creazione delle serie di immagini di train, validazione e test
non si usa più l'idea di prendere percentuali di immagini dall'insieme di tutte le immagini
ma percentuali di pazienti dall'insieme di tutti i pazienti
"""

def get_patient_list(annotations_path):
    the_list = [f for f in os.listdir(annotations_path) if not f.startswith('.')]
    random.seed(123)
    random.shuffle(the_list)
    return the_list



def main(type):
    #convert_patient_split('../annotations_csv', True, 0)
    if type == 'image_split':
        convert_image_split('../annotations_csv/trainval', True, 0)
        convert_image_split('../annotations_csv/test', True, 0)
    else:
        convert_patient_split('../annotations_csv/trainval', True, 0)
        convert_patient_split('../annotations_csv/test', True, 0)

if __name__ == '__main__':
    main('patient_split')
