# Studio della posa di neonati pre-termine e regressione dei loro movimenti mediante l’utilizzo di reti con architettura ad alta risoluzione


## (Rete utilizzata HRNet-DEKR: Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression)

## Introduzione
Il monitoraggio dei movimenti e la stima della posa nei neonati pre-termine sono di fondamentale importanza in quanto forniscono informazioni sullo sviluppo neurologico, aiutando nella diagnosi precoce di paralisi celebrale.		

## Quick start
### Installazione
1. Clonare questo repository, la directpry di clonazione verrà indicata con ${POSE_ROOT}.
2. Installare le dipendenze:
   ```
   pip install -r requirements.txt
   ```
3. Scaricare i [file zip del model e dell'output](https://univpm-my.sharepoint.com/:f:/g/personal/s1114440_studenti_univpm_it/ErvBPJ8CsxJLvIrAtFRsF4wBvU6ki1k9TsS6e7AGRQbzyA?e=YQaoXC) e decomprimerli nella home directory. La home directory dovrà contenere le seguenti cartelle:
```
${POSE_ROOT}
├── annotation_csv
├── application
├── data
├── model
├── experiments
├── lib
├── tools 
└── output
```

### Training e Testing

#### Trainig e validation del modello pre addestrto su crowdpose sul dataset presente in data/babypose/json
```
python tools/train.py  \
    --cfg experiments/babypose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_babypose_x300.yaml \
    MODEL.PRETRAINED model/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth
```

#### Testing del modello addestrato con il dataset image split
```
python tools/valid.py  \
    --cfg experiments/babypose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_babypose_x300.yaml \
    TEST.MODEL_FILE output/baby_pose_kpt_carbon_image_split/hrnet_dekr/w32_4x_reg03_bs10_512_adam_lr1e-3_babypose_x300/model_best.pth.tar \
    DATASET.MAX_NUM_PEOPLE 1
```
#### Testing del modello addestrato con il dataset patient split
```
python tools/valid.py  \
    --cfg experiments/babypose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_babypose_x300.yaml \
    TEST.MODEL_FILE output/baby_pose_kpt_carbon_patient_split/hrnet_dekr/w32_4x_reg03_bs10_512_adam_lr1e-3_babypose_x300/model_best.pth.tar \
    DATASET.MAX_NUM_PEOPLE 1
```
#### Testing del modello addestrato con il dataset crowdpose senza fine tuning su babypose
```
python tools/valid.py  \
    --cfg experiments/babypose/w32/w32_4x_reg03_bs10_512_adam_lr1e-3_babypose_x300.yaml \
    TEST.MODEL_FILE model/pose_crowdpose/pose_dekr_hrnetw32_crowdpose.pth \
    DATASET.MAX_NUM_PEOPLE 1
```
#### Utilizzo dell'applicazione demo
```
python tools/inference_demo.py --cfg experiments/crowdpose/inference_demo_crowdpose.yaml \
    --videoFile path_to_your_video_file \
    --outputDir output \
    --visthre 0.3 \
	--inferenceFps 5\
	TEST.MODEL_FILE model/pose_babypose/model_image_split.pth.tar
```
### Riconoscimenti
Il nostro codice è basato principalmente su  [HRNet-DEKR](https://github.com/HRNet/DEKR). 

Per maggiori informazioni o per richiedere il dataset contrattare [Claudio](mailto:s1114440@studenti.univpm.it) o [Matteo Lorenzo](mailto:s1114138@studenti.univpm.it)




