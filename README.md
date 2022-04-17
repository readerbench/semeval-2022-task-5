# UPB at Semeval 2022 Task 5

UPB contribution to Semeval 2022 Task 5: MAMI - Multimedia Automatic Misogyny Identification.



## Requirements
* Visual features extractor https://github.com/MILVLG/bottom-up-attention.pytorch
* Visual sentiment analysis https://github.com/fabiocarrara/visual-sentiment-analysis.git
* NVIDIA APEX https://github.com/NVIDIA/apex
* UNITER pretrained model  
https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-base.pt
https://acvrpublicycchen.blob.core.windows.net/uniter/pretrained/uniter-large.pt



## Installing APEX
```bash
        export CUDA_HOME=/usr/local/cuda-10.2
        git clone https://github.com/NVIDIA/apex
        pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```

## Extract Features with bottom-up-attention
```bash
        python extract_features.py --mode caffe --config-file ../src/bua-configs/extract-bua-caffe-r152-box-only.yaml --image-dir ../data/TRAINING/ --out-dir ../data/own_features_bbox-152/ --resume
        python extract_features.py --mode caffe --config-file ../src/bua-configs/extract-bua-caffe-r152-gt-bbox.yaml --image-dir ../data/TRAINING/ --gt-bbox-dir ../data/own_features_bbox-152/ --out-dir ../data/own_features_FasterRCNN-152/ --resume
```

## Extract Visual Sentiment Features
Get the repository into "visual_sentiment" folder, download the pretrained model into the 
"visual_sentiment/model" folder
```bash
        python save_embeddings.py --image-folder=../data/TRAINING/img --model-file=models/vgg19_finetuned_all.pth --output-folder=../data/TRAINING/sentiment
```

## Graph creation 
input csv should be in the form
'id', 'meme_text','object_ids'

where object_ids is a list of detected objects in the file 

```bash
python create_graph.py --bert-model=bert-base-cased --dataset=object_cooccurrences.csv --data-folder=data/ --output-file=graph_arr.pkl --remove-numeric --text-window-stride=1 --multithread --num-workers=12
```
