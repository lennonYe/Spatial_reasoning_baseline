## Baseline code for spatial reasoning project
For VGG,Vit and resnet folder, to use the pretrained checkpoints from the CSR repo, you will need to download the checkpoints from 
[here](https://prior-model-weights.s3.us-east-2.amazonaws.com/embodied-ai/csr/checkpoints.tar.gz)

Make sure to extract to root of VGG,Vit and resnet folder  

  

If you are creating a conda environment using environment.yml, please use:
```
conda env create --file environment.yml --name csr
```

You will also need to download Detectron2 separately by following their [guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html#build-detectron2-from-source).
