# SPARK project
Welcome to the world-life challenge! <br>

## The Goal

Our goal is to develop a model to diagnose [Parkinson's](https://www.who.int/news-room/fact-sheets/detail/parkinson-disease) disease from Apple Watch data (rotation and accelaration) and questionnaire data. <br>
Here are the [pitch slides](https://docs.google.com/presentation/d/1gdGfYFbZ1nauL5MrTryWylVPRaeiTgPQRvsO12wWe0M/edit?slide=id.g377c5352b2f_0_2530&pli=1#slide=id.g377c5352b2f_0_2530).

## The Data
The PADS dataset is available on PhysioNet

Here are the instructions to download:
Download the ZIP file (735.0 MB)
Download the files using your terminal:
wget -r -N -c -np https://physionet.org/files/parkinsons-disease-smartwatch/1.0.0/
Download the files using AWS command line tools:
aws s3 sync --no-sign-request s3://physionet-open/parkinsons-disease-smartwatch/1.0.0/ DESTINATION

## The baseline (work already done)

https://imigitlab.uni-muenster.de/published/pads-project
