mkdir -p ./eval_model/
cd ./eval_model/
echo "The pretrained model will be stored in the 'checkpoints' folder\n"
# InterHuman
echo "Downloading the evaluation model..."
gdown https://drive.google.com/uc?id=1sYOJEdbB8hDI5pY1-X960njdFD2WcwFq #--proxy http://localhost:58591
echo "Evaluation model downloaded"