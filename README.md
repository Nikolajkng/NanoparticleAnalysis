# NanoparticleAnalysis
Bachelor Project 2025


# Run from unet directly from main from path=src/
python -m model.UNet_old


# Flow for when user clicks "Train model"
= FILE              FUNCTION 
-> GUI              (on_train_model_clicked) 
-> Controller       (process_command: RETRAIN) 
-> RequstHandler    (process_request_train)
-> CrossValidation  (cv_kfold || cv_holdout)
-> UNet             (train_model)