# NanoparticleAnalysis
Bachelor Project 2025


# Run from unet directly from main from path=src/
python -m model.UNet_old


# Flow
GUI (on_train_model_clicked) 
-> controller (command: retrain) 
-> user_request (process_request_train)
-> cross-validation (train_model_kfold)
-> UNet ()