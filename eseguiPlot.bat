@echo off

py tsnePlotOriginals-unprotected-protected.py -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" -p "antidb" -pp "noisy_upscaling"
py tsnePlotOriginals-unprotected-protected.py -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" -p "glaze" -pp "noisy_upscaling"
py tsnePlotOriginals-unprotected-protected.py -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" -p "mist" -pp "noisy_upscaling"

pause