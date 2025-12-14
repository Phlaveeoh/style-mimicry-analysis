@echo off

::NAIVE MIMICRY UNPROTECTED vs NAIVE MIMICRY PROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training"

pause