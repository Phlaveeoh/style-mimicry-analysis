python script/tsneParametrizzato2.py ^
  -m "ConvNext-Finetuned-V3.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "Immagini Generate da Baseline|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling" ^
  --group "Immagini Generate da Naive Mimicry|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze" ^
  --group "Immagini Generate da Robust Mimicry|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"