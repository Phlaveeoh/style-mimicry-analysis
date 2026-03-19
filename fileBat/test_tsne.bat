python script/tsneParametrizzato2.py ^
  -m "ConvNext-Finetuned-V3.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "Immagini Originali|tipo=Original,categoria=Training" 

python script/tsneParametrizzato2.py ^
  -m "ConvNext-Finetuned-V3.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "Immagini Originali|tipo=Original,categoria=Training" ^
  --group "Immagini Protette con Glaze|tipo=Protected,categoria=Training,protezione=glaze"

python script/tsneParametrizzato2.py ^
  -m "ConvNext-Finetuned-V3.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "Immagini Originali|tipo=Original,categoria=Training" ^
  --group "Immagini Protette con Mist|tipo=Protected,categoria=Training,protezione=mist"

python script/tsneParametrizzato2.py ^
  -m "ConvNext-Finetuned-V3.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "Immagini Originali|tipo=Original,categoria=Training" ^
  --group "Immagini Protette con AntiDB|tipo=Protected,categoria=Training,protezione=antidb"


python script/tsneParametrizzato2.py ^
  -m "ConvNext-Finetuned-V3.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "Immagini Originali|tipo=Original,categoria=Training" ^
  --group "Immagini Generate da Baseline|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling" 

python script/tsneParametrizzato2.py ^
  -m "ConvNext-Finetuned-V3.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "Immagini Generate da Baseline|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"