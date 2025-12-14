@echo off

::NAIVE MIMICRY UNPROTECTED vs NAIVE MIMICRY PROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "nm_mist|tipo=Naive-Mimicry,categoria=Generated,protezione=mist,metodo_processamento=none" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=mist,metodo_processamento=noisy_upscaling"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=none" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "nm_antidb|tipo=Naive-Mimicry,categoria=Generated,protezione=antidb,metodo_processamento=none" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=antidb,metodo_processamento=noisy_upscaling"

::ROBUST MIMICRY VS NAIVE MIMICRY PROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=none" ^
  --group "rm_glaze|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

::ROBUST MIMICRY VS NAIVE MIMICRY UNPROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling" ^
  --group "rm_glaze|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

pause