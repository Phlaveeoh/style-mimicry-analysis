@echo off

::NAIVE MIMICRY UNPROTECTED vs NAIVE MIMICRY PROTECTED vs ORIGINALS
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "nm_mist|tipo=Naive-Mimicry,categoria=Generated,protezione=mist,metodo_processamento=none" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=mist,metodo_processamento=noisy_upscaling"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=none" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "nm_antidb|tipo=Naive-Mimicry,categoria=Generated,protezione=antidb,metodo_processamento=none" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=antidb,metodo_processamento=noisy_upscaling"

::ROBUST MIMICRY VS NAIVE MIMICRY PROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze" ^
  --group "rm_glaze|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

::ROBUST MIMICRY VS NAIVE MIMICRY UNPROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling" ^
  --group "rm_glaze_noisy_upscaling|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=diffpure" ^
  --group "rm_glaze_diffpure|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=diffpure"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=gaussian_noise_0.05" ^
  --group "rm_glaze_gaussian_noise|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=gaussian_noise_0.05"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=impress++" ^
  --group "rm_glaze_impress++|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=impress++"

::Naive Mimicry PROTECTED vs Naive Mimicry UNPROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_mist|tipo=Naive-Mimicry,categoria=Generated,protezione=mist" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=mist,metodo_processamento=noisy_upscaling"

::ROBUST MIMICRY vs ORIGINALS
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "rm_glaze_noisy_upscaling|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "rm_glaze_diffpure|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=diffpure"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "rm_glaze_gaussian_noise|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=gaussian_noise_0.05"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "originals|tipo=Original,categoria=Training" ^
  --group "rm_glaze_impress++|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=impress++"

::ROBUST MIMICRY vs NAIVE MIMICRY PROTECTED vs NAIVE MIMICRY UNPROTECTED
py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling" ^
  --group "rm_glaze_noisy_upscaling|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=noisy_upscaling"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=diffpure" ^
  --group "rm_glaze_diffpure|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=diffpure"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=gaussian_noise_0.05" ^
  --group "rm_glaze_gaussian_noise|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=gaussian_noise_0.05"

py tsneParametrizzato.py ^
  -m "C:\\Users\\Flavio\\Desktop\\modelli_fine-tuned\\convnext_finetuned_v2.keras" ^
  -d "C:\\Users\\Flavio\\Desktop\\datasetTotale" ^
  --group "nm_glaze|tipo=Naive-Mimicry,categoria=Generated,protezione=glaze" ^
  --group "nm_unprotected|tipo=No-Protections,categoria=Generated,protezione=glaze,metodo_processamento=impress++" ^
  --group "rm_glaze_impress++|tipo=Robust-Mimicry,categoria=Generated,protezione=glaze,metodo_processamento=impress++"
pauses