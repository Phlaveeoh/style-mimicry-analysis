python script/classificaImmagini.py ^
-m "ConvNext-Finetuned-V4.keras" ^
-d "C:\Users\Flavio\Desktop\datasetTotale" ^
-c "wikiart_albrecht-durer,wikiart_alphonse-mucha,wikiart_anna-ostroumova-lebedeva,wikiart_edvard-munch,wikiart_edward-hopper" ^
--artista "wikiart_edward-hopper" --categoria "Training" --tipo "Original"

python script/classificaImmagini.py ^
-m "ConvNext-Finetuned-V4.keras" ^
-d "C:\Users\Flavio\Desktop\datasetTotale" ^
-c "wikiart_albrecht-durer,wikiart_alphonse-mucha,wikiart_anna-ostroumova-lebedeva,wikiart_edvard-munch,wikiart_edward-hopper" ^
--artista "wikiart_edward-hopper" --categoria "Training" --tipo "Protected" --protezione "glaze"

python script/classificaImmagini.py ^
-m "ConvNext-Finetuned-V4.keras" ^
-d "C:\Users\Flavio\Desktop\datasetTotale" ^
-c "wikiart_albrecht-durer,wikiart_alphonse-mucha,wikiart_anna-ostroumova-lebedeva,wikiart_edvard-munch,wikiart_edward-hopper" ^
--artista "wikiart_edward-hopper" --categoria "Generated" --tipo "No-Protections" --protezione "glaze" --metodo "noisy_upscaling"

python script/classificaImmagini.py ^
-m "ConvNext-Finetuned-V4.keras" ^
-d "C:\Users\Flavio\Desktop\datasetTotale" ^
-c "wikiart_albrecht-durer,wikiart_alphonse-mucha,wikiart_anna-ostroumova-lebedeva,wikiart_edvard-munch,wikiart_edward-hopper" ^
--artista "wikiart_edward-hopper" --categoria "Generated" --tipo "Robust-Mimicry" --protezione "glaze" --metodo "noisy_upscaling"