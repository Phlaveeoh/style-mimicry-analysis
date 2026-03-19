import subprocess

modello = "ConvNext-Finetuned-V3.keras"
dataset = "C:\\Users\\Flavio\\Desktop\\datasetTotale"

protezioni = ["glaze", "mist", "antidb"]
purificazioni = ["gaussian_noise_0.05", "diffpure", "impress++", "noisy_upscaling"]

for prot in protezioni:
    for pur in purificazioni:
        print(f"\n Avvio elaborazione: Protezione={prot.upper()} | Purificazione={pur.upper()}")
        
        # Generazione dinamica dei parametri di filtro
        gruppo_baseline = f"Generate Baseline ({prot}-{pur})|tipo=No-Protections,categoria=Generated,protezione={prot},metodo_processamento={pur}"
        gruppo_naive = f"Generate Naive ({prot})|tipo=Naive-Mimicry,categoria=Generated,protezione={prot}"
        gruppo_robust = f"Generate Robust ({prot}-{pur})|tipo=Robust-Mimicry,categoria=Generated,protezione={prot},metodo_processamento={pur}"
        
        comando = [
            "python", "script/tsneParametrizzato2.py",
            "-m", modello,
            "-d", dataset,
            "--group", gruppo_baseline,
            "--group", gruppo_naive,
            "--group", gruppo_robust
        ]
        
        # Esecuzione bloccante del processo figlio
        processo = subprocess.run(comando)
        
        if processo.returncode != 0:
            print(f"[-] Errore. Esecuzione interrotta.")
            exit(1)

print("\n[=] Mappatura topologica batch completata con successo.")