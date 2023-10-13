# AI_ThermoDiameters
Estimates the RSW nugget diameters from the thermographic videos

# Setup

Creare un virtual environment per installare tutte le dipendenze.

```
virtualenv .venv
source .venv/bin/activate
```

Se tutto va bene, il nome del venv appare davanti alla shell.

Installa il FLIR SDK con pip. Guarda la guida [qua](https://flir.custhelp.com/app/answers/detail/a_id/3504/~/getting-started-with-flir-science-file-sdk-for-python).

Ora installa anche gli altri pacchetti necessari

```
pip install tensorflow imutils opencv-python matplotlib pandas imageio scipy
``` 

# trainer.py
Questo script python prende i video di Flir da una cartella,fa in training della rete neurale e poi ne testa i risultati. 

Devi creare anche 2 file csv per dare qualche informazione sui video. I file si chiamano train.csv e test.csv e hanno entramvi una struttura tipo

```
fileName,diam
13KA12380000.ats,6.26
....
``` 

I nomi dei file sono completi del path e "diam" sta per diametri :)

Il dataset di training viene esteso di `TRAIN_RANDOM_AUG` varianti casuali per ogni video. Per ogni video, sono analizzati i primi `MAX_SEQ_LENGTH` campioni da quanto parte il riscaldamento.

Alla fine produce un file csv `out.csv` che contiene i risultati della validazione.

TODO: parallelizzare un po' il processo di analisi e caricamento dei video...

# flirvideo.py

Classe di Python per l'analisi dei video della termocamera. È la copia del file di Matlab che abbiamo usato per l'altro lavoro. Terrei questa classe nei vari progetti - da espandare in base alle necessità.