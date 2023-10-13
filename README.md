# AI_ThermoDiameters
Estimates the RSW nugget diameters from the thermographic videos

# Setup

Creare un virtual environment per installare tutte le dipendenze.

```
virtualenv .venv
source .venv/bin/activate
```

Se tutto va bene, il nome del venv appare davanti alla shell.

Installa il FLIR SDK con pip. Guarda la guida (qua)[https://flir.custhelp.com/app/answers/detail/a_id/3504/~/getting-started-with-flir-science-file-sdk-for-python].

Ora installa anche gli altri pacchetti necessari

```
pip install tensorflow imutils opencv-python matplotlib pandas imageio scipy
``` 

# generaDataSet.py
Questo script python prende i video di Flir da una cartella e genera il dataset come file npz (save di una variabile NumPY).

Va eseguito 2 volte per creare il dataset di training e di test. Le cartelle (vengono create qua) devono avere il nome "train" e "test".

Come se non bastasse, devi creare anche 2 file csv per dare qualche informazione sui video. I file si chiamano train.csv e test.csv e hanno entramvi una struttura tipo

```
fileName,diam
13KA12380000.npz,6.26
....
``` 

Ovviamente i nomi dei file sono quelle delle rispettive cartelle e "diam" sta per diametri :)

# analisi.py

Questo file si prende i dati generati in precedenza per fare il train e testare la RNN. Nota: se commenti un po' di roba puoi saltare il train (devi però salvare la struttura della rete neurale). Guarda il codice per vedere dove salva i parametri della RNN.
