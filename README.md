# AI_ThermoDiameters
Estimates the RSW nugget diameters from the thermographic videos

# Setup
Non mi ricordo quali package di python ho installato con pip
Sicuramente va messo l'SDK di Flir. Tanti auguri ad installarlo. Prima o poi ci scrivo una guida.

# generaDataSet.py
Questo script python prende i video di Flir da una cartella e genera il dataset come file npz (save di una variabile NumPY).
Va eseguito 2 volte per creare il dataset di training e di test. Le cartelle (vengono create qua) devono avere il nome "train" e "test".
Come se non bastasse, devi creare anche 2 file csv per dare qualche informazione sui video. I file si chiamano train.csv e test.csv e hanno entramvi una struttura tipo

'''
fileName,diam
13KA12380000.npz,6.26
....
'''

Ovviamente i nomi dei file sono quelle delle rispettive cartelle e "diam" sta per diametri :)

# analisy.py

Questo file si prende i dati generati in precedenza per fare il train e testare la RNN. Nota: se commenti un po' di roba puoi saltare il train (devi però salvare la struttura della rete neurale). Guarda il codice per vedere dove salva i parametri della RNN.