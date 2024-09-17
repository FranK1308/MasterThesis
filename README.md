# Diplomski
Codes used in my master thesis

**requirements.txt** holds all the packages needed to run the code.

**Removing_self_energies.py** is a script that strips the energy of molecules down so only the bonding energy remains.

The folder **"ani1ccx"** contains all the necessary codes to train the neural network with only one type of data.

The folder **"combined"** contains all the necessary codes to train the neural network with multiple types of data. 

Both folders contain similar files:

- **egnn.py** is the implementation of EGNN and RFF
- **convert.py** (or **convert_ani1x.py** and **convert_ani1ccx.py**) prepare the data for training
- **main.py** is the main script from which the training is initiated


The folder **structures** has all the same scripts as **ani1ccx** with the addition to **egnn.py** to adjust for periodic boundaries of crystal structures.
