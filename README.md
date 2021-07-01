



### Generate point cloud


Use POINTNET files to generate point cloud data for proteins and ligands.All data must be in PDB format. All data should remove solvents, metals and ions. And use openbabel to add polar hydrogen (--AddPolarH)

  ```sh
  ./POINTNET {protein_path} {ligand_path} {out_path}
  ./POINTNET-2048 {protein_path} {ligand_path} {out_path}
  ./POINTNET-atomchannel {protein_path} {ligand_path} {out_path}

  ```


### Prediction


1. Installation dependent environment
  ```sh
   conda create -n point_cloud_envs
   conda activate point_cloud_envs
   conda install python=3.7
   conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
   ```
2. Prediction
   ```sh
   python pred.py --file ./example/5c2h_11.09 --model PointTransformer
   ```
Other model parameters can be downloaded [here](https://drive.google.com/file/d/1VzAqTEoxFd4hgiAoWvy6cG80Ct5AK2AY/view?usp=sharing)


### Machine Learn


The machine learning script, as well as the training and test data, are available via the URL below.

PDBBind-2007 [here](https://drive.google.com/file/d/1b7XZqEFIBdzLcVakjCItXvyaXRMNLsN5/view?usp=sharing)  
  
PDBBind-2013 [here](https://drive.google.com/file/d/1NXi7RybbJ6Q5IFR0CZMyMPtskFZvP92m/view?usp=sharing)  
  
PDBBind-2016 [here](https://drive.google.com/file/d/1Ut10Bkd7cRwTwjBOp0nq9rfLwS8kkhd4/view?usp=sharing)  


### Train Data


Download the PDBBind-2016 Refine set and Bigdata point cloud data using the URL below.

PDBBind-2016 Refine set     [here](https://drive.google.com/file/d/1ylh8UsBsI95AVSRXWRPGCMClZ2a4iZdO/view?usp=sharing)

PDBBind-2016 Bigdata        [here](https://drive.google.com/file/d/16YiLhJABX8l89HwVci8of3pT6bBKxfVq/view?usp=sharing)


