



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
Other model parameters can be downloaded [here](https://drive.google.com/file/d/1Sw-R0S2grtUVuGt8znFJHFaV6YP20Nsm/view?usp=sharing)


### Machine Learn data


The machine learning script, as well as the training and test data, are available via the URL below.

Feature and script  [here](https://drive.google.com/file/d/1vZVD9JJI91omPBoQ9xowwsmzaSrVDsFf/view?usp=sharing)  

model for extract feature [here](https://drive.google.com/file/d/1648W5aGGBukxh-H3_PqlbTYAl4yZaaJV/view?usp=sharing)
  

### PointCloud Data


Download the PDBBind all data and Bigdata point cloud data using the URL below.

PDBBind all data     [here](https://drive.google.com/drive/folders/1XiuaIM7f1lB_H2o46VCH4apBaxFYbGOn?usp=sharing)




