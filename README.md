



### Generate point cloud

Use POINTNET files to generate point cloud data for proteins and ligands.All data should remove solvents, metals and ions. And use openbabel to add polar hydrogen (--AddPolarH)

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
