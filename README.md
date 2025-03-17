<div align="center">

<h1>CromSS: Cross-modal pretraining with noisy labels for remote sensing image segmentation</h1>

<div>
    <strong>Explore the potential of large-scale easy-to-come-by noisy labels in pretraining for remote sensing image segmentation</strong>
</div>

<div>
    Chenying Liu<sup>1,2,3</sup>&emsp;
    Conrad M Albrecht<sup>2</sup>&emsp;
    Yi Wang<sup>1</sup>&emsp;
    Xiao Xiang Zhu<sup>1,3</sup>&emsp;
</div>
<div>
    <sup>1</sup>Technical University of Munich (TUM)&emsp;
    <sup>2</sup>German Aerospace Center (DLR)&emsp;
    <sup>3</sup>Munich Center for Machine Learning (MCML)&emsp;
</div>

<div>
    <h4 align="center">
    <!-- • <a href="" target='_blank'>[TGRS]</a>  -->
    • TGRS
    • <a href="https://arxiv.org/abs/2405.01217" target='_blank'>[arXiv]</a>
    • <a href="https://github.com/zhu-xlab/CromSS" target='_blank'>[Project]</a>
    </h4>
</div>

</div>

<div align="center">
<img src="media\CromSS_overview.png" width="100%"/>
Overview of the proposed cross-modal sample selection (CromSS) method.

</div>

## Abstract
We explore the potential of large-scale noisily labeled data to enhance feature learning by pretraining semantic segmentation models within a multi-modal framework for geospatial applications. We propose a novel <ins>Cro</ins>ss-modal <ins>S</ins>ample <ins>S</ins>election (<ins>*CromSS*</ins>) method, a weakly supervised pretraining strategy designed to improve feature representations through cross-modal consistency and noise mitigation techniques. Unlike conventional pretraining approaches, CromSS exploits massive amounts of noisy and easy-to-come-by labels for improved feature learning beneficial to semantic segmentation tasks. We investigate middle and late fusion strategies to optimize the multi-modal pretraining architecture design. We also introduce a cross-modal sample selection module to mitigate the adverse effects of label noise, which employs a cross-modal entangling strategy to refine the estimated confidence masks within each modality to guide the sampling process. Additionally, we introduce a spatial-temporal label smoothing technique to counteract overconfidence for enhanced robustness against noisy labels.

To validate our approach, we assembled the multi-modal dataset, *NoLDO-S12*, which consists of <ins>a large-scale noisy label subset</ins> from Google's Dynamic World (DW) dataset for pretraining and <ins>two downstream subsets</ins> with high-quality labels from Google DW and OpenStreetMap (OSM) for transfer learning. Experimental results on two downstream tasks and the publicly available *DFC2020* dataset demonstrate that when effectively utilized, the low-cost noisy labels can significantly enhance feature learning for segmentation tasks. All data, code, and pretrained weights will be made publicly available.


<!-- ## Dependencies and Installation


```
# 1. git clone this repository
git clone 
cd 

# 2. create new anaconda env
conda create -n CromSS python=3.11
conda activate CromSS

# install torch and dependencies
pip install -r requirements.txt
# The dependent versions are not strict, and in general you only need to pay attention to mmcv and mmsegmentation.
``` -->

## NOLDO-S12 Dataset
[**NoLDO-S12**](https://huggingface.co/datasets/vikki23/NoLDO-S12) contains two splits: **SSL4EO-S12@NoL** with <ins>noisy labels</ins> for pretraining, and two downstream datasets, **SSL4EO-S12@DW** and **SSL4EO-S12@OSM**, with <ins>exact labels</ins> for transfer learning.

### - SSL4EO-S12@NoL
<div align="center">
<img src="media\SSL4EO_NS_Pretraining_Part.png" width="100%"/>
Fig. 1. Illustration of the pretraining set SSL4EO-S12@NoL in NoLDO-S12. From left to right: global distribution of samples (left), 4-season samples (top-down) at 3 geolocations (middle), and statistics of the classes of the noisy labels (right).
</div>
<br>

• **SSL4EO-S12@NoL** paired the large-scale, multi-modal, and multi-temporal self-supervised <a href='https://github.com/zhu-xlab/SSL4EO-S12' target='_blank'>SSL4EO-S12</a> dataset with the 9-class noisy labels (NoL) sourced from the <a href='https://dynamicworld.app/' target='_blank'>Google Dynamic World (DW) project</a> on Google Earth Engine (GEE). To keep the dataset's multi-temporal characteristics, we only retain the S1-S2-noisy label triples from the locations where all 4 timestamps of S1-S2 pairs have corresponding DW labels, resulting in about 41\% (103,793 out of the 251,079 locations) noisily labeled data of the SSL4EO-S12 dataset. SSL4EO-S12@NoL well reflects real-world use cases where noisy labels remain more difficult to obtain than bare S1-S2 image pairs.  

The paired noisy label masks along with corresponding image IDs in SSL4EO-S12 can be downloaded from [ssl4eo_s12_nol.zip](https://huggingface.co/datasets/vikki23/NoLDO-S12/blob/main/ssl4eo_s12_nol.zip)



### - SSL4EO-S12@DW \& SSL4EO-S12@OSM
<div align="center">
<img src="media\SSL4EO_NS_Downstream_Part.png" width="100%"/>
Fig. 2. Illustration of the two downstream tasks in NoLDO-S12 with different label sources (SSL4EO-S12@DW and SSL4EO-S12@OSM). Top (left and right): global data distributions(DW and OSM). Middle (left and right): class distributions of training and test sets along with corresponding legends (DW and OSM). Bottom: examples from 2 locations. The legend for DW labels is the same as that in Fig. 1.
</div>

<br>

We construct two downstream datasets, **SSL4EO-S12@DW** and **SSL4EO-S12@OSM** for transfer learning experiments. Both are selected on the DW project’s manually annotated training and validation datasets, yet paired with different label sources from DW and OSM. 

• **SSL4EO-S12@DW** was constructed from the DW expert labeled training subset of 4,194 tiles with given dimensions of 510×510 pixels and its hold-out validation set of 409 tiles with given dimensions of 512×512. The human labeling process allows some ambiguous areas left unmarked (white spots in DW masks in Fig. 2). We spatial-temporally aligned the S1 and S2 data for the training and test tiles with GEE, leading to 3,574 training tiles and 340 test tiles, that is, a total of 656,758,064 training pixels and 60,398,506 test pixels. The class distributions can be found in Fig. 2.

The SSL4EO-S12@DW downstream dataset can be downloaded from [ssl4eo_s12_dw.zip](https://huggingface.co/datasets/vikki23/NoLDO-S12/blob/main/ssl4eo_s12_dw.zip)

• **SSL4EO-S12@OSM** adopts 13-class fine-grained labels derived from OpenStreetMap (OSM) following the work of <a href='https://osmlanduse.org/#12/8.7/49.4/0/' target='_blank'>Schultz et al.</a> We retrieved 2,996 OSM label masks among the 3,914=3,574+340 DW tiles, with the remaining left without OSM labels. After an automatic check with DW labels as reference assisted by some manual inspection, we construct SSL4EO-S12@OSM with 1,375 training tiles and 400 test tiles, that is, a total of 165,993,707 training pixels and 44,535,192 test pixels.

The SSL4EO-S12@DW downstream dataset can be downloaded from [ssl4eo_s12_osm.zip](https://huggingface.co/datasets/vikki23/NoLDO-S12/blob/main/ssl4eo_s12_osm.zip)

• Some downloading scripts can be found in `data_prepare/data_check_SSL4EO/get_dw_data` and `data_prepare/data_check_SSL4EO/get_osm_labels`

## Data preparation
• Write the SSL4EO-S12-NoL pretraining images (S1/S2) to lmdbs: `data_prepare/data_check_SSL4EO/construct_pretrain_lmdb/write_labels_to_lmdb.py`\
• Write the SSL4EO-S12-NoL pretraining noisy labels to lmdb: `data_prepare/data_check_SSL4EO/construct_pretrain_lmdb/write_labels_to_lmdb.py`\
• Write the <a href='https://ieee-dataport.org/competitions/2020-ieee-grss-data-fusion-contest' target='_blank'>DFC2020</a> downstream dataset to lmdb: `data_prepare/data_check_DFC20/write_data_to_lmdb.py`\
• Write the SSL4EO-S12@DW/OSM downstream dataset to lmdb: `data_prepare/data_check_SSL4EO/construct_dw_osm_lmdb/read_dw_data_to_lmdb.py`

## Pretraining
Implement pretraining on the SLURM system with `fusion_type=mid/late` for middle and late fusion settings, respectively.
### - CromSS (proposed):
```bash
srun python -u py_scripts_SSL4EO/train_SSL4EO_unet_pl_pretrain_mm_sscom.py \
        --data_path $data_directory_path \
        --data_name_s1 0k_251k_uint8_s1.lmdb \
        --data_name_s2 0k_251k_uint8_s2c.lmdb \
        --data_name_label dw_labels.lmdb \
        --input_type s12 \
        --fusion_type mid \
        --save_dir $save_path \
        --model_type resnet50 \
        --n_channels 13 \
        --n_classes 9 \
        --loss_type cd \
        --consist_loss_type ce \
        --label_smoothing \
        --label_smoothing_factor 0.15 0.05 \
        --label_smoothing_prior_type ts \
        --sample_selection \
        --sample_selection_rmup_func exp \
        --sample_selection_rmdown_epoch 80 \
        --sample_selection_prop 0.5 \
        --sample_selection_confidence_type ce \
        --experiment ns_labels \
        --validation 0.01 \
        --batch_size  128 \
        --num_workers 12 \
        --accelerator gpu \
        --slurm \
        --epochs $ep \
        --optimizer adam \
        --learning_rate 0.005 \
        --lr_adjust_type rop \
        --lr_adjust 30
```



<table>
  <tr>
    <td>#S2 bands</td>
    <td>Fusion type</td>
    <td>Link</td>
  </tr>
  <tr>
    <td rowspan="2">13B</td>
    <td>middle</td>
    <td><a href='https://huggingface.co/datasets/vikki23/NoLDO-S12/blob/main/weights-cromss-13B-midFusion-epoch%3D199.ckpt' target='_blank'>weights-cromss-13B-midFusion-epoch=199.ckpt</a></td>
  </tr>
  <tr>
    <td>late</td>
    <td><a href='https://huggingface.co/datasets/vikki23/NoLDO-S12/blob/main/weights-cromss-13B-lateFusion-epoch%3D199.ckpt' target='_blank'>weights-cromss-13B-lateFusion-epoch=199.ckpt</a></td>
  </tr>
  <tr>
    <td rowspan="2">9B</td>
    <td>middle</td>
    <td><a href='https://huggingface.co/datasets/vikki23/NoLDO-S12/blob/main/weights-cromss-9B-midFusion-epoch%3D199.ckpt' target='_blank'>weights-cromss-9B-midFusion-epoch=199.ckpt</a>
  </tr>
  <tr>
    <td>late</td>
    <td><a href='https://huggingface.co/datasets/vikki23/NoLDO-S12/blob/main/weights-cromss-9B-lateFusion-epoch%3D199.ckpt' target='_blank'>weights-cromss-9B-lateFusion-epoch=199.ckpt</a>
    </td>
  </tr>
</table>



### - Multi-modal pretraining with noisy labels yet without cross-modal sample selection:
Use the `py_scripts_SSL4EO/train_SSL4EO_unet_pl_pretrain_mm.py` script.
<!-- ```
srun python -u py_scripts_SSL4EO/train_SSL4EO_unet_pl_pretrain_mm.py \
        --data_path $data_directory_path \
        --data_name_s1 0k_251k_uint8_s1.lmdb \
        --data_name_s2 0k_251k_uint8_s2c.lmdb \
        --data_name_label dw_labels.lmdb \
        --input_type s12 \
        --fusion_type mid \
        --save_dir $save_path \
        --model_type resnet50 \
        --n_channels 13 \
        --n_classes 9 \
        --loss_type cd \
        --consist_loss_type ce \
        --experiment ns_labels \
        --validation 0.01 \
        --batch_size  128 \
        --num_workers 12 \
        --accelerator gpu \
        --slurm \
        --epochs 200 \
        --optimizer adam \
        --learning_rate 0.005 \
        --project_name $wandb_project_name \
        --entity_name $wandb_entity_name \
        --wandb_mode $wandb_mode 
``` -->

### - Single-modal pretraining with noisy labels:
Use the `py_scripts_SSL4EO/train_SSL4EO_unet_pl_pretrain.py` script with `input_type=s1/s2` for each single modality

## Fine-tuning
Example bash scripts using one single GPU:<br>
`train_SSL4EO_pl_ft_DFC2020.sh`<br>
`train_SSL4EO_pl_ft_DW.sh`<br>
`train_SSL4EO_pl_ft_OSM.sh`<br>

## Results
<img src="media\Results_DFC2020.png" width="100%"/>
Table. 1. Transfer learning results (%) on the DFC2020 dataset from PSPNet with ResNet-50 backbones and UperNet with ViT-large backbones. 3B\9B\10B\13B in brackets indicate the number of S2 bands as inputs for each model. Frozen and Fine-tuned represent no updates of encoders and optimizing encoders along with decoders in the transfer learning setting. The best results are highlighted in bold. The annotations are the same in the following tables.

<br>

<img src="media\Results_DW.png" width="100%"/>
Table. 2. Transfer learning results (%) on the SSL4EO-S12@DW dataset from DeepLabv3+ with ResNet-50 and UperNet with ViT-large.

<br>

<img src="media\Results_OSM.png" width="100%"/>
Table. 3. Transfer learning results (%) on the SSL4EO-S12@OSM dataset from FPN with ResNet-50 and UperNet with ViT-large.

## Citation
```BibTeX
@ARTICLE{liu-cromss,
  author={Liu, Chenying and Albrecht, Conrad M and Wang, Yi and Zhu, Xiao Xiang},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CromSS: Cross-modal pretraining with noisy labels for remote sensing image segmentation}, 
  year={2025},
  volume={},
  number={},
  pages={in press}}
```