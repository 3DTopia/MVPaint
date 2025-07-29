### sh ./run_pipeline.sh

export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
export HF_ENDPOINT=https://hf-mirror.com
export OPENCV_IO_ENABLE_OPENEXR=1

CODE_ROOT=$(pwd)
CKPT_PATH=${CODE_ROOT}/stage1_low/ckpt/mvdream-controlnet-sd21.ckpt
SAMPLE_DIR=${CODE_ROOT}/samples
prompt_file=${SAMPLE_DIR}/annot.json
OUT_ROOT=${CODE_ROOT}/sample_results

conda activate mvdream

### preprocessing 
cd ${CODE_ROOT}
echo "=== preprossing ===" 
raw_dir=${SAMPLE_DIR}
preprocess_dir=${OUT_ROOT}/preprocess
### if the provided mesh doesn't have uv coords, then set --auto_uv to generate uv coords
python preprocess_mesh.py \
    --in_dir ${raw_dir} --out_dir ${preprocess_dir} # --auto_uv 

### stage 1: low resolution results 
cd ${CODE_ROOT}/stage_1_low_res
echo "=== stage1 low res ==="
stage1_low_dir=${OUT_ROOT}/stage1_low
exp=sd2-inv-uv-8-infer
python run_stage1_low.py \
    --base ./mvdream/configs/${exp}.yaml \
    --logdir ./logs \
    --name ${exp} \
    --outdir ${stage1_low_dir} \
    data.params.root=${preprocess_dir} \
    data.params.prompt_file=${prompt_file} \
    data.params.pid=0 data.params.pnum=1 \
    resume_model_from_checkpoint=${CKPT_PATH}

conda deactivate 

### stage 1: high resolution results 
conda activate syncmvd
cd ${CODE_ROOT}/stage_1_high_res
echo "=== stage1 high res ==="
stage1_high_dir=${OUT_ROOT}/stage1_high
scale=0.6
python run_stage1_high.py \
    --config ./config.yaml \
    --mesh_folder ${preprocess_dir} \
    --lowres_folder ${stage1_low_dir} \
    --output ${stage1_high_dir} \
    --prompt_file ${prompt_file} \
    --camera_azims 0 90 180 270 45 135 225 315 \
    --no_top_cameras \
    --latent_view_size 128 --latent_tex_size 512 --extra_strength 1 ${scale} 

### stage 2 & stage 3
cd ${CODE_ROOT}/stage_2_3
echo "=== stage 2 & 3 ==="
stage23_dir=${OUT_ROOT}/stage23
python run_stage23.py \
    --stage1_folder ${stage1_high_dir} \
    --mesh_folder ${preprocess_dir} \
    --output ${stage23_dir} 

conda deactivate 

### visualization
conda activate mvdream 
cd ${CODE_ROOT}
echo "=== rendering ===" 
render_dir=${OUT_ROOT}/render_final
python render_final.py \
    --in_dir ${stage23_dir} \
    --out_dir ${render_dir}  
conda deactivate 

echo "=== Finish All ===" 