"""
# Raccoonbot_Openvla
**2021741026 송인규**

## 과제 확장 내용 (Assignment Extensions)

### Dataset Extension
- **새로운 오브젝트 추가**: orange box (기존 cylinder에서 확장)
- **새로운 instruction**: "grasp the orange box"
- **데이터셋**: `raccoon_grasp_box_dataset/` (100 episodes)
- **에피소드 시각화**: `episode_visualization.png` 참고
- **TFDS**: `tensorflow_datasets/raccoon_box/` (train 90, val 10)
- **short LoRA test**: 500 steps 완료 (`openvla-7b+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--box-test--image_aug`)

### Code Improvement
- `openvla_server.py`: 추론 시간 로깅 추가 (inference timing log)
- `raccoon_grasp_multicolor_scene_dataset.py`: box 오브젝트 생성 코드 추가
- `Raccoon_colored_cylinder.xml`: box 오브젝트 XML 추가
- `openvla/Untitled.ipynb`: MuJoCo rollout 프레임을 연속으로 저장하여 mp4 영상으로 시각화

---

⭐ 1~3번은 직접 finetuning을 진행하는 내용이니 체크포인트를 불러와서 사용하는 경우 0번과 4번만 진행

0~3번 server에서 실행, 4번 local-server 실행

## 0. Dependencies
git clone https://github.com/Songik40/Raccoonbot_Openvla.git

필요한 패키지 설치
apt update

apt install -y \

libegl1 \

libgl1 \

libglvnd0 \

libglx0 \

libopengl0 \

libgles2 \

libegl1-mesa \

libegl1-mesa-dev \

mesa-utils
cd Raccoonbot_Openvla/openvla

pip install .

## 1. Dataset 생성 (원본: cylinder 4색상)
cd /data/2021741026/Raccoonbot_Openvla/Mujoco

python raccoon_grasp_multicolor_scene_dataset.py
→ `raccoon_grasp_colored_cylinder/` 에 400 episodes 생성

## 1-1. Dataset 생성 (확장: orange box)
box 오브젝트를 추가한 확장 데이터셋 생성
cd /data/2021741026/Raccoonbot_Openvla/Mujoco

python raccoon_grasp_multicolor_scene_dataset.py --object box
→ `raccoon_grasp_box_dataset/` 에 100 episodes 생성

## 2. rlds 파일 변환 (원본)
cd /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_dataset

python convert_raw_to_openvla_rlds_intermediate.py \

--raw_root /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_colored_cylinder \

--out_root /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate \

--val_ratio 0.1

## 2-1. rlds builder (원본)
cd /data/2021741026/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_pick_place

tfds build --overwrite

mv /root/tensorflow_datasets /data/2021741026/Raccoonbot_Openvla/

## 2-2. rlds 파일 변환 (확장: box)
cd /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_dataset

python convert_raw_to_openvla_rlds_intermediate.py \

--raw_root /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_box_dataset \

--out_root /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_dataset/box_rlds_intermediate \

--val_ratio 0.1

## 2-3. rlds builder (확장: box)
cd /data/2021741026/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_box

tfds build --overwrite

mv /root/tensorflow_datasets/raccoon_pick_place /data/2021741026/Raccoonbot_Openvla/tensorflow_datasets/raccoon_box

## 3. Raccoonbot 기반 OpenVLA finetuning
cd /data/2021741026/Raccoonbot_Openvla/openvla

export PYTHONPATH=/data/2021741026/Raccoonbot_Openvla/openvla:$PYTHONPATH
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 \

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \

--vla_path openvla/openvla-7b \

--data_root_dir /data/2021741026/Raccoonbot_Openvla/tensorflow_datasets \

--dataset_name raccoon_pick_place \

--run_root_dir /data/2021741026/Raccoonbot_Openvla/openvla/openvla-runs \

--adapter_tmp_dir /data/2021741026/Raccoonbot_Openvla/openvla/openvla-adapter-tmp \

--lora_rank 32 \

--batch_size 8 \

--grad_accumulation_steps 2 \

--learning_rate 5e-4 \

--max_steps 30000 \

--save_steps 30000 \

--run_id_note raccoon-eef-v100

## 4. Mujoco 환경 Inference (local-server)

## 4-1. Hugging Face에서 모델 다운로드
pip install -U huggingface_hub

hf download fair-lab/openvla-7b-finetuned-raccoonbot \

--local-dir /data/2021741026/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot

## 4-2. 서버 실행
cd /data/2021741026/Raccoonbot_Openvla/openvla

CUDA_VISIBLE_DEVICES=0 python openvla_server.py \

--model_path /data/2021741026/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot \

--default-unnorm-key raccoon_pick_place \

--host 0.0.0.0 \

--port 8000 \

--device cuda

## 4-3. 클라이언트 환경 설정
클라이언트 파일 [다운로드](https://drive.google.com/drive/folders/1xrH3FoTfKC9CiUE-kDRorxTKMMq0O7Px?usp=sharing)
pip install -r requirements.txt

## 4-4. 클라이언트 실행 (MuJoCo 시뮬레이션)
rollout 결과는 mp4 영상으로 저장됨
python openvla_multicolor_client.py --server_url http://서버IP:8000 --xml_path Raccoon_colored_cylinder.xml --target_color red --use_viewer
"""

with open("/data/2021741026/Raccoonbot_Openvla/README.md", "w") as f:
    f.write(readme_content)
print("완료")
