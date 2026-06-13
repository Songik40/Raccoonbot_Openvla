# Raccoonbot_Openvla
**2021741026 송인규**

## 과제 확장 내용 (Assignment Extensions)

### Dataset Extension
- **새로운 오브젝트 추가**: orange box (`target_object_box`) — 기존 cylinder 4색상에서 확장
- **Diverse language instructions**: grasp / pick up / grab / lift 4종 템플릿 적용
- **통합 데이터셋**: cylinder 400 episodes + box 100 episodes = 총 500 episodes
- **RLDS 변환**: train 449개, val 50개 (1개 실패 에피소드 제외)
- **TFDS**: `raccoon_pick_place/1.0.0` (290MB)
- **에피소드 시각화**: `client/episode_visualization.png` (rollout 주요 프레임 6장 그리드)
- **LoRA 파인튜닝**: 500 steps 완료 (`openvla-7b+raccoon_pick_place+b16+lr-0.0005+lora-r32+dropout-0.0--raccoon-diverse-instructions--image_aug`)

### Code Improvement
- `openvla/openvla_server.py`: 7D→4DOF action mapping + 추론 시간 로깅 추가
- `client/raccoon_env.py`: 4D action 지원 (기존 7D only → 4D/7D 호환)
- `Mujoco/raccoon_grasp_multicolor_scene_dataset.py`: box 오브젝트 + diverse instruction 추가
- `Mujoco/Raccoon_colored_cylinder.xml` / `client/Raccoon_colored_cylinder.xml`: orange box body 추가
- `openvla/vla-scripts/finetune.py`: QLoRA + DDP 호환 픽스 (`enable_input_require_grads`, `_set_static_graph`)
- Rollout 시각화: MuJoCo rollout 프레임을 mp4로 저장 → `client/rollout_box.mp4`

---

⭐ 1~3번은 직접 파인튜닝을 진행하는 내용. 체크포인트를 불러와서 사용하는 경우 0번과 4번만 진행.

0~3번 server에서 실행, 4번 local에서 실행.

## 0. Dependencies

```bash
git clone https://github.com/Songik40/Raccoonbot_Openvla.git
```

필요한 패키지 설치:
```bash
apt update && apt install -y libegl1 libgl1 libglvnd0 libglx0 libopengl0 libgles2 libegl1-mesa libegl1-mesa-dev mesa-utils

cd Raccoonbot_Openvla/openvla
pip install .
pip install bitsandbytes==0.41.3
```

## 1. Dataset 생성 (통합: cylinder 400 + box 100, diverse instructions)

```bash
cd /data/2021741026/Raccoonbot_Openvla/Mujoco
python raccoon_grasp_multicolor_scene_dataset.py
```
→ `raccoon_grasp_colored_cylinder/` 에 400 episodes, `raccoon_grasp_box_dataset/` 에 100 episodes 생성

## 2. 통합 데이터셋 병합 및 RLDS 변환

```bash
# cylinder + box 합치기 (box 에피소드 번호 +400 오프셋)
mkdir -p /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_combined
cp -r /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_colored_cylinder/episode_* \
      /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_combined/
for d in /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_box_dataset/episode_*/; do
    orig=$(basename "$d")
    num=$(echo "$orig" | sed 's/episode_0*//')
    new_name=$(printf "episode_%06d" $((num + 400)))
    cp -r "$d" /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_combined/"$new_name"
done
```

```bash
# RLDS intermediate 변환
cd /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_dataset
python convert_raw_to_openvla_rlds_intermediate.py \
  --raw_root /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_grasp_combined \
  --out_root /data/2021741026/Raccoonbot_Openvla/Mujoco/raccoon_dataset/openvla_rlds_intermediate \
  --val_ratio 0.1
```

```bash
# TFDS 빌드
cd /data/2021741026/Raccoonbot_Openvla/Mujoco/rlds_dataset_builder/raccoon_pick_place
tfds build --overwrite
cp -r /root/tensorflow_datasets/raccoon_pick_place \
      /data/2021741026/Raccoonbot_Openvla/tensorflow_datasets/
```

## 3. LoRA 파인튜닝

```bash
cd /data/2021741026/Raccoonbot_Openvla/openvla
BNB_CUDA_VERSION=118 \
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
  --max_steps 500 \
  --save_steps 500 \
  --run_id_note raccoon-diverse-instructions
```

> **참고**: `BNB_CUDA_VERSION=118` 은 PyTorch CUDA 12.1 환경에서 CUDA 11.3 toolkit을 사용할 때 필요. PEFT가 bitsandbytes를 항상 import하므로 필수.

## 4. MuJoCo 환경 Inference

### 4-1. 체크포인트 다운로드 (교수님 제공)
```bash
pip install -U huggingface_hub
hf download fair-lab/openvla-7b-finetuned-raccoonbot \
  --local-dir /data/2021741026/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot
```

### 4-2. 서버 실행
```bash
cd /data/2021741026/Raccoonbot_Openvla/openvla
CUDA_VISIBLE_DEVICES=0 python openvla_server.py \
  --model_path /data/2021741026/Raccoonbot_Openvla/openvla/openvla-runs/openvla-7b-finetuned-raccoonbot \
  --default-unnorm-key raccoon_pick_place \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda
```

### 4-3. 클라이언트 환경 설정
`client/` 폴더의 파일들을 사용:
```bash
cd client
pip install -r requirements.txt
```

### 4-4. 클라이언트 실행 (MuJoCo 시뮬레이션)
rollout 결과는 mp4 영상으로 저장됨:
```bash
# cylinder (red/blue/green/yellow)
python openvla_multicolor_client.py --server_url http://서버IP:8000 \
  --xml_path Raccoon_colored_cylinder.xml --target_color red --use_viewer

# orange box
python openvla_multicolor_client.py --server_url http://서버IP:8000 \
  --xml_path Raccoon_colored_cylinder.xml --target_color box --use_viewer
```
