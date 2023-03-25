CUDA_VISIBLE_DEVICES=3 python main.py data/room/ --workspace trial_merf_360_room        --enable_cam_center --eval_cnt 1 --save_cnt 1 --downscale 4
CUDA_VISIBLE_DEVICES=3 python main.py data/bonsai/ --workspace trial_merf_360_bonsai    --enable_cam_center --eval_cnt 1 --save_cnt 1 --downscale 4
CUDA_VISIBLE_DEVICES=3 python main.py data/kitchen/ --workspace trial_merf_360_kitchen  --enable_cam_center --eval_cnt 1 --save_cnt 1 --downscale 4
CUDA_VISIBLE_DEVICES=3 python main.py data/counter/ --workspace trial_merf_360_counter  --enable_cam_center --eval_cnt 1 --save_cnt 1 --downscale 4
CUDA_VISIBLE_DEVICES=3 python main.py data/garden/ --workspace trial_merf_360_garden    --enable_cam_center --eval_cnt 1 --save_cnt 1 --downscale 4
CUDA_VISIBLE_DEVICES=3 python main.py data/stump/ --workspace trial_merf_360_stump      --enable_cam_center --eval_cnt 1 --save_cnt 1 --downscale 4
CUDA_VISIBLE_DEVICES=3 python main.py data/bicycle/ --workspace trial_merf_360_bicycle  --enable_cam_center --eval_cnt 1 --save_cnt 1 --downscale 4