generate:
	uv run maturin develop --release
run:
	uv run python main.py
run-mujoco:
	uv run python main_mujoco.py
run-rl:
	uv run python train_simple.py
run-ppo:
	uv run python train_ppo.py
eval-ppo:
	uv run python train_ppo.py --eval
run-ball:
	# python仮想環境じゃないとpyo3がコンパイルできない
	uv run cargo run
tensorboard:
	uv run tensorboard --logdir=logs