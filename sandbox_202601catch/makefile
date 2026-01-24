generate:
	uv run maturin develop --release
run:
	uv run python main.py
run-mujoco:
	uv run python main_mujoco.py
run-ball:
	# python仮想環境じゃないとpyo3がコンパイルできない
	uv run cargo run