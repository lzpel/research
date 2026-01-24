import numpy as np
import catch
def main():
    print("Hello from sandbox-202601catch!")

    x = np.random.rand(10_000_000).astype(np.float64)
    print(catch.sum_f64(x))


if __name__ == "__main__":
    main()
