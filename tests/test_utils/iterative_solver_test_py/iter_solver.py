import numpy as np

def main():
   A = np.array(
      [
         [2, 0, 2, 0],
         [0, 2, 0, 2],
         [2, 0, 3, 1],
         [0, 2, 1, 3],
      ],
      dtype=np.float32
   )

   M = np.array(
      [
         [2, 0, 0, 0],
         [0, 2, 0, 0],
         [0, 0, 3, 0],
         [0, 0, 0, 3],
      ],
      dtype=np.float32
   )

   N = np.array(
      [
         [0, 0, 2, 0],
         [0, 0, 0, 2],
         [2, 0, 0, 1],
         [0, 2, 1, 0],
      ],
      dtype=np.float32
   )

   b = np.array([1, 1, 1, 1], dtype=np.float32)

   x_ = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

   M_inv = np.linalg.inv(M)

   # this is a linear system solver
   # for i in range(1000):
   #    x_ = np.dot(M_inv, b) - np.dot(np.dot(M_inv, N), x_)

   # this is an LCP solver
   for _ in range(1000):
      for i in range(4):
         delta_x = (b[i] - np.dot(A[i], x_)) / A[i, i]
         x_[i] += delta_x
         np.clip(x_[i], 0.0, np.inf)

   print("x: ", x_)

if __name__ == "__main__":
   main()
