import numpy as np


def savetxt(filename, data, labels):
  with open(filename, 'w') as f:
    # labels = list(map(' '.join, np.eye(10, dtype=np.uint).astype(str)))
    for i, row in enumerate(data):
      row_str = row.astype(str)
      # import ipdb; ipdb.set_trace()
      label_str = ' '.join(labels[i].astype(str))
      feature_str = ' '.join(row_str)
      f.write('|labels {} |features {}\n'.format(label_str, feature_str))

def main():
  sz = 128
  data = (np.random.normal(size=(1, sz*sz))*255).astype(np.uint8)
  labels = (np.random.normal(size=(1, sz*sz))*255).astype(np.uint8)
  # labels = (np.random.randint(0, 10, size=(1, 1))).astype(np.uint8)
  # data_labels = np.hstack([data, labels])
  savetxt("data/mnist.txt", data, labels)

if __name__ == "__main__":
  main()
