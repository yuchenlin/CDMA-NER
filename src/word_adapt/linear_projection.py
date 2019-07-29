import sys
import numpy

def load_w2v_model(model_file):
    model = {}
    with open(model_file, 'r') as fin:
        for line in fin:
            items = line.strip().split()
            if len(items) < 2:
                continue
            word = items[0]
            vector = numpy.array(map(float, items[1:]))
            model[word] = vector
    return model


if __name__ == '__main__':

    source_space = load_w2v_model(sys.argv[1])
    target_space = load_w2v_model(sys.argv[2])

    fp = open(sys.argv[3], 'r')
    target = set()
    source = set()
    interest_a = dict()
    interest_b = dict()
    for word in target_space.keys():
        interest_a[word] = 1
    for word in source_space.keys():
        interest_b[word] = 1

    used = dict()
    cnt = 0
    for line in fp.readlines():
        arr = line.split('\t')
        if arr[0] in used:
            continue
        if arr[1] in used:
            continue
        cnt += 1
        if (cnt > 5000):
            break
        if arr[0] not in interest_b:
            continue
        if arr[1] not in interest_a:
            continue
        source.add(arr[0])
        target.add(arr[1])
        used[arr[0]] = 1
        used[arr[1]] = 1

    source_matrix = numpy.array([target_space[word] for word in target])
    target_matrix = numpy.array([source_space[word] for word in source])
    source_matrix_with_c = numpy.vstack([source_matrix.T, numpy.ones(len(source_matrix))]).T
    result = numpy.linalg.lstsq(source_matrix_with_c, target_matrix)[0]
    with open(sys.argv[2], 'r') as fin, open(sys.argv[4], 'w') as fout:
        for line in fin:
            items = line.strip().split()
            if len(items) <= 300:
                continue
            else:
                target_vector = target_space[items[0]]
                transformed_target_vector = numpy.dot(numpy.append(target_vector, 1), result)
                fout.write(items[0] + ' ' + ' '.join(map(str, transformed_target_vector)) + '\n')