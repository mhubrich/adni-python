import csv


def write_prediction(fname, predictions, filenames):
    with open(fname, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in xrange(0, len(predictions)):
            tmp = [filenames[i]]
            for p in predictions[i]:
                tmp.append(p)
            writer.writerow(tmp)
    print('%d predicitons written.' % (len(predictions)))
