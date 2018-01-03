import pandas
import cv2
import numpy as np
import sys
sys.path.append('/home/please/work/download_source/caffe/python/')
import caffe


caffe.set_mode_gpu()

def load_net():
    nnet = caffe.Net('deploy.prototxt',
                     'vgg19_cvgj_iter_200000.caffemodel',
                     caffe.TEST)
    return nnet

def load_mean():
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    with open('mean.binaryproto') as f:
        mean_blob.ParseFromString(f.read())
    mean_array =  np.asarray(mean_blob.data, dtype=np.float32).reshape((mean_blob.channels, mean_blob.height, mean_blob.width))
    return mean_array

def load_transformer(mean, net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', mean)
    transformer.set_transpose('data', (2, 0, 1))
    return transformer


net = load_net()
mean = load_mean()
transformer = load_transformer(mean,net)
ans = pandas.read_csv('sample_submission.csv')
labels = pandas.read_csv('breeds.csv')


for i in xrange(0,len(ans)):
    path = './test/'+ans['id'][i] + '.jpg'
    img = caffe.io.load_image(path)
    net.blobs['data'].data[...] = transformer.preprocess('data',img)
    out = net.forward()
    prob = net.blobs['prob'].data[0]
    for j in xrange(0,len(prob)):
        ans[labels['name'][j]][i] = prob[j]
    print i
    # print net.blobs['prob'].data[0]



ans.to_csv('relabeled.csv',index=None)
