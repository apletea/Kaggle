import pandas as pd
import pylab
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm

labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.99, random_state=0)

test_images[test_images>0]=1
train_images[train_images>0]=1
i=1
img=train_images.iloc[i].as_matrix().reshape((28,28))
plt.imshow(img,cmap='binary')
plt.title(train_labels.iloc[i])
plt.hist(train_images.iloc[i])
pylab.show()


clf = svm.SVC()
clf.fit(train_images, train_labels.values.ravel())
print(clf.score(test_images,test_labels))

test_data = pd.read_csv('test.csv')
test_data[test_data>0]=1
resulrt = clf.predict(test_data[:])

df = pd.DataFrame(resulrt)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv('results.csv', header=True)