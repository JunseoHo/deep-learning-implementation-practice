from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import model_selection
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # 0=ALL, 1=INFO, 2=WARN, 3=ERROR

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.6, random_state=42)

svm = svm.SVC(kernel='linear', C=1.0, gamma=0.5) # SVC는 Support Vector Classification의 준말.

"""

일단 위 코드에서 C는 오분류 허용치이다. 예를 들어 선형커널에서는 C가 커질수록 오분류를 적게 허용하고 (=하드마진), C가 작아질수록 오분류를 많이 허용한다(=소프트마진).
감마는 결정경계의 곡률을 의미하는데 너무 크면 다항식, 가우시안 RBF 커널과 같은 비선형 커널에서는 과적합의 위험이 높아지므로 주의해야한다.

"""

svm.fit(X_train, y_train)
predictions = svm.predict(X_test)
score = metrics.accuracy_score(y_test, predictions)

print("Accuracy: {0:f}".format(score))

