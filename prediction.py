import joblib
def predict(data):
    clf = joblib.load(“rf_model.sav”)
return clf.predict(data)
from prediction import predict
if st.button(“Predict type of Iris”):
             result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
st.text(result[0])