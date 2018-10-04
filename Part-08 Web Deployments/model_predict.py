from sklearn.externals import joblib


def read_predict():
    model = joblib.load("model.pkl")
    # print(model)

    with open(r"1_3.txt", "r") as infile:
        test_contents = infile.read()

    with open(r".\\data\\aclImdb\\train\neg\\1_1.txt", "r") as infile:
        test_neg_contents = infile.read()

    with open(r".\\data\\aclImdb\\train\pos\\0_9.txt", "r") as infile:
        test_pos_contents = infile.read()

    predictions = model.predict([test_contents, test_neg_contents, test_pos_contents])
    return predictions


predictions = read_predict()
for p in predictions:
    print("pos" if p else "neg")
