
def predict_single(PetalLengthCm, PetalWidthCm , model):

    return model.predict([[PetalLengthCm, PetalWidthCm]])[0],model.predict_proba([[PetalLengthCm, PetalWidthCm]])[0] 

