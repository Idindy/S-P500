def predict(train, test, predictors, model):
    """
    This Allows you to make the prediction if stock goes up (1) or 
    goes down (0), with a threshhold of 60% could be increased to 
    increase accuracy.
    """
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined