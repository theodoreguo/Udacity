def dErrors(X, y, y_hat):
    DErrorsDx1 = [X[i][0]*(y[i]-y_hat[i]) for i in range(len(y))]
    DErrorsDx2 = [X[i][1]*(y[i]-y_hat[i]) for i in range(len(y))]
    DErrorsDb = [y[i]-y_hat[i] for i in range(len(y))]
    return DErrorsDx1, DErrorsDx2, DErrorsDb

def gradientDescentStep(X, y, W, b, learn_rate = 0.01):
    y_hat = prediction(X,W,b)
    errors = error_vector(y, y_hat)
    derivErrors = dErrors(X, y, y_hat)
    W[0] += sum(derivErrors[0])*learn_rate
    W[1] += sum(derivErrors[1])*learn_rate
    b += sum(derivErrors[2])*learn_rate
    return W, b, sum(errors)
