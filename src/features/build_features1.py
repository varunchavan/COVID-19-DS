from sklearn import linear_model
import numpy as np
reg=linear_model.LinearRegression(fit_intercept=True)

def get_rate(in_array):
    y=np.array(in_array)
    x=np.arange(-1,2).reshape(-1,1)
    reg.fit(x,y)
    assert len(in_array)==3
    a=reg.coef_
    b=reg.intercept_
    return(b/a)
    


if __name__=='__main__':
    test_data=np.array([2,4,6])
    result=get_rate(test_data)
    print('the test slope is:' +str(result))



