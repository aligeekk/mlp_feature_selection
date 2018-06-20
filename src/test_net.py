import torch
from torch.autograd import Variable


def test_net(model, test_df):
    # create tensors for test data
    test_array = test_df.as_matrix()
    x_test_array, y_test_array = test_array[:, 1:], test_array[:, 0]
    X_test, Y_test = Variable(torch.Tensor(x_test_array).float()), Variable(torch.Tensor(y_test_array).float())

    output = model(X_test)
    predicted = torch.round(output.squeeze())

    # calculate and print accuracy
    total_test = predicted.size(0)
    correct_test = sum(Y_test.data.numpy() == predicted.data.numpy())
    # print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))
    return 100 * correct_test / total_test
