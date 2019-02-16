from classifier_py_file import Node


def predict_result(tree, test_data):
    correct_predicted_counter = 0
    total_number_test_data = len(test_data)
    for data in test_data:
        real_class = data[len(data) - 1]
        predicted_class = tree.predict(data)
        if real_class == predicted_class:
            correct_predicted_counter += 1

    print('Total number of test data: ', total_number_test_data)
    print('Correct predicted: ', correct_predicted_counter)
    print('Percentage: ', (correct_predicted_counter / total_number_test_data) * 100)
