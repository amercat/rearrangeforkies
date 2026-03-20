import matplotlib.pyplot as plt

X_test_feature = X_test[:, 0]  
sorted_idx = np.argsort(X_test_feature)

plt.figure(figsize=(8,5))

plt.scatter(X_test_feature, y_test, label='Actual Y', s=40)

plt.scatter(X_test_feature, y_pred, label='Predicted Y', marker='x', s=40)

plt.plot(
    X_test_feature[sorted_idx],
    y_pred[sorted_idx],
    label='Regression Line',
    linewidth=2
)

plt.xlabel("Feature a")
plt.ylabel("Target Y")
plt.title("Linear Regression Model Visualization (Using Feature a)")
plt.legend()
plt.grid(True)
plt.show()
