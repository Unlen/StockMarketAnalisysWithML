
# %%
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# evaluate the model
yhat = model.predict(X_test)
# evaluate predictions
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# %%
# Create a new random forest classifier
rf = RandomForestClassifier()

# Dictionary of all values we want to test for n_estimators
params_rf = {'n_estimators': [110, 130, 140, 150, 160, 180, 200]}

# Use gridsearch to test all values for n_estimators
rf_gs = GridSearchCV(rf, params_rf, cv=5)

# Fit model to training data
rf_gs.fit(X_train, y_train)

# Save best model
rf_best = rf_gs.best_estimator_

# Check best n_estimators value
print(rf_gs.best_params_)


# %%
prediction = rf_best.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

# %%
knn = KNeighborsClassifier()
# Create a dictionary of all values we want to test for n_neighbors
params_knn = {'n_neighbors': np.arange(1, 25)}

# Use gridsearch to test all values for n_neighbors
knn_gs = GridSearchCV(knn, params_knn, cv=5)

# Fit model to training data
knn_gs.fit(X_train, y_train)

# Save best model
knn_best = knn_gs.best_estimator_

# Check best n_neigbors value
print(knn_gs.best_params_)

prediction = knn_best.predict(X_test)

# %%
prediction = rf_best.predict(X_test)
print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))