# %%
import tensorflow as tf
print("TensorFlow is working! Version:", tf.__version__)

# %%
import os

dataset_path = r"C:\Users\prave\OneDrive\Desktop\DDSMDataset"  
print("Files in dataset folder:", os.listdir(dataset_path))

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=15,  
    width_shift_range=0.05,  
    height_shift_range=0.05,
    shear_range=0.1, 
    zoom_range=0.1,  
    brightness_range=[0.8, 1.2],  
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)


train_generator = datagen.flow_from_directory(
    dataset_path,  
    target_size=(224, 224),
    batch_size=32, 
    class_mode='binary', 
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,  
    target_size=(224, 224),
    batch_size=32, 
    class_mode='binary', 
    subset='validation'
)

# %%
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.models import Model

def build_cnn_feature_extractor():
    input_layer = Input(shape=(224, 224, 3))

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalAveragePooling2D()(x) 
    x = Flatten()(x)  
    
    x = BatchNormalization()(x) 

    feature_extractor = Model(inputs=input_layer, outputs=x)
    return feature_extractor

# %%
cnn_model = build_cnn_feature_extractor()
cnn_model.summary()

# %%
def extract_features(generator, model):
    total_samples = generator.samples
    batch_size = generator.batch_size
    feature_shape = model.output_shape[1]

    features = np.zeros((total_samples, feature_shape))
    labels = np.zeros((total_samples,))

    i = 0
    for batch_images, batch_labels in generator:
        batch_size_actual = batch_images.shape[0]
        features[i:i+batch_size_actual] = model.predict(batch_images, verbose=0)
        labels[i:i+batch_size_actual] = batch_labels
        i += batch_size_actual
        if i >= total_samples:
            break  

    return features, labels

# %%
import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

X_train, y_train = extract_features(train_generator, cnn_model)
X_val, y_val = extract_features(val_generator, cnn_model)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_val_scaled = scaler.transform(X_val)

np.save("X_train.npy", X_train_scaled)
np.save("y_train.npy", y_train)
np.save("X_val.npy", X_val_scaled)
np.save("y_val.npy", y_val)

print("✅ Feature extraction & scaling complete. Extracted features saved!")

# %%
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)


best_svm = grid_search.best_estimator_
print("🔥 Best SVM Parameters:", grid_search.best_params_)


y_pred = best_svm.predict(X_val_scaled)


accuracy = accuracy_score(y_val, y_pred)
print(f"✅ Best SVM Accuracy: {accuracy:.4f}")


print("\nClassification Report:\n", classification_report(y_val, y_pred))

# %%
import joblib  

joblib.dump(best_svm, "best_svm_model.pkl")  


joblib.dump(scaler, "scaler.pkl")  

cnn_model.save_weights("cnn_feature_extractor.weights.h5")  

print("🔥 Model, Scaler & Weights saved successfully!")

# %%
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib  

def predict_cancer(img_path):
    
    best_svm = joblib.load("best_svm_model.pkl")  
    scaler = joblib.load("scaler.pkl")  

   
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    feature = cnn_model.predict(img_array)  
    feature_scaled = scaler.transform(feature)  

    
    prediction = best_svm.predict(feature_scaled)  
    result = "Cancer Detected 🚨" if prediction[0] == 1 else "No Cancer ✅"

    print(f"🔬 Prediction: {result}")

# %%
import numpy as np

X_train_scaled = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# %%
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# %%
def svm_objective(params):
    C, gamma = params  
    svm = SVC(C=C, gamma=gamma, kernel='rbf')
    score = cross_val_score(svm, X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    return score

# %%
def firefly_algorithm(objective_func, param_bounds, n_fireflies=10, max_iter=20, alpha=0.2, beta0=1, gamma=1):
    np.random.seed(42)
    
    # Step 1: Initialize fireflies randomly within bounds
    fireflies = np.random.uniform(
        [b[0] for b in param_bounds], 
        [b[1] for b in param_bounds], 
        (n_fireflies, len(param_bounds))
    )
    
    # Step 2: Evaluate fitness for all fireflies
    fitness = np.array([objective_func(f) for f in fireflies])
    
    # Step 3: Iteratively update fireflies
    for t in range(max_iter):
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] > fitness[i]:  # Move firefly i towards brighter firefly j
                    r = np.linalg.norm(fireflies[i] - fireflies[j])  # Distance between fireflies
                    beta = beta0 * np.exp(-gamma * r**2)  # Attraction function
                    
                    # Update firefly position
                    fireflies[i] += beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(len(param_bounds)) - 0.5)
                    
                    # Clip values to ensure they remain in valid range
                    fireflies[i] = np.clip(fireflies[i], [b[0] for b in param_bounds], [b[1] for b in param_bounds])
                    
                    # Recalculate fitness
                    fitness[i] = objective_func(fireflies[i])
    
    # Step 4: Return the best solution
    best_idx = np.argmax(fitness)
    return fireflies[best_idx]

# %%
def firefly_algorithm_fast(objective_func, param_bounds, n_fireflies=5, max_iter=10, alpha=0.2, beta0=1, gamma=1, verbose=True):
    np.random.seed(42)
    dim = len(param_bounds)

    # Initialize fireflies randomly within bounds
    fireflies = np.random.uniform(
        [b[0] for b in param_bounds],
        [b[1] for b in param_bounds],
        size=(n_fireflies, dim)
    )

    # Evaluate initial fitness
    fitness = np.array([objective_func(f) for f in fireflies])

    # Cache to avoid re-evaluating same positions
    cache = {tuple(fireflies[i]): fitness[i] for i in range(n_fireflies)}

    for t in range(max_iter):
        if verbose:
            print(f"Iteration {t+1}/{max_iter}, Best Fitness: {np.max(fitness):.4f}")
        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if fitness[j] > fitness[i]:
                    r = np.linalg.norm(fireflies[i] - fireflies[j])
                    beta = beta0 * np.exp(-gamma * r ** 2)

                    # Movement
                    step = beta * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(dim) - 0.5)
                    fireflies[i] += step
                    fireflies[i] = np.clip(fireflies[i], [b[0] for b in param_bounds], [b[1] for b in param_bounds])

                    # Check cache before computing fitness
                    key = tuple(np.round(fireflies[i], decimals=5))
                    if key in cache:
                        fitness[i] = cache[key]
                    else:
                        fitness[i] = objective_func(fireflies[i])
                        cache[key] = fitness[i]

    best_idx = np.argmax(fitness)
    return fireflies[best_idx]


# %%
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample

def svm_objective(params, X, y):
    C, gamma = params
    # Optional: Use a small random sample for speed
    X_sample, y_sample = resample(X, y, n_samples=300, random_state=42)

    model = SVC(C=C, gamma=gamma)
    scores = cross_val_score(model, X_sample, y_sample, cv=2)  # Keep CV folds low
    return scores.mean()


# %%
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target


# %%
best_params = firefly_algorithm_fast(
    lambda p: svm_objective(p, X, y),
    param_bounds=[(0.1, 10), (0.001, 0.1)],
    n_fireflies=5,
    max_iter=10
)
print("Best Parameters (C, gamma):", best_params)


# %%
best_params = firefly_algorithm_fast(
    lambda p: svm_objective(p, X, y),
    param_bounds=[(0.1, 10), (0.001, 0.1)],
    n_fireflies=5,
    max_iter=10
)


# %%
param_bounds = [(0.1, 100), (0.001, 1)]  # (Min, Max) for C and gamma
firefly_algorithm(svm_objective, param_bounds, alpha=0.5)

# %%
best_params_fa = firefly_algorithm(
    lambda p: svm_objective(p, X, y),  # wrapping with lambda to pass X and y
    param_bounds,
    n_fireflies=5,
    max_iter=10,
    alpha=0.5
)
print("🔥 Best Parameters from Firefly Algorithm:", best_params_fa)


# %%
best_params_fa = firefly_algorithm(svm_objective, param_bounds, n_fireflies=5, max_iter=10, alpha=0.5)
print("🔥 Best Parameters from Firefly Algorithm:", best_params_fa)

# %%
best_C, best_gamma = best_params_fa  # Extract optimized values

best_svm_fa = SVC(C=best_C, gamma=best_gamma, kernel='rbf')  
best_svm_fa.fit(X_train_scaled, y_train)  

y_pred_fa = best_svm_fa.predict(X_val_scaled)  

from sklearn.metrics import accuracy_score, classification_report
accuracy_fa = accuracy_score(y_val, y_pred_fa)

print(f"🔥 SVM Accuracy (FA Optimized): {accuracy_fa:.4f}")  
print("\nClassification Report:\n", classification_report(y_val, y_pred_fa))

# %%
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# %%
def svm_objective(params):
    C, gamma = params  # Extract SVM hyperparameters
    svm = SVC(C=C, gamma=gamma, kernel='rbf')
    score = cross_val_score(svm, X_train_scaled, y_train, cv=3, scoring='accuracy').mean()  # 3-fold CV for speed
    return score

# %%
def flower_pollination_algorithm(objective_func, param_bounds, n_flowers=10, max_iter=20, p=0.8):
    np.random.seed(42)

    # Step 1: Initialize flower positions randomly
    flowers = np.random.uniform(
        [b[0] for b in param_bounds], 
        [b[1] for b in param_bounds], 
        (n_flowers, len(param_bounds))
    )

    # Step 2: Evaluate fitness of all flowers
    fitness = np.array([objective_func(f) for f in flowers])

    # Find the best flower (global best solution)
    best_idx = np.argmax(fitness)
    best_flower = flowers[best_idx]

    # Step 3: Iterate through generations
    for t in range(max_iter):
        for i in range(n_flowers):
            if np.random.rand() < p:  # Global pollination (exploration)
                L = np.random.normal(0, 1, len(param_bounds))  # Lévy flight step
                new_flower = flowers[i] + L * (flowers[i] - best_flower)
            else:  # Local pollination (exploitation)
                j, k = np.random.randint(0, n_flowers, 2)
                new_flower = flowers[i] + np.random.rand() * (flowers[j] - flowers[k])

            # Clip to ensure within bounds
            new_flower = np.clip(new_flower, [b[0] for b in param_bounds], [b[1] for b in param_bounds])

            # Evaluate new fitness
            new_fitness = objective_func(new_flower)

            # Update if the new position is better
            if new_fitness > fitness[i]:
                flowers[i] = new_flower
                fitness[i] = new_fitness

                # Update global best if necessary
                if new_fitness > fitness[best_idx]:
                    best_idx = i
                    best_flower = new_flower

    return best_flower

# %%
param_bounds = [(0.1, 100), (0.001, 1)]  # Bounds for C and gamma

# %%
best_params_fpa = flower_pollination_algorithm(svm_objective, param_bounds)
print("🌸 Best Parameters from Flower Pollination Algorithm:", best_params_fpa)

# %%
import numpy as np
from sklearn.utils import resample
from scipy.stats import levy

def flower_pollination_algorithm(objective_func, param_bounds, n_flowers=5, max_iter=10, p=0.8, verbose=True):
    np.random.seed(42)
    dim = len(param_bounds)

    # Step 1: Initialize flowers
    flowers = np.random.uniform(
        [b[0] for b in param_bounds],
        [b[1] for b in param_bounds],
        (n_flowers, dim)
    )

    # Step 2: Evaluate fitness
    fitness = np.array([objective_func(f) for f in flowers])
    best = flowers[np.argmax(fitness)]
    best_fitness = np.max(fitness)

    for t in range(max_iter):
        if verbose:
            print(f"Iteration {t+1}/{max_iter}, Best Fitness: {best_fitness:.4f}")
        for i in range(n_flowers):
            if np.random.rand() < p:
                # Global pollination with Lévy flight
                step = levy.rvs(size=dim)
                flowers[i] += step * (flowers[i] - best)
            else:
                # Local pollination
                j, k = np.random.choice(n_flowers, 2, replace=False)
                flowers[i] += np.random.rand() * (flowers[j] - flowers[k])

            # Bound the values
            flowers[i] = np.clip(flowers[i], [b[0] for b in param_bounds], [b[1] for b in param_bounds])

            # Re-evaluate fitness
            current_fit = objective_func(flowers[i])
            if current_fit > fitness[i]:
                fitness[i] = current_fit

            # Update global best
            if current_fit > best_fitness:
                best = flowers[i]
                best_fitness = current_fit

    return best


# %%
def svm_objective(params, X, y):
    C, gamma = params
    X_sample, y_sample = resample(X, y, n_samples=300, random_state=42)
    model = SVC(C=C, gamma=gamma)
    return cross_val_score(model, X_sample, y_sample, cv=2).mean()


# %%
best_params_fpa = flower_pollination_algorithm(
    lambda p: svm_objective(p, X, y),  # Wrap with lambda
    param_bounds=[(0.1, 10), (0.001, 0.1)],
    n_flowers=5,
    max_iter=10
)
print("🌸 Best Parameters from Flower Pollination Algorithm:", best_params_fpa)


# %%
from sklearn.svm import SVC

best_C = 100.0
best_gamma = 0.0037

svm_classifier = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
svm_classifier.fit(X_train, y_train)

# %%
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data        # Features
y = data.target      # Labels

# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Assuming X and y are your features and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use best parameters found by Flower Pollination Algorithm
best_C = best_params_fpa[0]
best_gamma = best_params_fpa[1]

# Initialize and train SVM
svm_classifier = SVC(C=best_C, gamma=best_gamma, kernel='rbf')
svm_classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ SVM Accuracy with optimized parameters: {accuracy * 100:.2f}%")

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("🔍 Confusion Matrix:\n", cm)

# Classification Report (includes precision, recall, f1-score)
report = classification_report(y_test, y_pred)
print("\n📊 Classification Report:\n", report)

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA (optional: change n_components to test different settings)
pca = PCA(n_components=30)  # You can reduce to fewer like 20 if needed
X_pca = pca.fit_transform(X_scaled)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Step 5: Train an SVM classifier
svm = SVC(C=1, gamma=0.01, kernel='rbf')  # You can plug in your optimized params
svm.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ SVM Accuracy after PCA: {accuracy * 100:.2f}%")

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # This can boost features from 30 to 465

# Step 3: Scale the polynomial features
scaler_poly = MinMaxScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly)

# Step 4: Train/Test split
X_train_poly, X_test_poly, y_train, y_test = train_test_split(
    X_poly_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train SVM on polynomial features
svm_poly = SVC(C=1, gamma=0.01, kernel='rbf')  # You can optimize this further
svm_poly.fit(X_train_poly, y_train)
y_pred_poly = svm_poly.predict(X_test_poly)
accuracy_poly = accuracy_score(y_test, y_pred_poly)

# Step 6: PCA (on original 30 features for comparison)
scaler_pca = MinMaxScaler()
X_scaled = scaler_pca.fit_transform(X)

pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca, y, test_size=0.2, random_state=42)

svm_pca = SVC(C=1, gamma=0.01, kernel='rbf')
svm_pca.fit(X_train_pca, y_train)
y_pred_pca = svm_pca.predict(X_test_pca)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

# Step 7: Print results
print(f"📈 SVM Accuracy with Polynomial Features: {accuracy_poly * 100:.2f}%")
print(f"📉 SVM Accuracy with PCA Features: {accuracy_pca * 100:.2f}%")

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Expand features with PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)  # From 30 features to ~465

# Step 3: Scale the features
scaler = MinMaxScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_scaled, y, test_size=0.2, random_state=42)

# Step 5: Set up SVM and hyperparameter grid
svm = SVC(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

# Step 6: GridSearchCV
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Step 7: Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 8: Output results
print("✅ Best SVM Parameters:", grid_search.best_params_)
print(f"🎯 Accuracy with Polynomial Features + Grid Search: {accuracy * 100:.2f}%")

# %%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Polynomial feature expansion
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Step 3: Feature scaling
scaler = MinMaxScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_scaled, y, test_size=0.2, random_state=42)

# Step 5: SVM and GridSearchCV
svm = SVC(kernel='rbf')
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Step 6: Evaluate best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 7: Results
print("✅ Best SVM Parameters:", grid_search.best_params_)
print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")

# Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🧠 Confusion Matrix")
plt.show()

# Step 9: Classification Report
print("📝 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set your dataset path
dataset_path = r"C:\Users\prave\OneDrive\Desktop\DDSMDataset"

# Create data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% train, 20% validation
)

# Training generator
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation generator
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build a simple CNN model for classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train for 50 epochs
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50
)


# %%
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Accuracy over 50 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()


# %%
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("CNN Loss over 50 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build a simple CNN model for classification
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train for 100 epochs
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100
)


# %%
import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Accuracy over 100 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("CNN Loss over 100 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()


# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set your dataset path
dataset_path = r"C:\Users\prave\OneDrive\Desktop\DDSMDataset"

# Create data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 80% train, 20% validation
)

# Training generator
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Validation generator
val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ✅ Train for 150 epochs
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=75
)


# %%
import matplotlib.pyplot as plt

# Accuracy plot
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("CNN Accuracy over 150 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("CNN Loss over 150 Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()



