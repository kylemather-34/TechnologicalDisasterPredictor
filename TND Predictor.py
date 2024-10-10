import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_excel('/Users/kylemather/Downloads/TND Training Data.xlsx')

# Prepare features and labels
X = df[['Disaster Group', 'Disaster Type', 'Disaster Subtype', 'Start Year', 'Start Month', 'Start Day', 'Latitude', 'Longitude']]
df['Disaster Occurred'] = 1
y = df['Disaster Occurred']

# Encode categorical features
X_categorical = X[['Disaster Group', 'Disaster Type', 'Disaster Subtype']]
encoder = OneHotEncoder(sparse_output=False)
X_encoded = pd.DataFrame(encoder.fit_transform(X_categorical), columns=encoder.get_feature_names_out())

# Combine encoded features with numerical features
X_combined = pd.concat([X_encoded, X[['Start Year', 'Start Month', 'Start Day', 'Latitude', 'Longitude']]], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict on test set (optional)
y_prediction = model.predict(X_test)
print(classification_report(y_test, y_prediction))

# Prediction input section
print("Welcome to the Technological Natural Disaster Predictor. This program allows you to enter a multitude of different categories."
      " Based on these, the model will predict whether or not a disaster will occur or not.")
print("Press E to enter a disaster or N to exit: ")
decision = input().upper()

while decision == 'E':
    # Get user input
    dGroup = input("Enter the Disaster Group: ")
    dType = input("Enter the Disaster Type: ")
    dSubtype = input("Enter the Disaster Subtype: ")
    sYear = int(input("Enter the Start Year: "))  # Directly converting to int
    sMonth = int(input("Enter the Start Month: "))  # Directly converting to int
    sDay = int(input("Enter the Start Day: "))  # Directly converting to int
    latitude = float(input("Enter the Latitude: "))  # Directly converting to float
    longitude = float(input("Enter the Longitude: "))  # Directly converting to float

    # Create DataFrame with user input
    new_data = pd.DataFrame({
        'Disaster Group': [dGroup],
        'Disaster Type': [dType],
        'Disaster Subtype': [dSubtype],
        'Start Year': [sYear],
        'Start Month': [sMonth],
        'Start Day': [sDay],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    # Encode categorical variables
    new_data_encoded = pd.DataFrame(
        encoder.transform(new_data[['Disaster Group', 'Disaster Type', 'Disaster Subtype']]),
        columns=encoder.get_feature_names_out(['Disaster Group', 'Disaster Type', 'Disaster Subtype'])
    )

    # Combine encoded features with numerical data
    new_data_combined = pd.concat(
        [new_data_encoded.reset_index(drop=True), new_data[['Start Year', 'Start Month', 'Start Day', 'Latitude', 'Longitude']].reset_index(drop=True)],
        axis=1
    )

    # Align new data with the model's feature names
    new_data_combined = new_data_combined.reindex(columns=model.feature_names_in_, fill_value=0)

    # Predict whether a disaster will occur
    new_prediction = model.predict(new_data_combined)
    print(f'Prediction: Disaster will occur (1) or not (0)? {new_prediction[0]}')

    # Ask the user if they want to continue or exit
    decision = input("Do you want to enter another disaster? (E to enter, any other key to exit): ").upper()
