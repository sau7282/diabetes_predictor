# Diabetes Prediction API

This API uses a trained machine learning model to predict diabetes based on user input. It supports both real-time and batch predictions using the Pima Indians Diabetes dataset.

## Features
- **Fetches dataset**: Downloads the Pima Indians Diabetes dataset from an online source.
- **Trains and saves a model**: Uses RandomForestClassifier to train and save the model.
- **Real-time prediction endpoint**: Accepts JSON input and returns predictions.
- **Batch prediction endpoint**: Accepts CSV file input and returns predictions.
- **Stores results**: Saves predictions in CSV files for reference.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sau7282/diabetes_predictor.git
   ```
2. Navigate to the project directory:
   ```bash
   cd diabetes_predictor
   ```
3. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### **Run the API**
```bash
python app.py
```
The API will be available at `http://127.0.0.1:5000/`

### **Endpoints**
#### **1. Real-Time Prediction**
- **URL**: `/predict`
- **Method**: `POST`
- **Request Body (JSON)**:
  ```json
  {
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 85,
    "BMI": 28.5,
    "Diabetespedigreefunction": 0.5,
    "age": 30
  }
  ```
- **Response**:
  ```json
  {"predictions": 1}
  ```

#### **2. Batch Prediction**
- **URL**: `/batch-predict`
- **Method**: `POST`
- **Request Body**: CSV file upload
- **Response**:
  ```json
  {"message": "Batch predictions are saved", "output_file": "data/batch_prediction.csv"}
  ```

## Using Postman for Testing
You can use [Postman](https://www.postman.com/) to test the API endpoints:
1. Open Postman and create a new request.
2. Select `POST` as the request type.
3. Enter the API endpoint (`http://127.0.0.1:5000/predict` for real-time or `http://127.0.0.1:5000/batch-predict` for batch processing).
4. For real-time prediction, go to the `Body` tab, select `raw`, choose `JSON`, and enter the input data.
5. For batch prediction, go to the `Body` tab, choose `form-data`, and upload the CSV file.
6. Click `Send` and view the response.

## Model Training
The model is trained using the Pima Indians Diabetes dataset. If no trained model exists, it is automatically trained and saved.

## Output Files
- **Real-time predictions**: `data/real_time_prediction.csv`
- **Batch predictions**: `data/batch_prediction.csv`
- **Online dataset**: `data/online_data.csv`
- **Trained model**: `model.pkl`

## Contributing
Feel free to open issues or submit pull requests to improve this project.

## License
This project is licensed under the MIT License.
