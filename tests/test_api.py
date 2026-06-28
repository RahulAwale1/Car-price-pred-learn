from fastapi.testclient import TestClient
import pytest
from app.main import app

client = TestClient(app)

@pytest.fixture
def client1():
    with TestClient(app) as test_client:
        yield test_client

def test_health_check():
    response = client.get("/health")
    
    assert response.status_code == 200
    
    data = response.json()

    assert "status" in data
    assert "model_loaded" in data
    
def test_model_info():
    response = client.get("/model-info")
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "model_name" in data
    assert "model_version" in data
    assert "model_type" in data
    assert "input_schema_version" in data
    assert "currency" in data
    assert "model_loaded" in data
    

def test_predict_validate_input(client1):
    payload = {
        # Numerical features
        "symboling": 3,
        "wheelbase": 88.6,
        "carlength": 168.8,
        "carwidth": 64.1,
        "carheight": 48.8,
        "curbweight": 2548,
        "enginesize": 130,
        "boreratio": 3.47,
        "stroke": 2.68,
        "compressionratio": 9.0,
        "horsepower": 111,
        "peakrpm": 5000,
        "citympg": 7,
        "highwaympg": 10,

        # Categorical features
        "fueltype": "gas",
        "aspiration": "std",
        "doornumber": "two",
        "carbody": "convertible",
        "drivewheel": "rwd",
        "enginelocation": "front",
        "enginetype": "dohc",
        "cylindernumber": "four",
        "fuelsystem": "mpfi",
        "brand": "alfa-romero"
    }
    
    response = client1.post("/predict", json=payload)
    
    assert response.status_code == 200
    
    data = response.json()
    
    assert "predicted_price" in data
    assert "currency" in data
    
    assert isinstance(data["predicted_price"], float)
    assert data["currency"] == "CAD"

def test_predict_invalid_citympg_returns_422():
    payload = {
        # Numerical features
        "symboling": 3,
        "wheelbase": 88.6,
        "carlength": 168.8,
        "carwidth": 64.1,
        "carheight": 48.8,
        "curbweight": 2548,
        "enginesize": 130,
        "boreratio": 3.47,
        "stroke": 2.68,
        "compressionratio": 9.0,
        "horsepower": 111,
        "peakrpm": 5000,
        "citympg": -7,
        "highwaympg": 10,

        # Categorical features
        "fueltype": "gas",
        "aspiration": "std",
        "doornumber": "two",
        "carbody": "convertible",
        "drivewheel": "rwd",
        "enginelocation": "front",
        "enginetype": "dohc",
        "cylindernumber": "four",
        "fuelsystem": "mpfi",
        "brand": "alfa-romero"
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 422

def test_predict_extra_field_returns_422():
    payload = {
        # Numerical features
        "symboling": 3,
        "wheelbase": 88.6,
        "carlength": 168.8,
        "carwidth": 64.1,
        "carheight": 48.8,
        "curbweight": 2548,
        "enginesize": 130,
        "boreratio": 3.47,
        "stroke": 2.68,
        "compressionratio": 9.0,
        "horsepower": 111,
        "peakrpm": 5000,
        "citympg": 7,
        "highwaympg": 10,

        # Categorical features
        "fueltype": "gas",
        "aspiration": "std",
        "doornumber": "two",
        "carbody": "convertible",
        "drivewheel": "rwd",
        "enginelocation": "front",
        "enginetype": "dohc",
        "cylindernumber": "four",
        "fuelsystem": "mpfi",
        "brand": "alfa-romero",
        "extra": "dsad",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    

def test_predict_batch_valid_input(client1):
    payload = {
        "items": [
            {
                # Numerical features
                "symboling": 3,
                "wheelbase": 88.6,
                "carlength": 168.8,
                "carwidth": 64.1,
                "carheight": 48.8,
                "curbweight": 2548,
                "enginesize": 130,
                "boreratio": 3.47,
                "stroke": 2.68,
                "compressionratio": 9.0,
                "horsepower": 111,
                "peakrpm": 5000,
                "citympg": 7,
                "highwaympg": 10,

                # Categorical features
                "fueltype": "gas",
                "aspiration": "std",
                "doornumber": "two",
                "carbody": "convertible",
                "drivewheel": "rwd",
                "enginelocation": "front",
                "enginetype": "dohc",
                "cylindernumber": "four",
                "fuelsystem": "mpfi",
                "brand": "alfa-romero"
            },
            {
                # Numerical features
                "symboling": 3,
                "wheelbase": 88.6,
                "carlength": 168.8,
                "carwidth": 64.1,
                "carheight": 48.8,
                "curbweight": 2548,
                "enginesize": 130,
                "boreratio": 3.47,
                "stroke": 2.68,
                "compressionratio": 9.0,
                "horsepower": 111,
                "peakrpm": 5000,
                "citympg": 7,
                "highwaympg": 10,

                # Categorical features
                "fueltype": "gas",
                "aspiration": "std",
                "doornumber": "two",
                "carbody": "convertible",
                "drivewheel": "rwd",
                "enginelocation": "front",
                "enginetype": "dohc",
                "cylindernumber": "four",
                "fuelsystem": "mpfi",
                "brand": "alfa-romero"
            }
        ]
    }

    response = client1.post("/predict-batch", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert "count" in data
    assert "model_version" in data
    assert "request_id" in data

    assert data["count"] == 2
    assert len(data["predictions"]) == 2

    assert data["predictions"][0]["index"] == 0
    assert data["predictions"][1]["index"] == 1
    
def test_predict_batch_empty_returns_422():
    payload = {
        "items": []
    }

    response = client.post("/predict-batch", json=payload)

    assert response.status_code == 422