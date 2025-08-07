Run the Flask server to make predictions via HTTP requests:

python app.py
This will start the Flask application locally on http://127.0.0.1:5000.

Test the API:
You can now send POST requests to the /predict endpoint to get sentiment predictions from your model. Use Postman or curl to test the API.
For example, using curl:

curl -X POST -H "Content-Type: application/json" \
    -d '{"text": "I love this movie!"}' \
    http://127.0.0.1:5000/predict
You should get a JSON response with the model's prediction:

{"prediction": 1}

Deploying on a Production Server
If you wish to deploy the model in a production environment:

a. Use Docker to containerize the application.
b. Deploy the containerized app using Kubernetes or Docker Swarm for scalable and resilient deployment.