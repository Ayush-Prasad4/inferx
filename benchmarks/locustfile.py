from locust import HttpUser, task, between


class InferXUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict(self):
        self.client.post(
            "/predict",
            json={"text": "I really enjoyed this product."},
        )
