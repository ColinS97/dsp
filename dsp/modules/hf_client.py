import requests

from dsp.modules.hf import HFModel


class HFModelClient(HFModel):
    def __init__(self, url, port=None, model="HFRemoteModel"):
        super().__init__(model=model, is_client=True)
        if port is None:
            self.url = url
        else:
            self.url = f"{url}:{port}"
        self.headers = {"Content-Type": "application/json; charset=utf-8"}

    def _generate(self, prompt, **kwargs):
        payload = {"prompt": prompt, "kwargs": kwargs}
        #print(f'#> kwargs: "{kwargs}" (type={type(kwargs)})')
        #print(f'#> payload: "{payload}" (type={type(payload)})')
        response = requests.post(self.url, json=payload, headers=self.headers)
        try:
            return response.json()
        except:
            print("Failed to parse JSON response:", response.text)
            raise Exception("Received invalid JSON response from server")