"""Safe OpenWeather client that never includes API credentials in errors."""
def validate_status(status_code):
    """Convert HTTP status codes to credential-safe user messages."""
    if status_code == 401:
        raise RuntimeError("OpenWeather rejected the API key. Check that the key is active and set correctly.")
    if status_code == 404:
        raise ValueError("OpenWeather could not find that city. Please choose another location.")
    if status_code < 200 or status_code >= 300:
        raise RuntimeError(f"The weather service returned an error ({status_code}). Please try again later.")


def get_five_day_forecast(city, api_key, timeout=12):
    import requests
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError("Could not contact the weather service. Please try again.") from exc
    validate_status(response.status_code)
    return response.json()
