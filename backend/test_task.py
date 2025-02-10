import requests

# Define the URL of the Flask server
url = "http://127.0.0.1:5000/task"  # Update if needed

# Define the payload with task data
payload = {
    "id": 1,  # Make sure this key is included
    "load": 10,  # Use "load" instead of complexity if your function expects "load"
    "complexity": 10,  # This will be added for completeness if needed
    "size": 5  # Similarly, ensure the size is part of the payload if it's required
}

# Send POST request with task data in JSON format
response = requests.post(url, json=payload)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    try:
        # Print the response JSON (assigned server details)
        print("Response JSON:", response.json())  # Expecting {"assigned_server": "some_ip"}
    except ValueError as e:
        print("Error parsing JSON response:", e)
else:
    print(f"Request failed with status code {response.status_code}")
